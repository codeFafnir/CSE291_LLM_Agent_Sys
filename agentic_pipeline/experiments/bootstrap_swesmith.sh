#!/usr/bin/env bash
# One-time EC2 setup for the SWE-smith-cpp agent experiments. Run after bootstrap_ec2.sh. Safe to re-run.
# Run from the repo root on the box: bash experiments/bootstrap_swesmith.sh
# Builds the sweagent and swesmith envs, adds plotly to the vllm env, picks instances, pulls the 3 images.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA=/data
SWE_AGENT_COMMIT=3ea751c087f32b16e039a2233dd6eefecef325d5
SWE_SMITH_COMMIT=9b74ac08118a85c39c356802f7961893af73e07f
SWEBENCH_VERSION=4.1.0
DATASETS_VERSION=5.2.0
DOCKER_PY_VERSION=7.2.0
LITELLM_VERSION=1.110.0
IMAGE_KEYS=(openh264 srt thrift)
MAX_MODEL_LEN=30000
INSTANCE_COUNT=250
SEED=0
INSTANCES_DIR="${DATA}/swesmith"

export PATH="${HOME}/.local/bin:${PATH}"
export HF_HOME="${DATA}/hf" HUGGINGFACE_HUB_CACHE="${DATA}/hf/hub" TMPDIR="${DATA}/tmp" UV_CACHE_DIR="${DATA}/uv-cache"

clone_at() {
  local url="$1" dir="$2" commit="$3"
  [[ -d "${dir}/.git" ]] || git clone "${url}" "${dir}"
  git -C "${dir}" fetch --quiet origin "${commit}"
  git -C "${dir}" checkout --quiet "${commit}"
}

make_env() {
  local env_dir="$1"
  [[ -x "${env_dir}/bin/python" ]] || uv venv --python 3.12 "${env_dir}"
}

docker info > /dev/null
mkdir -p "${INSTANCES_DIR}"

clone_at https://github.com/SWE-agent/SWE-agent.git "${DATA}/SWE-agent" "${SWE_AGENT_COMMIT}"
make_env "${DATA}/envs/sweagent"
VIRTUAL_ENV="${DATA}/envs/sweagent" uv pip install -e "${DATA}/SWE-agent"
VIRTUAL_ENV="${DATA}/envs/sweagent" uv pip install "litellm==${LITELLM_VERSION}"

clone_at https://github.com/SWE-bench/SWE-smith.git "${DATA}/SWE-smith" "${SWE_SMITH_COMMIT}"
make_env "${DATA}/envs/swesmith"
VIRTUAL_ENV="${DATA}/envs/swesmith" uv pip install -e "${DATA}/SWE-smith[validate]" "swebench==${SWEBENCH_VERSION}" "datasets==${DATASETS_VERSION}" "docker==${DOCKER_PY_VERSION}"

# The orchestrator, proxy and report run in the vllm env (aiohttp and pydantic already there).
VIRTUAL_ENV="${DATA}/envs/vllm" uv pip install plotly

"${DATA}/envs/vllm/bin/python" "${REPO_ROOT}/swesmith/select_instances.py" \
  --images "${IMAGE_KEYS[@]}" --max-model-len "${MAX_MODEL_LEN}" \
  --count "${INSTANCE_COUNT}" --seed "${SEED}" --out-dir "${INSTANCES_DIR}"

for key in "${IMAGE_KEYS[@]}"; do
  image="$(head -n 1 "${INSTANCES_DIR}/instances_${key}.jsonl" | "${DATA}/envs/vllm/bin/python" -c 'import json,sys; print(json.loads(sys.stdin.readline())["image_name"])')"
  docker pull --platform linux/amd64 "${image}"
done

echo "swesmith bootstrap done"
