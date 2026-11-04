#!/usr/bin/env bash
# One-time EC2 setup on the Deep Learning Base GPU AMI (Ubuntu 22.04). Safe to re-run.
# Run from the repo root on the box: bash experiments/bootstrap_ec2.sh
# Mounts the 300 GB EBS volume at /data, builds the vllm and sglang envs, installs Harbor,
# downloads models and picks the EASY-30 ids.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA=/data
DATA_BYTES=$((300 * 1024 * 1024 * 1024))
FORK_BRANCH=cse291-agent-evict
BENCH_PKGS=(prometheus_client pydantic nvidia-ml-py datasets transformers)
OTEL_PKGS=(opentelemetry-sdk opentelemetry-api opentelemetry-exporter-otlp opentelemetry-semantic-conventions-ai)
MODELS=(Qwen/Qwen3-14B-AWQ RedHatAI/Qwen3-14B-speculator.eagle3 thoughtworks/Qwen3-14B-Eagle3)

export PATH="${HOME}/.local/bin:${PATH}"

mount_data() {
  mountpoint -q "${DATA}" && return 0
  local dev
  dev="$(lsblk -dnbo NAME,SIZE | awk -v size="${DATA_BYTES}" '$2 == size {print "/dev/" $1}' | head -n 1)"
  if [[ -z "${dev}" ]]; then echo "no ${DATA_BYTES}-byte volume found" >&2; exit 1; fi
  if ! sudo blkid "${dev}" > /dev/null 2>&1; then sudo mkfs.ext4 -L cse291data "${dev}"; fi
  sudo mkdir -p "${DATA}"
  sudo mount "${dev}" "${DATA}"
  sudo chown "$(id -u):$(id -g)" "${DATA}"
  grep -q "LABEL=cse291data" /etc/fstab || echo "LABEL=cse291data ${DATA} ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab > /dev/null
}

write_env() {
  grep -q "# cse291 env" ~/.bashrc && return 0
  cat >> ~/.bashrc <<EOF
# cse291 env
export HF_HOME=${DATA}/hf
export HUGGINGFACE_HUB_CACHE=${DATA}/hf/hub
export TMPDIR=${DATA}/tmp
export UV_CACHE_DIR=${DATA}/uv-cache
export PATH=\$HOME/.local/bin:\$PATH
EOF
}

fetch_forks() {
  local name
  for name in vllm sglang; do
    local dir="${REPO_ROOT}/../third_party/${name}"
    if [[ ! -d "${dir}/.git" ]]; then
      rm -rf "${dir}"
      git clone --branch "${FORK_BRANCH}" "https://github.com/codeFafnir/${name}.git" "${dir}"
    fi
  done
}

build_env() {
  local name="$1"
  local env_dir="${DATA}/envs/${name}"
  [[ -x "${env_dir}/bin/python" ]] || uv venv --python 3.12 "${env_dir}"
  case "${name}" in
    vllm) VIRTUAL_ENV="${env_dir}" VLLM_USE_PRECOMPILED=1 uv pip install -e "${REPO_ROOT}/../third_party/vllm" --torch-backend=auto ;;
    sglang) VIRTUAL_ENV="${env_dir}" SGLANG_BUILD_RUST_EXTS=none uv pip install --prerelease=allow -e "${REPO_ROOT}/../third_party/sglang/python" ;;
  esac
  VIRTUAL_ENV="${env_dir}" uv pip install "${BENCH_PKGS[@]}"
}

# OTLP span export for vLLM request traces. Refuses to run if it would change a pinned core package.
install_otel() {
  local env_dir="${DATA}/envs/vllm" plan
  plan="$(VIRTUAL_ENV="${env_dir}" uv pip install --dry-run "${OTEL_PKGS[@]}" 2>&1)"
  if grep -E '^\s*- (vllm|torch|transformers|protobuf)==' <<< "${plan}"; then
    echo "installing OTel would change a pinned package (lines above); aborting" >&2
    exit 1
  fi
  VIRTUAL_ENV="${env_dir}" uv pip install "${OTEL_PKGS[@]}"
}

mount_data
mkdir -p "${DATA}/tmp" "${DATA}/hf" "${DATA}/cse291-results"
write_env
export HF_HOME="${DATA}/hf" HUGGINGFACE_HUB_CACHE="${DATA}/hf/hub" TMPDIR="${DATA}/tmp" UV_CACHE_DIR="${DATA}/uv-cache"

sudo apt-get update -y
sudo apt-get install -y tmux git curl
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

fetch_forks
build_env vllm
install_otel
build_env sglang

uv tool install --force 'harbor>=0.22'
uv tool install --force huggingface_hub
uv tool install --force py-spy
for model in "${MODELS[@]}"; do hf download "${model}"; done

bash "${REPO_ROOT}/experiments/prepare_sglang_draft.sh"

[[ -d "${DATA}/SWE-bench_Pro-os" ]] || git clone https://github.com/scaleapi/SWE-bench_Pro-os.git "${DATA}/SWE-bench_Pro-os"
"${DATA}/envs/vllm/bin/python" "${REPO_ROOT}/experiments/select_easy30.py" \
  --tasks-dir "${DATA}/SWE-bench_Pro-os/v2/tasks" --max-model-len 12288 --count 30 \
  --out "${REPO_ROOT}/experiments/easy30_ids.txt" --copy-to "${DATA}/SWE-bench_Pro-os/v2/easy30_ids.txt"

echo "bootstrap done"
