#!/usr/bin/env bash
# One-time EC2 setup for the SWE-bench Verified agent smoke. Run after bootstrap_ec2.sh and bootstrap_swesmith.sh
# (they build /data/envs/vllm and /data/envs/sweagent). Safe to re-run.
# Run from the repo root on the box: bash experiments/bootstrap_swebench.sh
# Writes the 200-task slice and the smoke set, pulls the smoke images, and checks the first one resets to its base commit.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA=/data
VLLM_PY="${DATA}/envs/vllm/bin/python"
INSTANCES_DIR="${DATA}/swebench"
FILL_COUNT=6
SMOKE_COUNT=20
SEED=291
SMOKE_FILE="${INSTANCES_DIR}/instances_smoke${SMOKE_COUNT}.jsonl"

export PATH="${HOME}/.local/bin:${PATH}"
export HF_HOME="${DATA}/hf" HUGGINGFACE_HUB_CACHE="${DATA}/hf/hub" TMPDIR="${DATA}/tmp"

docker info > /dev/null
mkdir -p "${INSTANCES_DIR}"

"${VLLM_PY}" "${REPO_ROOT}/swebench/select_verified.py" \
  --fill-count "${FILL_COUNT}" --smoke-count "${SMOKE_COUNT}" --seed "${SEED}" --out-dir "${INSTANCES_DIR}"

# One "<image> <base_commit>" line per smoke instance.
mapfile -t PAIRS < <("${VLLM_PY}" -c '
import json, sys
for line in open(sys.argv[1]):
    row = json.loads(line)
    print(row["image_name"], row["base_commit"])
' "${SMOKE_FILE}")

for pair in "${PAIRS[@]}"; do
  docker pull --platform linux/amd64 "${pair%% *}"
done

# Same sequence SWE-agent runs on reset (git fetch, git checkout <base_commit>); fail unless HEAD lands on that SHA.
image="${PAIRS[0]%% *}"
commit="${PAIRS[0]##* }"
head="$(docker run --rm --platform linux/amd64 --entrypoint bash "${image}" -c \
  "cd /testbed && git fetch && git checkout -q ${commit} && git rev-parse HEAD && /opt/miniconda3/envs/testbed/bin/python --version >&2" | tail -n 1)"
if [[ "${head}" != "${commit}" ]]; then
  echo "${image}: HEAD is ${head}, expected ${commit}" >&2
  exit 1
fi

echo "swebench bootstrap done: ${#PAIRS[@]} images, ${image} reset to ${commit}"
