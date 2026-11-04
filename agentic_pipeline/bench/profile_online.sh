#!/usr/bin/env bash
# Open a torch-profiler window on a running server, wait, then close it.
#
#   bench/profile_online.sh <vllm|sglang> <out_dir> <seconds>
#
# vLLM traces land in <serve RUN_DIR>/torch_profile, not in out_dir.
set -euo pipefail

usage() { echo "usage: $0 <vllm|sglang> <out_dir> <seconds>" >&2; exit 2; }
[[ $# -eq 3 ]] || usage
ENGINE="$1"
OUT_DIR="$2"
SECONDS_WINDOW="$3"
SGLANG_STAGE_STEPS=20

case "${ENGINE}" in
  vllm) PORT=8000 ;;
  sglang) PORT=30000 ;;
  *) usage ;;
esac
BASE="http://127.0.0.1:${PORT}"
mkdir -p "${OUT_DIR}"

if [[ "${ENGINE}" == "sglang" ]]; then
  curl -sf -X POST "${BASE}/start_profile" -H 'Content-Type: application/json' \
    -d "{\"output_dir\":\"${OUT_DIR}\",\"profile_by_stage\":true,\"activities\":[\"CPU\",\"GPU\"],\"with_stack\":true,\"num_steps\":${SGLANG_STAGE_STEPS}}"
else
  curl -sf -X POST "${BASE}/start_profile"
fi
sleep "${SECONDS_WINDOW}"
# SGLang per-stage profiling stops itself after SGLANG_STAGE_STEPS prefill and decode steps.
if [[ "${ENGINE}" == "vllm" ]]; then curl -sf -X POST "${BASE}/stop_profile"; fi
