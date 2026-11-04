#!/usr/bin/env bash
# Run a serve script under nsys. Collection starts after NSYS_DELAY_S and lasts NSYS_DURATION_S;
# the server keeps running afterwards. Same env knobs as serve_<engine>.sh, plus the two NSYS_ vars.
#
#   bench/nsys_serve.sh <vllm|sglang>
set -euo pipefail

usage() { echo "usage: $0 <vllm|sglang>" >&2; exit 2; }
[[ $# -eq 1 ]] || usage
ENGINE="$1"
case "${ENGINE}" in
  vllm | sglang) ;;
  *) usage ;;
esac

for name in RUN_ID NSYS_DELAY_S NSYS_DURATION_S; do
  if [[ -z "${!name:-}" ]]; then echo "missing required env var: ${name}" >&2; exit 2; fi
done

HERE="$(dirname "${BASH_SOURCE[0]}")"
RUN_DIR="/data/cse291-results/${RUN_ID}"
mkdir -p "${RUN_DIR}"

exec nsys profile --trace cuda,nvtx,osrt --trace-fork-before-exec=true \
  --delay "${NSYS_DELAY_S}" --duration "${NSYS_DURATION_S}" --kill none \
  --force-overwrite true -o "${RUN_DIR}/nsys" \
  bash "${HERE}/serve_${ENGINE}.sh"
