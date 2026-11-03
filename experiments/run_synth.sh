#!/usr/bin/env bash
# Run the synthetic coding-agent workload against an already running vLLM or SGLang server 
#
#   SERVE_RUN_ID=... MAX_MODEL_LEN=12288 MAX_NUM_SEQS=8 MAX_BATCHED_TOKENS=8192 EVICTION_POLICY=lru \
#   K=4 experiments/run_synth.sh vllm mixed [extra synth_run.py flags, e.g. --rate 1 --arrival bursty]
#
# Mixes: smoke coding_short coding_mid coding_long mixed heldout 

set -euo pipefail

usage() { echo "usage: $0 <vllm|sglang> <mix> [synth_run.py flags...]" >&2; exit 2; }
[[ $# -ge 2 ]] || usage
ENGINE="$1"
MIX="$2"
shift 2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT=/data/cse291-results

for name in SERVE_RUN_ID K MAX_MODEL_LEN MAX_NUM_SEQS MAX_BATCHED_TOKENS EVICTION_POLICY; do
  if [[ -z "${!name:-}" ]]; then echo "missing required env var: ${name}" >&2; exit 2; fi
done

case "${ENGINE}" in
  vllm) PORT=8000 ;;
  sglang) PORT=30000 ;;
  *) usage ;;
esac

RUN_ID="synth-${ENGINE}-${MIX}-k${K}-b${MAX_NUM_SEQS}-t${MAX_BATCHED_TOKENS}-l${MAX_MODEL_LEN}-${EVICTION_POLICY}${RUN_TAG:+-${RUN_TAG}}"
RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"
SERVE_DIR="${RESULTS_ROOT}/${SERVE_RUN_ID}"
mkdir -p "${RUN_DIR}"
cp "${SERVE_DIR}/serve_config.env" "${RUN_DIR}/serve_config.env"
SERVE_LOG_OFFSET="$(wc -c < "${SERVE_DIR}/serve.log")"

curl -sf "http://127.0.0.1:${PORT}/v1/models" > /dev/null

python "${REPO_ROOT}/bench/collect_server_metrics.py" \
  --engine "${ENGINE}" --url "http://127.0.0.1:${PORT}/metrics" \
  --out "${RUN_DIR}/server_metrics.jsonl" --interval-s 5 --timeout-s 5 &
COLLECTOR_PID=$!
nvidia-smi dmon -s pucvmet -d 1 > "${RUN_DIR}/gpu_dmon.log" &
DMON_PID=$!
trap 'kill "${COLLECTOR_PID}" "${DMON_PID}" 2>/dev/null || true' EXIT

# Exit 3 means the mix cannot fit MAX_MODEL_LEN at all; still stop the collectors and report.
set +e
python "${REPO_ROOT}/bench/synth_run.py" \
  --base-url "http://127.0.0.1:${PORT}" --mix "${MIX}" --out-dir "${RUN_DIR}" \
  --max-model-len "${MAX_MODEL_LEN}" --k "${K}" "$@"
RC=$?
set -e

kill "${COLLECTOR_PID}" "${DMON_PID}"
wait "${COLLECTOR_PID}" "${DMON_PID}" 2>/dev/null || true
trap - EXIT
if (( RC != 0 )); then exit "${RC}"; fi

tail -c "+$((SERVE_LOG_OFFSET + 1))" "${SERVE_DIR}/serve.log" > "${RUN_DIR}/serve.log"
python "${REPO_ROOT}/bench/summarize_sla.py" run --engine "${ENGINE}" --run-dir "${RUN_DIR}"
