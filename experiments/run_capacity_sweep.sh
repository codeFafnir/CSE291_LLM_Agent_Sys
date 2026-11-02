#!/usr/bin/env bash
# Capacity and OOM grid. Outer loop restarts the server per (MAX_MODEL_LEN, MAX_NUM_SEQS).
# Inner loop sends synthetic prompts at each concurrency and prompt length, no restart
# unless a cell hit a runtime OOM.
#
#   EVICTION_POLICY=lru PREEMPTION_POLICY=fcfs MAX_BATCHED_TOKENS=8192 \
#   experiments/run_capacity_sweep.sh vllm
#
# Run from the engine's env. Optional overrides: CAP_LENS, CAP_SEQS, CAP_CONCS, CAP_PROMPTS (space separated).
# tlru on sglang also needs TLRU_THRESHOLD and TLRU_NEXT_PROMPT.
set -euo pipefail

usage() { echo "usage: $0 <vllm|sglang>" >&2; exit 2; }
[[ $# -eq 1 ]] || usage
ENGINE="$1"
case "${ENGINE}" in
  vllm) PORT=8000 ;;
  sglang) PORT=30000 ;;
  *) usage ;;
esac

for name in EVICTION_POLICY PREEMPTION_POLICY MAX_BATCHED_TOKENS; do
  if [[ -z "${!name:-}" ]]; then echo "missing required env var: ${name}" >&2; exit 2; fi
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT=/data/cse291-results
MAX_TOKENS=64
READY_TIMEOUT_S=900
LENS="${CAP_LENS:-8192 12288 16384 32768}"
SEQS="${CAP_SEQS:-1 2 4 8 16}"
CONCS="${CAP_CONCS:-1 2 4 8 16}"
PROMPTS="${CAP_PROMPTS:-1024 2048 4096 8192 12000 16000 24000 32000}"
PRIORITY_MODE=none
if [[ "${PREEMPTION_POLICY}" == "priority" ]]; then PRIORITY_MODE=index; fi

TOP_ID="capacity-${ENGINE}-${EVICTION_POLICY}"
TOP_DIR="${RESULTS_ROOT}/${TOP_ID}"
mkdir -p "${TOP_DIR}"

SERVER_PID=""

start_server() {
  setsid bash "${REPO_ROOT}/bench/serve_${ENGINE}.sh" > /dev/null 2>&1 &
  SERVER_PID=$!
  local waited=0
  while (( waited < READY_TIMEOUT_S )); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then return 1; fi
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" > /dev/null; then return 0; fi
    sleep 5
    waited=$((waited + 5))
  done
  return 1
}

stop_server() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
  fi
  # Wait for the GPU to be released before the next start.
  while (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1) > 1000 )); do sleep 3; done
}
trap stop_server EXIT

for len in ${LENS}; do
  for seqs in ${SEQS}; do
    export MAX_MODEL_LEN="${len}" MAX_NUM_SEQS="${seqs}"
    export RUN_ID="${TOP_ID}/cfg-l${len}-s${seqs}"
    RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"
    echo "== ${RUN_ID}"
    if ! start_server; then
      stop_server
      python "${REPO_ROOT}/bench/capacity_sweep.py" startup-failed --engine "${ENGINE}" --run-dir "${RUN_DIR}"
      continue
    fi
    for conc in ${CONCS}; do
      if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        stop_server
        if ! start_server; then
          stop_server
          python "${REPO_ROOT}/bench/capacity_sweep.py" startup-failed --engine "${ENGINE}" --run-dir "${RUN_DIR}"
          break
        fi
      fi
      set +e
      python "${REPO_ROOT}/bench/capacity_sweep.py" sweep \
        --engine "${ENGINE}" --run-dir "${RUN_DIR}" --base-url "http://127.0.0.1:${PORT}" \
        --concurrency "${conc}" --prompt-tokens ${PROMPTS} \
        --max-tokens "${MAX_TOKENS}" --request-priority "${PRIORITY_MODE}"
      rc=$?
      set -e
      if (( rc == 10 )); then stop_server; elif (( rc != 0 )); then exit "${rc}"; fi
    done
    stop_server
  done
done

cat "${TOP_DIR}"/cfg-*/capacity_log.jsonl > "${TOP_DIR}/capacity_log.jsonl"
python "${REPO_ROOT}/bench/summarize_sla.py" capacity --run-dir "${TOP_DIR}"
