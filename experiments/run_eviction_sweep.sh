#!/usr/bin/env bash
# Level 1 and 2 eviction sweep: for each EVICTION_POLICY, start the server, run EASY-30, stop it.
#
#   POLICIES="lru lfu slru priority tlru agent" PREEMPTION_POLICY=length CONCURRENCY=4 \
#   MAX_MODEL_LEN=12288 MAX_NUM_SEQS=8 MAX_BATCHED_TOKENS=8192 experiments/run_eviction_sweep.sh sglang
#
# vLLM only has lru and agent (agent needs the cse291-agent-evict branch of third_party/vllm).
# tlru needs TLRU_THRESHOLD and TLRU_NEXT_PROMPT, agent needs AGENT_IDLE_S and AGENT_SHARED_HITS (sglang).
set -euo pipefail

usage() { echo "usage: $0 <vllm|sglang>" >&2; exit 2; }
[[ $# -eq 1 ]] || usage
ENGINE="$1"
case "${ENGINE}" in
  vllm) PORT=8000 ;;
  sglang) PORT=30000 ;;
  *) usage ;;
esac

for name in POLICIES PREEMPTION_POLICY CONCURRENCY MAX_MODEL_LEN MAX_NUM_SEQS MAX_BATCHED_TOKENS; do
  if [[ -z "${!name:-}" ]]; then echo "missing required env var: ${name}" >&2; exit 2; fi
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
READY_TIMEOUT_S=900
SERVER_PID=""

stop_server() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
  fi
  while (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1) > 1000 )); do sleep 3; done
}
trap stop_server EXIT

for policy in ${POLICIES}; do
  export EVICTION_POLICY="${policy}"
  export RUN_ID="serve-${ENGINE}-evict-${policy}-${PREEMPTION_POLICY}-c${CONCURRENCY}"
  export SERVE_RUN_ID="${RUN_ID}"
  setsid bash "${REPO_ROOT}/bench/serve_${ENGINE}.sh" > /dev/null 2>&1 &
  SERVER_PID=$!
  waited=0
  until curl -sf "http://127.0.0.1:${PORT}/v1/models" > /dev/null; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null || (( waited >= READY_TIMEOUT_S )); then
      echo "server for ${policy} did not start; see /data/cse291-results/${RUN_ID}/serve.log" >&2
      stop_server
      continue 2
    fi
    sleep 5
    waited=$((waited + 5))
  done
  "${REPO_ROOT}/experiments/run_baseline.sh" "${ENGINE}" easy30
  stop_server
done
