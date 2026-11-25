#!/usr/bin/env bash
# Run ONE replay cell: recorded SWE-smith sessions replayed through the session proxy against a running vLLM.
# No docker, no sweagent: only the recorded requests, tool gaps (scaled by GAP_SCALE) and Poisson arrivals with in-flight cap C.
#
#   SERVE_RUN_ID=... CONCURRENCY=4 MAX_MODEL_LEN=30000 MAX_NUM_SEQS=8 MAX_BATCHED_TOKENS=8192 \
#   EVICTION_POLICY=lru PREEMPTION_POLICY=fcfs COUNT=40 RATE=0.02 SEED=0 GAP_SCALE=1 CORPUS=/data/.../corpus \
#   UPSTREAM=http://127.0.0.1:8000 PROXY_PORT=8100 experiments/run_replay.sh
#
# Required env: SERVE_RUN_ID CONCURRENCY MAX_MODEL_LEN MAX_NUM_SEQS MAX_BATCHED_TOKENS EVICTION_POLICY PREEMPTION_POLICY
#   RATE (sessions/s) COUNT SEED GAP_SCALE (1 = recorded gaps) CORPUS (<corpus>/<session>/tNNN.json + proxy_requests.jsonl)
#   UPSTREAM (vLLM base url) PROXY_PORT.
# Optional env: RUN_TAG; NO_SPECULATOR=1 (server started without a speculator: skip the spec-live gate);
#   RESULTS_ROOT VLLM_PY override the box defaults below.
# Exit codes: 2 bad env, 3 speculator not live, 4 /reset_prefix_cache did not return success true, 6 proxy did not start.
# SERVE_RUN_ID is the RUN_ID the server was started with, so its serve.log and knobs are copied in.
# The run dir ends with proxy_requests.jsonl, recorded/, replay_results.jsonl, report/ and cell_done.json.
# A dead server or failed step is recorded in cell_done.json, not fatal.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-/data/cse291-results}"
VLLM_PY="${VLLM_PY:-/data/envs/vllm/bin/python}"

for name in SERVE_RUN_ID CONCURRENCY MAX_MODEL_LEN MAX_NUM_SEQS MAX_BATCHED_TOKENS EVICTION_POLICY PREEMPTION_POLICY \
  RATE COUNT SEED GAP_SCALE CORPUS UPSTREAM PROXY_PORT; do
  if [[ -z "${!name:-}" ]]; then echo "missing required env var: ${name}" >&2; exit 2; fi
done

START_EPOCH="$(date +%s)"
UTC_START="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="replay-c${CONCURRENCY}-b${MAX_NUM_SEQS}-t${MAX_BATCHED_TOKENS}-l${MAX_MODEL_LEN}-${EVICTION_POLICY}-p${PREEMPTION_POLICY}${RUN_TAG:+-${RUN_TAG}}-${UTC_START}"
RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"
SERVE_DIR="${RESULTS_ROOT}/${SERVE_RUN_ID}"
mkdir -p "${RUN_DIR}/recorded"
cp "${SERVE_DIR}/serve_config.env" "${RUN_DIR}/serve_config.env"
SERVE_LOG_OFFSET="$(wc -c < "${SERVE_DIR}/serve.log")"

curl -sf "${UPSTREAM}/v1/models" > /dev/null

# Send one request and require speculative decoding to be live, else this cell is not comparable.
curl -sf "${UPSTREAM}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"Qwen/Qwen3-14B-AWQ\",\"messages\":[{\"role\":\"user\",\"content\":\"Count from 1 to 40.\"}],\"max_tokens\":128}" > /dev/null
METRICS="$(curl -sf "${UPSTREAM}/metrics")"
LIVE="$(awk '/^vllm:spec_decode_num_draft_tokens_total/ {s+=$NF} END {print (s>0)}' <<< "${METRICS}")"
if [[ "${LIVE}" != "1" && "${NO_SPECULATOR:-0}" != "1" ]]; then
  echo "speculator not live on vllm; discard this run" >&2
  exit 3
fi

# Every cell starts from an empty prefix cache (the spec-live request above is already finished, so nothing holds blocks).
RESET_BODY="$(curl -sf -X POST "${UPSTREAM}/reset_prefix_cache" || true)"
if ! "${VLLM_PY}" -c 'import json, sys; sys.exit(0 if json.loads(sys.argv[1]).get("success") is True else 1)' "${RESET_BODY:-null}" 2> /dev/null; then
  echo "reset_prefix_cache did not return success true (got: ${RESET_BODY:-<no response>}); cell failed" >&2
  exit 4
fi

PROXY_PID=""
COLLECTOR_PID=""
DMON_PID=""
trap 'kill ${PROXY_PID} ${COLLECTOR_PID} ${DMON_PID} 2>/dev/null || true' EXIT

"${VLLM_PY}" "${REPO_ROOT}/swesmith/session_proxy.py" \
  --upstream "${UPSTREAM}" --port "${PROXY_PORT}" \
  --log-path "${RUN_DIR}/proxy_requests.jsonl" --record-dir "${RUN_DIR}/recorded" &
PROXY_PID=$!
PROXY_UP=0
for _ in $(seq 1 100); do
  if curl -sf "http://127.0.0.1:${PROXY_PORT}/healthz" > /dev/null; then PROXY_UP=1; break; fi
  sleep 0.3
done
if [[ "${PROXY_UP}" != "1" ]]; then echo "session proxy did not answer on port ${PROXY_PORT}" >&2; exit 6; fi
# report.py must never hit a missing file when no request reaches the proxy.
touch "${RUN_DIR}/proxy_requests.jsonl" "${RUN_DIR}/tool_spans.jsonl"

WINDOW_START="$(date +%s.%N)"
"${VLLM_PY}" "${REPO_ROOT}/bench/collect_server_metrics.py" \
  --engine vllm --url "${UPSTREAM}/metrics" \
  --out "${RUN_DIR}/server_metrics.jsonl" --interval-s 1 --timeout-s 5 &
COLLECTOR_PID=$!
nvidia-smi dmon -s pucvmet -d 1 > "${RUN_DIR}/gpu_dmon.log" &
DMON_PID=$!

# Each step records its exit status instead of aborting, so a dead server still yields cell_done.json.
STEP_STATUS=()
run_step() {
  local name="$1"
  shift
  local status=0
  "$@" || status=$?
  STEP_STATUS+=("${name}=${status}")
}

run_step replay "${VLLM_PY}" "${REPO_ROOT}/swesmith/replay.py" \
  --corpus "${CORPUS}" --proxy-base "http://127.0.0.1:${PROXY_PORT}" \
  --rate "${RATE}" --count "${COUNT}" --seed "${SEED}" \
  --concurrency "${CONCURRENCY}" --gap-scale "${GAP_SCALE}" \
  --out "${RUN_DIR}/replay_results.jsonl"

kill "${COLLECTOR_PID}" "${DMON_PID}" "${PROXY_PID}" 2>/dev/null || true
wait "${COLLECTOR_PID}" "${DMON_PID}" "${PROXY_PID}" 2>/dev/null || true
trap - EXIT

WINDOW_END="$(date +%s.%N)"

tail -c "+$((SERVE_LOG_OFFSET + 1))" "${SERVE_DIR}/serve.log" > "${RUN_DIR}/serve.log"

# KV pool size comes from the server's startup log (last "Available KV cache memory: 5.98 GiB" line in the full serve.log).
KV_POOL_GB="$(sed -nE 's/.*Available KV cache memory: ([0-9.]+) GiB.*/\1/p' "${SERVE_DIR}/serve.log" | tail -n 1)"
KV_ARGS=()
if [[ -n "${KV_POOL_GB}" ]]; then
  KV_ARGS=(--kv-pool-gb "${KV_POOL_GB}")
else
  echo "warning: no 'Available KV cache memory' line in ${SERVE_DIR}/serve.log; report.py runs without --kv-pool-gb" >&2
fi

run_step report env PYTHONPATH="${REPO_ROOT}/bench:${REPO_ROOT}/swesmith" \
  "${VLLM_PY}" "${REPO_ROOT}/swesmith/report.py" --run-dir "${RUN_DIR}" --window-start "${WINDOW_START}" "${KV_ARGS[@]}"

SERVE_ALIVE=false
if curl -sf --max-time 5 "${UPSTREAM}/v1/models" > /dev/null; then SERVE_ALIVE=true; fi
END_EPOCH="$(date +%s)"
"${VLLM_PY}" -c '
import json, sys
run_id, wall_s, serve_alive, kv_pool_gb, window_s, results_path, out, *steps = sys.argv[1:]
status = {name: int(code) for name, code in (s.split("=") for s in steps)}
counts = {}
try:
    for line in open(results_path):
        if line.strip():
            st = str(json.loads(line)["status"])
            counts[st] = counts.get(st, 0) + 1
except FileNotFoundError:
    counts = None
cell = {"run_id": run_id, "wall_s": int(wall_s), "window_s": float(window_s), "serve_alive": serve_alive == "true", "replay_exit_status": status["replay"], "kv_pool_gb": float(kv_pool_gb) if kv_pool_gb else None, "http_status_counts": counts, "step_exit_status": status}
open(out, "w").write(json.dumps(cell, indent=2) + "\n")
' "${RUN_ID}" "$((END_EPOCH - START_EPOCH))" "${SERVE_ALIVE}" "${KV_POOL_GB}" "$(awk -v a="${WINDOW_START}" -v b="${WINDOW_END}" 'BEGIN {print b - a}')" \
  "${RUN_DIR}/replay_results.jsonl" "${RUN_DIR}/cell_done.json" "${STEP_STATUS[@]}"
echo "cell done: ${RUN_DIR}"
