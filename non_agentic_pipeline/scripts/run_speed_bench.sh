#!/usr/bin/env bash
# Run the SPEED-Bench subset against an already running scripts/serve_vllm.sh server and write every metric CSV.
#
#   RESULTS_ROOT=... SERVE_RUN_ID=serve-vllm-l20000 SUBSET=<speed_subset.jsonl> CONCURRENCY=1 \
#   MAX_MODEL_LEN=20000 MAX_NUM_SEQS=8 MAX_BATCHED_TOKENS=8192 non_agentic_pipeline/scripts/run_speed_bench.sh
#
# SUBSET is any subset jsonl; for a quick check use `head -n 3 speed_subset.jsonl > speed_subset_3.jsonl`.
# Run from the vLLM env with pydantic, openai, prometheus_client, nvidia-ml-py and nvidia-smi available.
set -euo pipefail

for name in RESULTS_ROOT SERVE_RUN_ID SUBSET CONCURRENCY MAX_MODEL_LEN MAX_NUM_SEQS MAX_BATCHED_TOKENS; do
  if [[ -z "${!name:-}" ]]; then echo "missing required env var: ${name}" >&2; exit 2; fi
done

PIPELINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPEEDBENCH="${PIPELINE_ROOT}/speedbench"
SERVER_URL=http://127.0.0.1:8000
MAX_OUTPUT_TOKENS=2048

UTC_START="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="speed-vllm-c${CONCURRENCY}-b${MAX_NUM_SEQS}-t${MAX_BATCHED_TOKENS}-l${MAX_MODEL_LEN}-${UTC_START}"
RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"
SERVE_DIR="${RESULTS_ROOT}/${SERVE_RUN_ID}"
mkdir -p "${RUN_DIR}"
cp "${SERVE_DIR}/serve_config.env" "${RUN_DIR}/serve_config.env"
cp "${SUBSET}" "${RUN_DIR}/subset.jsonl"
SERVE_LOG_OFFSET="$(wc -c < "${SERVE_DIR}/serve.log")"

curl -sf "${SERVER_URL}/v1/models" > /dev/null

chat_once() {
  curl -sf "${SERVER_URL}/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"Qwen/Qwen3-14B-AWQ","messages":[{"role":"user","content":"Count from 1 to 40."}],"max_tokens":128,"chat_template_kwargs":{"enable_thinking":false}}' > /dev/null
}

# One request must show live speculative decoding, else this run is not the baseline.
chat_once
LIVE="$(curl -sf "${SERVER_URL}/metrics" | awk '/^vllm:spec_decode_num_draft_tokens_total/ {s+=$NF} END {print (s>0)}')"
if [[ "${LIVE}" != "1" ]]; then
  echo "speculator not live on vllm; discard this run" >&2
  exit 3
fi
# Two more warmup requests, outside the measured window.
chat_once
chat_once

python "${SPEEDBENCH}/server_metrics.py" collect \
  --url "${SERVER_URL}/metrics" --out "${RUN_DIR}/server_metrics.jsonl" --interval-s 1 --timeout-s 5 &
COLLECTOR_PID=$!
nvidia-smi dmon -s pucvmet -d 1 > "${RUN_DIR}/gpu_dmon.log" &
DMON_PID=$!
trap 'kill "${COLLECTOR_PID}" "${DMON_PID}" 2>/dev/null || true' EXIT

python "${SPEEDBENCH}/bench_client.py" \
  --subset "${RUN_DIR}/subset.jsonl" --run-dir "${RUN_DIR}" --server-url "${SERVER_URL}" \
  --max-tokens "${MAX_OUTPUT_TOKENS}" --concurrency "${CONCURRENCY}"

kill "${COLLECTOR_PID}" "${DMON_PID}"
wait "${COLLECTOR_PID}" "${DMON_PID}" 2>/dev/null || true
trap - EXIT

tail -c "+$((SERVE_LOG_OFFSET + 1))" "${SERVE_DIR}/serve.log" > "${RUN_DIR}/serve.log"

python "${SPEEDBENCH}/report.py" \
  --run-dir "${RUN_DIR}" --all-runs-csv "${RESULTS_ROOT}/speed_all_runs.csv" \
  --concurrency "${CONCURRENCY}" \
  --repo-sha "$(git -C "${PIPELINE_ROOT}" rev-parse --short HEAD)" --utc-start "${UTC_START}"

echo "run dir: ${RUN_DIR}"
