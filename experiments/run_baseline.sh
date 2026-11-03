#!/usr/bin/env bash
# Run Harbor (terminus-2) against an already running vLLM or SGLang server and write summary.json.
#
#   SERVE_RUN_ID=... CONCURRENCY=1 MAX_MODEL_LEN=12288 MAX_NUM_SEQS=8 MAX_BATCHED_TOKENS=8192 \
#   EVICTION_POLICY=lru PREEMPTION_POLICY=fcfs experiments/run_baseline.sh vllm easy30
#
# Optional RUN_TAG suffixes the run dir (profiling runs).
# SERVE_RUN_ID is the RUN_ID the server was started with, so its serve.log and knobs are copied in.
# Needs bench deps (prometheus_client, pydantic, nvidia-ml-py) in the active env and harbor on PATH.
set -euo pipefail

usage() { echo "usage: $0 <vllm|sglang> <smoke|easy30|nemotron-smoke>" >&2; exit 2; }
[[ $# -eq 2 ]] || usage
ENGINE="$1"
JOB="$2"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT=/data/cse291-results
TASKS_DIR=/data/SWE-bench_Pro-os/v2/tasks
IDS_FILE=/data/SWE-bench_Pro-os/v2/easy30_ids.txt
JOB_ARGS=()
MODEL=openai/Qwen/Qwen3-14B-AWQ
MAX_OUTPUT_TOKENS=2048

for name in SERVE_RUN_ID CONCURRENCY MAX_MODEL_LEN MAX_NUM_SEQS MAX_BATCHED_TOKENS EVICTION_POLICY PREEMPTION_POLICY; do
  if [[ -z "${!name:-}" ]]; then echo "missing required env var: ${name}" >&2; exit 2; fi
done

case "${ENGINE}" in
  vllm) PORT=8000 ;;
  sglang) PORT=30000 ;;
  *) usage ;;
esac
case "${JOB}" in
  smoke) mapfile -t IDS < <(head -n 2 "${IDS_FILE}") ;;
  easy30) mapfile -t IDS < "${IDS_FILE}" ;;
  nemotron-smoke)
    # The task-declared docker_image is NVIDIA-internal, so build each task from its own Dockerfile.
    # Summarization is off so a context overflow fails the trial instead of being hidden.
    TASKS_DIR=/data/datasets/nemotron-terminal/smoke5
    IDS_FILE=/data/datasets/nemotron-terminal/smoke5_ids.txt
    mapfile -t IDS < "${IDS_FILE}"
    JOB_ARGS=(--force-build --ak enable_summarize=false --ak proactive_summarization_threshold=0) ;;
  *) usage ;;
esac

RUN_ID="baseline-${ENGINE}-${JOB}-c${CONCURRENCY}-b${MAX_NUM_SEQS}-t${MAX_BATCHED_TOKENS}-l${MAX_MODEL_LEN}-${EVICTION_POLICY}${RUN_TAG:+-${RUN_TAG}}"
RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"
SERVE_DIR="${RESULTS_ROOT}/${SERVE_RUN_ID}"
mkdir -p "${RUN_DIR}"
cp "${SERVE_DIR}/serve_config.env" "${RUN_DIR}/serve_config.env"
SERVE_LOG_OFFSET="$(wc -c < "${SERVE_DIR}/serve.log")"

curl -sf "http://127.0.0.1:${PORT}/v1/models" > /dev/null

# Send one request and require speculative decoding to be live, else this run is not the baseline.
curl -sf "http://127.0.0.1:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"Qwen/Qwen3-14B-AWQ\",\"messages\":[{\"role\":\"user\",\"content\":\"Count from 1 to 40.\"}],\"max_tokens\":128}" > /dev/null
METRICS="$(curl -sf "http://127.0.0.1:${PORT}/metrics")"
if [[ "${ENGINE}" == "vllm" ]]; then
  LIVE="$(awk '/^vllm:spec_decode_num_draft_tokens_total/ {s+=$NF} END {print (s>0)}' <<< "${METRICS}")"
else
  LIVE="$(awk '/^sglang:spec_accept_length/ {if ($NF>1) ok=1} END {print ok+0}' <<< "${METRICS}")"
fi
if [[ "${LIVE}" != "1" ]]; then
  echo "speculator not live on ${ENGINE}; discard this run" >&2
  exit 3
fi

python "${REPO_ROOT}/bench/collect_server_metrics.py" \
  --engine "${ENGINE}" --url "http://127.0.0.1:${PORT}/metrics" \
  --out "${RUN_DIR}/server_metrics.jsonl" --interval-s 5 --timeout-s 5 &
COLLECTOR_PID=$!
nvidia-smi dmon -s pucvmet -d 1 > "${RUN_DIR}/gpu_dmon.log" &
DMON_PID=$!
trap 'kill "${COLLECTOR_PID}" "${DMON_PID}" 2>/dev/null || true' EXIT

BASE_HOST=host.docker.internal
# Harbor must know the real context window (else it never summarizes) and cap each reply; thinking is off so replies fit.
MODEL_INFO="{\"max_input_tokens\":${MAX_MODEL_LEN},\"max_output_tokens\":${MAX_OUTPUT_TOKENS},\"input_cost_per_token\":0,\"output_cost_per_token\":0}"
LLM_KWARGS="{\"max_tokens\":${MAX_OUTPUT_TOKENS},\"extra_body\":{\"chat_template_kwargs\":{\"enable_thinking\":false}}}"
ID_ARGS=()
for id in "${IDS[@]}"; do ID_ARGS+=(-i "${id}"); done

OPENAI_API_KEY=not-needed harbor run -p "${TASKS_DIR}" \
  -e docker -n "${CONCURRENCY}" \
  "${ID_ARGS[@]}" \
  -a terminus-2 \
  -m "${MODEL}" \
  --api-format openai_chat_completions \
  --base-url "http://${BASE_HOST}:${PORT}/v1" \
  --ak "model_info=${MODEL_INFO}" \
  --ak "llm_kwargs=${LLM_KWARGS}" \
  "${JOB_ARGS[@]}" \
  --jobs-dir "${RUN_DIR}" --job-name harbor

kill "${COLLECTOR_PID}" "${DMON_PID}"
wait "${COLLECTOR_PID}" "${DMON_PID}" 2>/dev/null || true
trap - EXIT

tail -c "+$((SERVE_LOG_OFFSET + 1))" "${SERVE_DIR}/serve.log" > "${RUN_DIR}/serve.log"
python "${REPO_ROOT}/bench/summarize_sla.py" run --engine "${ENGINE}" --run-dir "${RUN_DIR}"
