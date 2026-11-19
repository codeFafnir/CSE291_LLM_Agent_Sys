#!/usr/bin/env bash
# Run ONE SWE-bench Verified experiment cell (SWE-agent sessions, Poisson arrivals, in-flight cap C) against a running vLLM.
# Same cell as run_swesmith.sh minus the C++ image allow-list and scoring; the sessions come from select_verified.py.
#
#   SERVE_RUN_ID=... CONCURRENCY=1 MAX_MODEL_LEN=30000 MAX_NUM_SEQS=8 MAX_BATCHED_TOKENS=8192 \
#   EVICTION_POLICY=lru PREEMPTION_POLICY=fcfs COUNT=20 RATE=1000000 SEED=291 TOOL_TIMEOUT_S=10 \
#   experiments/run_swebench.sh
#
# Required env: SERVE_RUN_ID CONCURRENCY MAX_MODEL_LEN MAX_NUM_SEQS MAX_BATCHED_TOKENS EVICTION_POLICY PREEMPTION_POLICY
#   COUNT (first N rows of the instances file; 1 is the canary) RATE (sessions/s; 1000000 = back-to-back) SEED TOOL_TIMEOUT_S (10..15).
# Optional env: RUN_TAG; NO_SPECULATOR=1 (server started without a speculator: skip the spec-live gate);
#   INSTANCES_FILE (default ${INSTANCES_DIR}/instances_smoke20.jsonl);
#   RESULTS_ROOT INSTANCES_DIR SWEAGENT_PY VLLM_PY UPSTREAM PROXY_PORT DOCKER override the box defaults below.
# Exit codes: 2 bad env, 3 speculator not live, 5 a swe-rex container is already running (leftover from another cell).
# swe-rex containers are named <sanitized image>-<uuid4>; only containers matching that pattern are ever removed.
# SERVE_RUN_ID is the RUN_ID the server was started with, so its serve.log and knobs are copied in.
# The run dir ends with sessions/tool_spans/proxy_requests files, report/, run_metrics.csv, cell_done.json, and one
# row is appended to ${RESULTS_ROOT}/all_runs.csv. A dead server or failed step is recorded in cell_done.json, not fatal.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-/data/cse291-results}"
INSTANCES_DIR="${INSTANCES_DIR:-/data/swebench}"
INSTANCES_FILE="${INSTANCES_FILE:-${INSTANCES_DIR}/instances_smoke20.jsonl}"
SWEAGENT_PY="${SWEAGENT_PY:-/data/envs/sweagent/bin/python}"
VLLM_PY="${VLLM_PY:-/data/envs/vllm/bin/python}"
UPSTREAM="${UPSTREAM:-http://127.0.0.1:8000}"
DOCKER="${DOCKER:-docker}"
PROXY_PORT="${PROXY_PORT:-8100}"
IMAGE_KEY=verified

for name in SERVE_RUN_ID CONCURRENCY MAX_MODEL_LEN MAX_NUM_SEQS MAX_BATCHED_TOKENS EVICTION_POLICY PREEMPTION_POLICY \
  COUNT RATE SEED TOOL_TIMEOUT_S; do
  if [[ -z "${!name:-}" ]]; then echo "missing required env var: ${name}" >&2; exit 2; fi
done
if ! [[ "${TOOL_TIMEOUT_S}" =~ ^(1[0-5])$ ]]; then
  echo "TOOL_TIMEOUT_S must be an integer in [10, 15], got ${TOOL_TIMEOUT_S}" >&2; exit 2
fi
if [[ ! -f "${INSTANCES_FILE}" ]]; then echo "instances file not found: ${INSTANCES_FILE}" >&2; exit 2; fi

# Container helpers live in run.py (single source for the swe-rex name pattern); they get docker, then files/args.
run_py() { PYTHONPATH="${REPO_ROOT}/swesmith" "${VLLM_PY}" -c "$@"; }
LIST_RUNNING='
import sys
from pathlib import Path
import run
docker, *files = sys.argv[1:]
images = sorted({i for f in files for i in run.images_of(Path(f))})
print("\n".join(run.running_swerex_names(docker, images)))
'
CLEANUP='
import sys
from pathlib import Path
import run
docker, since, log, *files = sys.argv[1:]
images = sorted({i for f in files for i in run.images_of(Path(f))})
result = run.cleanup_swerex_containers(docker, images, float(since))
run.append_cleanup_log(Path(log), result.failures)
print(result.removed)
'

LEFTOVER="$(run_py "${LIST_RUNNING}" "${DOCKER}" "${INSTANCES_FILE}")"
if [[ -n "${LEFTOVER}" ]]; then
  echo "swe-rex container(s) already running, refusing to start the cell (stop them first):" >&2
  echo "${LEFTOVER}" >&2
  exit 5
fi

START_EPOCH="$(date +%s)"
UTC_START="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="swebench-${IMAGE_KEY}-c${CONCURRENCY}-b${MAX_NUM_SEQS}-t${MAX_BATCHED_TOKENS}-l${MAX_MODEL_LEN}-${EVICTION_POLICY}-p${PREEMPTION_POLICY}${RUN_TAG:+-${RUN_TAG}}-${UTC_START}"
RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"
SERVE_DIR="${RESULTS_ROOT}/${SERVE_RUN_ID}"
mkdir -p "${RUN_DIR}"
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

WINDOW_START="$(date +%s.%N)"
"${VLLM_PY}" "${REPO_ROOT}/bench/collect_server_metrics.py" \
  --engine vllm --url "${UPSTREAM}/metrics" \
  --out "${RUN_DIR}/server_metrics.jsonl" --interval-s 1 --timeout-s 5 &
COLLECTOR_PID=$!
nvidia-smi dmon -s pucvmet -d 1 > "${RUN_DIR}/gpu_dmon.log" &
DMON_PID=$!
trap 'kill "${COLLECTOR_PID}" "${DMON_PID}" 2>/dev/null || true' EXIT

# Each step records its exit status instead of aborting, so a dead server still yields cell_done.json.
STEP_STATUS=()
run_step() {
  local name="$1"
  shift
  local status=0
  "$@" || status=$?
  STEP_STATUS+=("${name}=${status}")
}

run_step run "${VLLM_PY}" "${REPO_ROOT}/swesmith/run.py" \
  --instances-jsonl "${INSTANCES_FILE}" \
  --count "${COUNT}" --concurrency "${CONCURRENCY}" --rate "${RATE}" --seed "${SEED}" \
  --run-dir "${RUN_DIR}" --upstream "${UPSTREAM}" --proxy-port "${PROXY_PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" --tool-timeout-s "${TOOL_TIMEOUT_S}" \
  --agent-config "${REPO_ROOT}/swebench/agent_config.yaml" \
  --agent-entry "${REPO_ROOT}/swesmith/agent_entry.py" \
  --sweagent-python "${SWEAGENT_PY}" --docker "${DOCKER}"

kill "${COLLECTOR_PID}" "${DMON_PID}" 2>/dev/null || true
wait "${COLLECTOR_PID}" "${DMON_PID}" 2>/dev/null || true
trap - EXIT

WINDOW_END="$(date +%s.%N)"

# Whatever run.py left behind (it crashed, or its own cleanup failed): remove swe-rex containers made since WINDOW_START.
# Runs after WINDOW_END so cleanup time is outside the measured window. Docker errors go to container_cleanup.log.
CONTAINERS_REMOVED=-1
if CLEANUP_OUT="$(run_py "${CLEANUP}" "${DOCKER}" "${WINDOW_START}" "${RUN_DIR}/container_cleanup.log" "${INSTANCES_FILE}")"; then
  CONTAINERS_REMOVED="${CLEANUP_OUT}"
  STEP_STATUS+=("container_cleanup=0")
else
  STEP_STATUS+=("container_cleanup=$?")
fi
CLEANUP_FAILURES=0
if [[ -f "${RUN_DIR}/container_cleanup.log" ]]; then CLEANUP_FAILURES="$(wc -l < "${RUN_DIR}/container_cleanup.log" | tr -d ' ')"; fi

tail -c "+$((SERVE_LOG_OFFSET + 1))" "${SERVE_DIR}/serve.log" > "${RUN_DIR}/serve.log"
run_step summarize_sla "${VLLM_PY}" "${REPO_ROOT}/bench/summarize_sla.py" run --engine vllm --run-dir "${RUN_DIR}"

# Let the last OTLP span batch reach the collector, then write every CSV (empty harbor/: tasks.csv is header-only).
sleep 10
ENGINE_VERSION="$("${VLLM_PY}" -c "import vllm; print(vllm.__version__)" || echo unknown)"
run_step run_csv "${VLLM_PY}" "${REPO_ROOT}/bench/run_csv.py" \
  --engine vllm --run-dir "${RUN_DIR}" \
  --window-start "${WINDOW_START}" --window-end "${WINDOW_END}" \
  --raw-source "${RESULTS_ROOT}/otel/spans.jsonl" --all-runs-csv "${RESULTS_ROOT}/all_runs.csv" \
  --max-context "${MAX_MODEL_LEN}" --job "swebench-${IMAGE_KEY}" --concurrency "${CONCURRENCY}" \
  --engine-version "${ENGINE_VERSION}" --repo-sha "$(git -C "${REPO_ROOT}" rev-parse --short HEAD)" \
  --utc-start "${UTC_START}" --profile-dir "${RUN_DIR}/torch_profile"

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
run_id, wall_s, serve_alive, removed, cleanup_failures, kv_pool_gb, sessions_path, out, *steps = sys.argv[1:]
status = {name: int(code) for name, code in (s.split("=") for s in steps)}
counts = {k: 0 for k in ("completed", "no_trajectory", "crashed", "wall_timeout")}
try:
    for line in open(sessions_path):
        if line.strip():
            st = json.loads(line)["status"]
            counts[st] = counts.get(st, 0) + 1
except FileNotFoundError:
    counts = None
cell = {"run_id": run_id, "wall_s": int(wall_s), "serve_alive": serve_alive == "true", "run_exit_status": status["run"], "containers_removed": int(removed), "container_cleanup_failures": int(cleanup_failures), "kv_pool_gb": float(kv_pool_gb) if kv_pool_gb else None, "status_counts": counts, "step_exit_status": status}
open(out, "w").write(json.dumps(cell, indent=2) + "\n")
' "${RUN_ID}" "$((END_EPOCH - START_EPOCH))" "${SERVE_ALIVE}" "${CONTAINERS_REMOVED}" "${CLEANUP_FAILURES}" "${KV_POOL_GB}" "${RUN_DIR}/sessions.jsonl" "${RUN_DIR}/cell_done.json" "${STEP_STATUS[@]}"
echo "cell done: ${RUN_DIR}"
