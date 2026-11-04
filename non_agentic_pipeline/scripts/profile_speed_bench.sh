#!/usr/bin/env bash
# Torch-profile one median-length easy and one medium SPEED-Bench prompt on a running scripts/serve_vllm.sh server.
#
#   RESULTS_ROOT=... SERVE_RUN_ID=serve-vllm-l20000 SUBSET=<speed_subset.jsonl> \
#   non_agentic_pipeline/scripts/profile_speed_bench.sh
#
# Output: ${RESULTS_ROOT}/speed-profile-vllm-<UTC>/torch_profile/<easy|medium>/{cold,warm}, plus
# gpu_dmon.log and server_metrics.jsonl for the profiling window.
# Run from the vLLM env with pydantic, prometheus_client, nvidia-ml-py and nvidia-smi available.
set -euo pipefail

for name in RESULTS_ROOT SERVE_RUN_ID SUBSET; do
  if [[ -z "${!name:-}" ]]; then echo "missing required env var: ${name}" >&2; exit 2; fi
done

PIPELINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPEEDBENCH="${PIPELINE_ROOT}/speedbench"
SERVER_URL=http://127.0.0.1:8000

UTC_START="$(date -u +%Y%m%dT%H%M%SZ)"
PROFILE_DIR="${RESULTS_ROOT}/speed-profile-vllm-${UTC_START}"
SERVER_TRACE_DIR="${RESULTS_ROOT}/${SERVE_RUN_ID}/torch_profile"
mkdir -p "${PROFILE_DIR}/torch_profile"
cp "${RESULTS_ROOT}/${SERVE_RUN_ID}/serve_config.env" "${PROFILE_DIR}/serve_config.env"

curl -sf "${SERVER_URL}/v1/models" > /dev/null

python "${SPEEDBENCH}/profile_replay.py" pick --subset "${SUBSET}" --out-dir "${PROFILE_DIR}"

python "${SPEEDBENCH}/server_metrics.py" collect \
  --url "${SERVER_URL}/metrics" --out "${PROFILE_DIR}/server_metrics.jsonl" --interval-s 1 --timeout-s 5 &
COLLECTOR_PID=$!
nvidia-smi dmon -s pucvmet -d 1 > "${PROFILE_DIR}/gpu_dmon.log" &
DMON_PID=$!
trap 'kill "${COLLECTOR_PID}" "${DMON_PID}" 2>/dev/null || true' EXIT

for tier in easy medium; do
  python "${SPEEDBENCH}/profile_replay.py" replay \
    --messages-json "${PROFILE_DIR}/profile_${tier}.json" \
    --out-dir "${PROFILE_DIR}/torch_profile/${tier}" --server-trace-dir "${SERVER_TRACE_DIR}"
done

kill "${COLLECTOR_PID}" "${DMON_PID}"
wait "${COLLECTOR_PID}" "${DMON_PID}" 2>/dev/null || true
trap - EXIT

echo "profile dir: ${PROFILE_DIR}"
