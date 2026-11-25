#!/usr/bin/env bash
# Runs on the GPU box. Replay matrix: one server restart per config, every cell of that config before the next restart.
#
#   CORPUS=/data/.../corpus COUNT=40 RATE=0.02 SEED=0 UPSTREAM=http://127.0.0.1:8000 PROXY_PORT=8100 \
#   [RUN_GAP_SCALE=1] experiments/run_matrix.sh
#
# Required env: CORPUS COUNT S_PILOT SEED UPSTREAM PROXY_PORT.
# Each cell's arrival rate is C / S_PILOT, so concurrency changes the offered load.
# Optional env: RUN_GAP_SCALE=1 adds the GAP_SCALE 10/30 cells (lru/fcfs and lru/continuum, C=4).
# A failed cell (or a failed server restart, which fails all its cells) is recorded and the matrix continues.
# Exit code: 0 when every cell passed, 1 when any cell failed, 2 bad env.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for name in CORPUS COUNT S_PILOT SEED UPSTREAM PROXY_PORT; do
  if [[ -z "${!name:-}" ]]; then echo "missing required env var: ${name}" >&2; exit 2; fi
done
RUN_GAP_SCALE="${RUN_GAP_SCALE:-0}"

FAILED_CELLS=()
TOTAL_CELLS=0

# run_cell <serve_run_id> <eviction> <preemption> <seqs> <tokens> <concurrency> <gap_scale>
run_cell() {
  local serve_run_id="$1" eviction="$2" preemption="$3" seqs="$4" tokens="$5" conc="$6" gap="$7"
  local label="${eviction}/${preemption} seqs=${seqs} tokens=${tokens} C=${conc} gap=${gap}"
  local status=0 rate
  rate="$(python3 -c "print(${conc} / ${S_PILOT})")"
  TOTAL_CELLS=$((TOTAL_CELLS + 1))
  echo "=== cell: ${label} rate=${rate} (serve ${serve_run_id})"
  env SERVE_RUN_ID="${serve_run_id}" CONCURRENCY="${conc}" MAX_MODEL_LEN=30000 \
    MAX_NUM_SEQS="${seqs}" MAX_BATCHED_TOKENS="${tokens}" \
    EVICTION_POLICY="${eviction}" PREEMPTION_POLICY="${preemption}" \
    GAP_SCALE="${gap}" RUN_TAG="gap${gap}" \
    RATE="${rate}" COUNT="${COUNT}" SEED="${SEED}" CORPUS="${CORPUS}" \
    UPSTREAM="${UPSTREAM}" PROXY_PORT="${PROXY_PORT}" \
    bash "${REPO_ROOT}/experiments/run_replay.sh" || status=$?
  echo "=== cell exit: ${status} (${label})"
  if [[ "${status}" != "0" ]]; then FAILED_CELLS+=("${label} exit=${status}"); fi
}

# run_config <eviction> <preemption> <ttl or ""> <seqs> <tokens> <"C:gap"...>
# Restarts the server once, then runs every cell.
run_config() {
  local eviction="$1" preemption="$2" ttl="$3" seqs="$4" tokens="$5"
  shift 5
  local serve_run_id="serve-verified-${eviction}-${preemption}-s${seqs}-t${tokens}"
  local ttl_args=()
  if [[ -n "${ttl}" ]]; then ttl_args=("${ttl}"); fi
  local status=0
  echo "##### config: ${serve_run_id} ttl=${ttl:-none}"
  export MAX_NUM_SEQS="${seqs}" MAX_BATCHED_TOKENS="${tokens}"
  bash "${REPO_ROOT}/bench/start_serve.sh" "${serve_run_id}" "${eviction}" "${preemption}" ${ttl_args[@]+"${ttl_args[@]}"} || status=$?
  echo "##### restart exit: ${status} (${serve_run_id})"
  local cell conc gap
  for cell in "$@"; do
    conc="${cell%%:*}"
    gap="${cell##*:}"
    if [[ "${status}" != "0" ]]; then
      TOTAL_CELLS=$((TOTAL_CELLS + 1))
      echo "=== cell: ${eviction}/${preemption} seqs=${seqs} tokens=${tokens} C=${conc} gap=${gap} skipped, restart failed"
      FAILED_CELLS+=("${eviction}/${preemption} seqs=${seqs} tokens=${tokens} C=${conc} gap=${gap} restart exit=${status}")
      continue
    fi
    run_cell "${serve_run_id}" "${eviction}" "${preemption}" "${seqs}" "${tokens}" "${conc}" "${gap}"
  done
}

lru_fcfs_cells=("1:1" "4:1" "8:1" "1:0" "2:0" "4:0" "8:0")
lru_continuum_cells=("1:1" "4:1" "8:1")
if [[ "${RUN_GAP_SCALE}" == "1" ]]; then
  lru_fcfs_cells+=("4:10" "4:30")
  lru_continuum_cells+=("4:10" "4:30")
fi

# Set A + B2 + gap loop share the 8/8192 server per policy.
run_config lru fcfs "" 8 8192 "${lru_fcfs_cells[@]}"
run_config lru continuum 2 8 8192 "${lru_continuum_cells[@]}"
run_config agent fcfs "" 8 8192 "1:1" "4:1" "8:1"

# Set B1: lru/fcfs, C=4, GAP_SCALE=1; 8/8192 already ran above.
for seqs in 1 2 4; do
  run_config lru fcfs "" "${seqs}" 8192 "4:1"
done
for tokens in 2048 4096; do
  run_config lru fcfs "" 8 "${tokens}" "4:1"
done

echo "##### matrix done: ${TOTAL_CELLS} cells, ${#FAILED_CELLS[@]} failed"
if [[ ${#FAILED_CELLS[@]} -gt 0 ]]; then
  printf 'failed: %s\n' "${FAILED_CELLS[@]}"
  exit 1
fi
