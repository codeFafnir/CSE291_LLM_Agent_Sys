#!/usr/bin/env bash
# Every experiment in order: baselines, capacity sweeps, eviction sweeps, profiling.
# Run inside tmux on the box after bootstrap_ec2.sh. A failed step is logged and the run continues.
#
#   tmux new -s run 'bash experiments/run_all.sh'
#
# Profiling replay needs /data/cse291-results/replay_turn.json (a captured terminus-2 chat request).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS=/data/cse291-results
LOG="${RESULTS}/run_all.log"
REPLAY_TURN="${RESULTS}/replay_turn.json"
ENGINES="vllm sglang"
READY_TIMEOUT_S=900
PROFILE_WINDOW_S=30
PROFILE_AFTER_S=120
NSYS_DELAY_S="${NSYS_DELAY_S:-420}"
NSYS_DURATION_S="${NSYS_DURATION_S:-60}"
PYSPY_DURATION_S=60

export MAX_MODEL_LEN=12288 MAX_NUM_SEQS=8 MAX_BATCHED_TOKENS=8192
export TLRU_THRESHOLD=4096 TLRU_NEXT_PROMPT=512 AGENT_IDLE_S=300 AGENT_SHARED_HITS=4
export PATH="${HOME}/.local/bin:${PATH}"

mkdir -p "${RESULTS}"

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "${LOG}"; }

step() {
  local name="$1"
  shift
  log "START ${name}"
  if "$@" >> "${LOG}" 2>&1; then log "OK ${name}"; else log "FAILED ${name} rc=$?"; fi
}

port_of() { [[ "$1" == "vllm" ]] && echo 8000 || echo 30000; }
preempt_of() { [[ "$1" == "vllm" ]] && echo fcfs || echo length; }
use_env() { source "/data/envs/$1/bin/activate"; }

SERVER_PID=""
PYSPY_PID=""

stop_server() {
  if [[ -n "${PYSPY_PID}" ]]; then kill "${PYSPY_PID}" 2>/dev/null || true; PYSPY_PID=""; fi
  if [[ -n "${SERVER_PID}" ]]; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
  fi
  while (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1) > 1000 )); do sleep 3; done
}
trap stop_server EXIT

# start_server <engine> <command...>. RUN_ID, EVICTION_POLICY, PREEMPTION_POLICY must be exported.
start_server() {
  local engine="$1" waited=0
  shift
  mkdir -p "${RESULTS}/${RUN_ID}"
  setsid "$@" > /dev/null 2>&1 &
  SERVER_PID=$!
  until curl -sf "http://127.0.0.1:$(port_of "${engine}")/v1/models" > /dev/null; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null || (( waited >= READY_TIMEOUT_S )); then
      echo "server ${RUN_ID} did not start" >&2
      stop_server
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

baselines() {
  local engine="$1"
  use_env "${engine}"
  export EVICTION_POLICY=lru PREEMPTION_POLICY
  PREEMPTION_POLICY="$(preempt_of "${engine}")"
  export RUN_ID="serve-${engine}-baseline" SERVE_RUN_ID="serve-${engine}-baseline"
  start_server "${engine}" bash "${REPO_ROOT}/bench/serve_${engine}.sh" || return 1
  local conc
  for conc in 1 4; do
    CONCURRENCY="${conc}" "${REPO_ROOT}/experiments/run_baseline.sh" "${engine}" easy30
  done
  stop_server
}

capacity() {
  local engine="$1"
  use_env "${engine}"
  EVICTION_POLICY=lru PREEMPTION_POLICY="$(preempt_of "${engine}")" "${REPO_ROOT}/experiments/run_capacity_sweep.sh" "${engine}"
}

capacity_no_speculator() {
  local engine="$1"
  use_env "${engine}"
  NO_SPECULATOR=1 CAP_TAG=nospec CAP_LENS=12288 CAP_SEQS=8 \
    EVICTION_POLICY=lru PREEMPTION_POLICY="$(preempt_of "${engine}")" "${REPO_ROOT}/experiments/run_capacity_sweep.sh" "${engine}"
}

# lru at c4 is already the baseline run, so only the other policies run here.
eviction() {
  local engine="$1" policies
  use_env "${engine}"
  [[ "${engine}" == "sglang" ]] && policies="lfu slru priority tlru agent" || policies="agent"
  POLICIES="${policies}" PREEMPTION_POLICY="$(preempt_of "${engine}")" CONCURRENCY=4 \
    "${REPO_ROOT}/experiments/run_eviction_sweep.sh" "${engine}"
}

offline_profile() {
  local engine="$1"
  use_env "${engine}"
  (cd "${REPO_ROOT}/bench" && PROFILE_DIR="${RESULTS}/profile-${engine}/offline" python "profile_${engine}.py")
}

# Profiling server. vLLM writes traces to ${RESULTS}/${RUN_ID}/torch_profile, SGLang to the output_dir of each window (PROFILE_DIR).
start_profile_server() {
  local engine="$1" tag="$2"
  use_env "${engine}"
  export EVICTION_POLICY=lru PREEMPTION_POLICY
  PREEMPTION_POLICY="$(preempt_of "${engine}")"
  export RUN_ID="profile-${engine}-${tag}-serve" SERVE_RUN_ID="profile-${engine}-${tag}-serve"
  export PROFILE_DIR="${RESULTS}/profile-${engine}/${tag}" VLLM_SERVER_DEV_MODE=1
  start_server "${engine}" bash "${REPO_ROOT}/bench/serve_${engine}.sh"
}

replay_profile() {
  local engine="$1"
  [[ -f "${REPLAY_TURN}" ]] || { echo "missing ${REPLAY_TURN}" >&2; return 1; }
  start_profile_server "${engine}" replay || return 1
  python "${REPO_ROOT}/bench/replay_profile.py" --engine "${engine}" --messages-json "${REPLAY_TURN}" \
    --out-dir "${RESULTS}/profile-${engine}/replay" --server-trace-dir "${RESULTS}/${RUN_ID}/torch_profile"
  stop_server
}

online_window() {
  local engine="$1" conc="$2"
  start_profile_server "${engine}" "online-c${conc}" || return 1
  RUN_TAG=online CONCURRENCY="${conc}" "${REPO_ROOT}/experiments/run_baseline.sh" "${engine}" smoke &
  local job_pid=$!
  sleep "${PROFILE_AFTER_S}"
  "${REPO_ROOT}/bench/profile_online.sh" "${engine}" "${PROFILE_DIR}" "${PROFILE_WINDOW_S}"
  wait "${job_pid}"
  stop_server
}

# One concurrency-4 window under nsys, with py-spy on the engine process over the same window.
kernel_timeline() {
  local engine="$1"
  use_env "${engine}"
  export EVICTION_POLICY=lru PREEMPTION_POLICY
  PREEMPTION_POLICY="$(preempt_of "${engine}")"
  export RUN_ID="profile-${engine}-nsys" SERVE_RUN_ID="profile-${engine}-nsys" NSYS_DELAY_S NSYS_DURATION_S
  unset PROFILE_DIR
  start_server "${engine}" bash "${REPO_ROOT}/bench/nsys_serve.sh" "${engine}" || return 1
  (
    sleep "${NSYS_DELAY_S}"
    local pid
    pid="$(pgrep -f 'EngineCore|sglang::scheduler' | head -n 1)"
    sudo "$(command -v py-spy)" record -d "${PYSPY_DURATION_S}" -p "${pid}" -o "${RESULTS}/${RUN_ID}/pyspy.svg"
  ) &
  PYSPY_PID=$!
  RUN_TAG=nsys CONCURRENCY=4 "${REPO_ROOT}/experiments/run_baseline.sh" "${engine}" smoke
  wait "${PYSPY_PID}" 2>/dev/null || true
  PYSPY_PID=""
  stop_server
}

log "run_all start"
for engine in ${ENGINES}; do step "baselines-${engine}" baselines "${engine}"; done
for engine in ${ENGINES}; do step "capacity-${engine}" capacity "${engine}"; done
for engine in ${ENGINES}; do step "capacity-nospec-${engine}" capacity_no_speculator "${engine}"; done
for engine in sglang vllm; do step "eviction-${engine}" eviction "${engine}"; done
for engine in ${ENGINES}; do
  step "offline-profile-${engine}" offline_profile "${engine}"
  step "replay-profile-${engine}" replay_profile "${engine}"
  step "online-window-c1-${engine}" online_window "${engine}" 1
  step "online-window-c4-${engine}" online_window "${engine}" 4
  step "kernel-timeline-${engine}" kernel_timeline "${engine}"
done
step "tar-results" tar czf /data/results.tgz -C /data cse291-results
log "run_all done"
