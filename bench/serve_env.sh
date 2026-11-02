#!/usr/bin/env bash
# Sourced by serve_*.sh. Fails when a knob is unset so every run records what it used.

require_env() {
  local name
  for name in "$@"; do
    if [[ -z "${!name:-}" ]]; then
      echo "missing required env var: ${name}" >&2
      exit 2
    fi
  done
}

RESULTS_ROOT=/data/cse291-results

require_env RUN_ID MAX_MODEL_LEN MAX_NUM_SEQS MAX_BATCHED_TOKENS EVICTION_POLICY PREEMPTION_POLICY

RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"
SERVE_LOG="${RUN_DIR}/serve.log"
mkdir -p "${RUN_DIR}"

# Record the exact knobs next to the log.
cat > "${RUN_DIR}/serve_config.env" <<EOF
RUN_ID=${RUN_ID}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
MAX_NUM_SEQS=${MAX_NUM_SEQS}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS}
EVICTION_POLICY=${EVICTION_POLICY}
PREEMPTION_POLICY=${PREEMPTION_POLICY}
EOF
