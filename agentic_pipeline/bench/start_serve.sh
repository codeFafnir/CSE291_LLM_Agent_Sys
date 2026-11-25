#!/usr/bin/env bash
# Runs on the GPU box. Restarts the vLLM tmux session for one policy and waits for /v1/models.
# Usage: start_serve.sh <run_id> <eviction: lru|agent> <preemption: fcfs|continuum> [continuum_ttl_s]
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 <run_id> <lru|agent> <fcfs|continuum> [continuum_ttl_s]" >&2
  exit 2
fi

run_id="$1"
eviction="$2"
preemption="$3"
ttl="${4:-}"

if [[ "${eviction}" != "lru" && "${eviction}" != "agent" ]]; then
  echo "eviction must be lru or agent, got ${eviction}" >&2
  exit 2
fi
if [[ "${preemption}" != "fcfs" && "${preemption}" != "continuum" ]]; then
  echo "preemption must be fcfs or continuum, got ${preemption}" >&2
  exit 2
fi
if [[ "${preemption}" == "continuum" && -z "${ttl}" ]]; then
  echo "continuum requires continuum_ttl_s" >&2
  exit 2
fi
if [[ "${preemption}" != "continuum" && -n "${ttl}" ]]; then
  echo "continuum_ttl_s is only valid with continuum, got ${ttl} with ${preemption}" >&2
  exit 2
fi

max_num_seqs="${MAX_NUM_SEQS:-8}"
max_batched_tokens="${MAX_BATCHED_TOKENS:-8192}"

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
serve_log="/data/cse291-results/${run_id}/serve.log"

# GPU_MEMORY_UTILIZATION=0.95 is the measured stable value; 0.98 OOMs.
# serve_vllm.sh records it in serve_config.env.
ttl_export=""
if [[ "${preemption}" == "continuum" ]]; then
  ttl_export="export CONTINUUM_TTL_S=$(printf '%q' "${ttl}")"
fi

serve_cmd="source /data/envs/vllm/bin/activate
cd $(printf '%q' "${repo_dir}")
export HF_HOME=/data/hf HUGGINGFACE_HUB_CACHE=/data/hf/hub TMPDIR=/data/tmp UV_CACHE_DIR=/data/uv-cache PATH=\$HOME/.local/bin:\$PATH
export RUN_ID=$(printf '%q' "${run_id}") MAX_MODEL_LEN=30000 MAX_NUM_SEQS=$(printf '%q' "${max_num_seqs}") MAX_BATCHED_TOKENS=$(printf '%q' "${max_batched_tokens}")
export EVICTION_POLICY=${eviction} PREEMPTION_POLICY=${preemption} GPU_MEMORY_UTILIZATION=0.95
${ttl_export}
bash bench/serve_vllm.sh"

tmux kill-session -t serve 2>/dev/null || true
pkill -f '[v]llm serve' || true

# Never start over a live server: it OOMs. Wait for the process to exit and the port to close.
for _ in $(seq 1 60); do
  if ! pgrep -f '[v]llm serve' >/dev/null && ! curl -sf localhost:8000/v1/models >/dev/null; then
    break
  fi
  sleep 1
done
if pgrep -f '[v]llm serve' >/dev/null || curl -sf localhost:8000/v1/models >/dev/null; then
  echo "old vllm serve still alive after 60s, refusing to start (run_id=${run_id})" >&2
  exit 1
fi

tmux new-session -d -s serve "${serve_cmd}"

for _ in $(seq 1 90); do
  if ! tmux has-session -t serve 2>/dev/null; then
    echo "tmux session serve exited before vLLM was ready (run_id=${run_id}); last 30 log lines:" >&2
    tail -n 30 "${serve_log}" >&2 || true
    exit 1
  fi
  if curl -sf localhost:8000/v1/models >/dev/null; then
    grep -E "Available KV cache|GPU KV cache size" "${serve_log}" | tail -n 2 || true
    exit 0
  fi
  sleep 5
done

echo "vLLM did not answer /v1/models within 450s (run_id=${run_id})" >&2
exit 1
