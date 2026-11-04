#!/usr/bin/env bash
# Baseline vLLM server: Qwen3-14B-AWQ plus the EAGLE3 head, fp16 KV, prefix caching on, no offload.
# Run from env-vllm. Needs RUN_ID, MAX_MODEL_LEN, MAX_NUM_SEQS, MAX_BATCHED_TOKENS,
# EVICTION_POLICY (lru, or agent from the cse291-agent-evict branch) and PREEMPTION_POLICY (fcfs or priority).
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/serve_env.sh"

if [[ "${EVICTION_POLICY}" != "lru" && "${EVICTION_POLICY}" != "agent" ]]; then
  echo "vLLM supports EVICTION_POLICY=lru or agent, got ${EVICTION_POLICY}" >&2
  exit 2
fi
export VLLM_KV_EVICTION_POLICY="${EVICTION_POLICY}"

# The torch profiler stays idle until /start_profile. DEV_MODE exposes /reset_prefix_cache.
export VLLM_SERVER_DEV_MODE=1
profile_args=(--profiler-config
  "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${RUN_DIR}/torch_profile\",\"torch_profiler_record_shapes\":true,\"torch_profiler_with_stack\":true,\"torch_profiler_use_gzip\":true}")

# Per-request spans (queue, prefill, decode, ttft, e2e) go to the OTLP collector.
bash "$(dirname "${BASH_SOURCE[0]}")/otel_collector.sh"
trace_args=(--otlp-traces-endpoint http://127.0.0.1:4317)

# NO_SPECULATOR=1 drops the EAGLE3 head to measure its memory cost.
spec_args=(--speculative-config '{"method":"eagle3","model":"RedHatAI/Qwen3-14B-speculator.eagle3","num_speculative_tokens":3}')
if [[ -n "${NO_SPECULATOR:-}" ]]; then spec_args=(); fi

vllm serve Qwen/Qwen3-14B-AWQ \
  --host 0.0.0.0 --port 8000 \
  --dtype float16 --quantization awq \
  --kv-cache-dtype float16 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --scheduling-policy "${PREEMPTION_POLICY}" \
  "${spec_args[@]}" \
  "${profile_args[@]}" \
  "${trace_args[@]}" \
  2>&1 | tee -a "${SERVE_LOG}"
