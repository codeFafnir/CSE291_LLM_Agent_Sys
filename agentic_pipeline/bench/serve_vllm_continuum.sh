#!/usr/bin/env bash
# Continuum vLLM server. Same knobs as serve_vllm.sh, plus the fork's
# continuum pin and LMCache CPU offload. Run from the vllm-continuum env.
# Needs the serve_vllm.sh env vars and LMCACHE_MAX_LOCAL_CPU_SIZE (GB).
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/serve_env.sh"

if [[ "${EVICTION_POLICY}" != "lru" && "${EVICTION_POLICY}" != "agent" ]]; then
  echo "EVICTION_POLICY must be lru or agent, got ${EVICTION_POLICY}" >&2
  exit 2
fi
if [[ -z "${LMCACHE_MAX_LOCAL_CPU_SIZE:-}" ]]; then
  echo "missing required env var: LMCACHE_MAX_LOCAL_CPU_SIZE" >&2
  exit 2
fi
export VLLM_KV_EVICTION_POLICY="${EVICTION_POLICY}"
export LMCACHE_MAX_LOCAL_CPU_SIZE
printf 'SCHEDULING_POLICY=continuum\nLMCACHE_MAX_LOCAL_CPU_SIZE=%s\n' "${LMCACHE_MAX_LOCAL_CPU_SIZE}" >> "${RUN_DIR}/serve_config.env"

# Recorder writes scheduler_timestamps here, including across a long run.
export RUN_OUTPUT_DIR="${RUN_DIR}"

export VLLM_SERVER_DEV_MODE=1
profile_args=(--profiler-config
  "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${RUN_DIR}/torch_profile\",\"torch_profiler_record_shapes\":true,\"torch_profiler_with_stack\":true,\"torch_profiler_use_gzip\":true}")

bash "$(dirname "${BASH_SOURCE[0]}")/otel_collector.sh"
trace_args=(--otlp-traces-endpoint http://127.0.0.1:4317)

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
  --scheduling-policy continuum \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  "${spec_args[@]}" \
  "${profile_args[@]}" \
  "${trace_args[@]}" \
  2>&1 | tee -a "${SERVE_LOG}"
