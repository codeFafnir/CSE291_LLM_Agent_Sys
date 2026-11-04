#!/usr/bin/env bash
# vLLM server for the SPEED-Bench runs: Qwen3-14B-AWQ plus the EAGLE3 head, fp16 KV, prefix caching on.
# Run from the vLLM env. Needs RESULTS_ROOT, RUN_ID, MAX_MODEL_LEN, MAX_NUM_SEQS, MAX_BATCHED_TOKENS, HF_HOME.
# No OTel collector (needs Docker). The torch profiler stays idle until /start_profile.
# NO_SPECULATOR=1 drops the EAGLE3 head.
set -euo pipefail

SCHEDULING_POLICY=fcfs
EVICTION_POLICY=lru

for name in RESULTS_ROOT RUN_ID MAX_MODEL_LEN MAX_NUM_SEQS MAX_BATCHED_TOKENS HF_HOME; do
  if [[ -z "${!name:-}" ]]; then echo "missing required env var: ${name}" >&2; exit 2; fi
done

# Record whatever vLLM is in this env. DataHub is CUDA 12.9, where the fork has no prebuilt wheel.
INSTALLED_VLLM_VERSION="$(python -c 'import vllm; print(vllm.__version__)')"

RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"
SERVE_LOG="${RUN_DIR}/serve.log"
mkdir -p "${RUN_DIR}"

# Record the exact knobs next to the log.
cat > "${RUN_DIR}/serve_config.env" <<EOF
RUN_ID=${RUN_ID}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
MAX_NUM_SEQS=${MAX_NUM_SEQS}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS}
SCHEDULING_POLICY=${SCHEDULING_POLICY}
EVICTION_POLICY=${EVICTION_POLICY}
VLLM_VERSION=${INSTALLED_VLLM_VERSION}
EOF

export VLLM_KV_EVICTION_POLICY="${EVICTION_POLICY}"
# DEV_MODE exposes /reset_prefix_cache.
export VLLM_SERVER_DEV_MODE=1
profile_args=(--profiler-config
  "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${RUN_DIR}/torch_profile\",\"torch_profiler_record_shapes\":true,\"torch_profiler_with_stack\":true,\"torch_profiler_use_gzip\":true}")

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
  --scheduling-policy "${SCHEDULING_POLICY}" \
  "${spec_args[@]}" \
  "${profile_args[@]}" \
  2>&1 | tee -a "${SERVE_LOG}"
