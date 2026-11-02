#!/usr/bin/env bash
# Baseline SGLang server: Qwen3-14B-AWQ plus the EAGLE3 head, fp16 KV, radix cache on, no offload.
# Run from env-sglang. Needs RUN_ID, MAX_MODEL_LEN, MAX_NUM_SEQS, MAX_BATCHED_TOKENS,
# EVICTION_POLICY (lru lfu slru priority tlru agent) and PREEMPTION_POLICY (length or priority).
# tlru also needs TLRU_THRESHOLD and TLRU_NEXT_PROMPT, both in tokens.
# agent (cse291-agent-evict branch) also needs AGENT_IDLE_S and AGENT_SHARED_HITS.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/serve_env.sh"

extra_args=()
if [[ "${EVICTION_POLICY}" == "tlru" ]]; then
  require_env TLRU_THRESHOLD TLRU_NEXT_PROMPT
  extra_args+=(--radix-eviction-policy-config
    "{\"threshold\": ${TLRU_THRESHOLD}, \"next_prompt_estimate\": ${TLRU_NEXT_PROMPT}}")
fi
if [[ "${EVICTION_POLICY}" == "agent" ]]; then
  require_env AGENT_IDLE_S AGENT_SHARED_HITS
  extra_args+=(--radix-eviction-policy-config
    "{\"idle_s\": ${AGENT_IDLE_S}, \"shared_hits\": ${AGENT_SHARED_HITS}}")
fi
if [[ "${PREEMPTION_POLICY}" == "priority" ]]; then
  extra_args+=(--enable-priority-scheduling)
fi

# NO_SPECULATOR=1 drops the EAGLE3 head to measure its memory cost.
spec_args=(--speculative-algorithm EAGLE3
  --speculative-draft-model-path thoughtworks/Qwen3-14B-Eagle3
  --speculative-num-steps 3
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4)
if [[ -n "${NO_SPECULATOR:-}" ]]; then spec_args=(); fi

# The EAGLE3 draft config declares a 2048 context; the target model supports the longer one.
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

python -m sglang.launch_server \
  --host 0.0.0.0 --port 30000 \
  --model-path Qwen/Qwen3-14B-AWQ \
  --dtype half --quantization awq \
  --context-length "${MAX_MODEL_LEN}" \
  --max-running-requests "${MAX_NUM_SEQS}" \
  --chunked-prefill-size "${MAX_BATCHED_TOKENS}" \
  --mem-fraction-static 0.85 \
  --radix-eviction-policy "${EVICTION_POLICY}" \
  --retraction-policy "${PREEMPTION_POLICY}" \
  --enable-metrics \
  "${spec_args[@]}" \
  "${extra_args[@]}" \
  2>&1 | tee -a "${SERVE_LOG}"
