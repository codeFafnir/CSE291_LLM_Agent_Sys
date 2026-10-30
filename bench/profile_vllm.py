"""Per-prompt Chrome traces for TraceLens. Prefill and decode are separate files.

Prefill is a cold pass (prefix cache cleared) that emits one token.
Decode reuses that prefix, so the second trace is the 128-token generation.
Traces default to /tmp/llm-traces. Override with PROFILE_DIR.
"""

import os
import time
from pathlib import Path

from vllm import LLM, SamplingParams

from long_prompts import long_prompts
from run_vllm import (
    GPU_MEMORY_UTILIZATION,
    MAX_MODEL_LEN,
    MAX_NEW_TOKENS,
    SPECULATIVE_TOKENS,
    SPECULATOR_MODEL,
    TARGET_MODEL,
)

TRACE_ROOT = os.environ.get("PROFILE_DIR", "/tmp/llm-traces")


def _params(max_tokens: int) -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        ignore_eos=True,
    )


def main() -> None:
    trace_dir = Path(TRACE_ROOT) / "vllm"
    trace_dir.mkdir(parents=True, exist_ok=True)
    engine = LLM(
        model=TARGET_MODEL,
        dtype="float16",
        quantization="awq",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_num_seqs=1,
        enable_prefix_caching=True,
        spec_method="eagle3",
        spec_model=SPECULATOR_MODEL,
        spec_tokens=SPECULATIVE_TOKENS,
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": str(trace_dir),
            "torch_profiler_record_shapes": True,
            "torch_profiler_with_stack": True,
            "torch_profiler_use_gzip": True,
        },
    )
    prompts = long_prompts()
    engine.generate([prompts[0].rendered()], _params(1), use_tqdm=False)
    for prompt in prompts:
        engine.reset_prefix_cache()
        engine.start_profile(f"{prompt.prompt_id}_prefill")
        engine.generate([prompt.rendered()], _params(1), use_tqdm=False)
        engine.stop_profile()
        engine.start_profile(f"{prompt.prompt_id}_decode")
        started = time.perf_counter()
        engine.generate([prompt.rendered()], _params(MAX_NEW_TOKENS), use_tqdm=False)
        elapsed = time.perf_counter() - started
        engine.stop_profile()
        print(f"{prompt.prompt_id} decode_profile_s={elapsed:.2f}")
    print(f"traces in {trace_dir}")
    print("TraceLens_generate_perf_report_pytorch --profile_json_path <one trace.json.gz>")


if __name__ == "__main__":
    main()
