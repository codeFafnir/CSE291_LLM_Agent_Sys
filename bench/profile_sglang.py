"""Per-prompt Chrome traces for TraceLens. SGLang splits prefill and decode.

One generate per prompt. profile_by_stage writes two traces:
<id>-<id>-TP-0-EXTEND.trace.json.gz is prefill, DECODE is decode.
Cache is flushed first so the prefill is cold. Traces default to /tmp/llm-traces.
"""

import os
from pathlib import Path

import sglang as sgl

from long_prompts import long_prompts
from run_sglang import (
    CONTEXT_LENGTH,
    MAX_NEW_TOKENS,
    MEM_FRACTION_STATIC,
    SPECULATIVE_DRAFT_TOKENS,
    SPECULATIVE_STEPS,
    SPECULATIVE_TOPK,
    SPECULATOR_MODEL,
    TARGET_MODEL,
)

TRACE_ROOT = os.environ.get("PROFILE_DIR", "/tmp/llm-traces")


def _stop(engine: sgl.Engine) -> None:
    try:
        engine.stop_profile()
    except RuntimeError as exc:
        if "not in progress" not in str(exc):
            raise


def main() -> None:
    trace_dir = Path(TRACE_ROOT) / "sglang"
    trace_dir.mkdir(parents=True, exist_ok=True)
    engine = sgl.Engine(
        model_path=TARGET_MODEL,
        dtype="half",
        quantization="awq",
        context_length=CONTEXT_LENGTH,
        mem_fraction_static=MEM_FRACTION_STATIC,
        max_running_requests=1,
        disable_radix_cache=False,
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path=SPECULATOR_MODEL,
        speculative_num_steps=SPECULATIVE_STEPS,
        speculative_eagle_topk=SPECULATIVE_TOPK,
        speculative_num_draft_tokens=SPECULATIVE_DRAFT_TOKENS,
    )
    params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": MAX_NEW_TOKENS,
        "ignore_eos": True,
    }
    prompts = long_prompts()
    try:
        warmup = dict(params)
        warmup["max_new_tokens"] = 1
        engine.generate(prompt=prompts[0].rendered(), sampling_params=warmup)
        for prompt in prompts:
            engine.flush_cache()
            engine.start_profile(
                output_dir=str(trace_dir),
                num_steps=MAX_NEW_TOKENS,
                activities=["CPU", "GPU"],
                profile_by_stage=True,
                with_stack=True,
                record_shapes=True,
                profile_id=prompt.prompt_id,
                profile_prefix=prompt.prompt_id,
                merge_profiles=False,
                detailed_annotations=True,
            )
            engine.generate(prompt=prompt.rendered(), sampling_params=params)
            _stop(engine)
            print(f"{prompt.prompt_id} profiled")
    finally:
        engine.shutdown()
    print(f"traces in {trace_dir}")
    print("EXTEND files are prefill. DECODE files are decode.")
    print("TraceLens_generate_perf_report_pytorch --profile_json_path <one trace.json.gz>")


if __name__ == "__main__":
    main()
