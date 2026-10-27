"""Per-prompt Metal frame captures for mlx-lm. Prefill and decode are separate .gputrace bundles.

Prefill covers the prompt pass and the first verify round, up to the first token.
Decode covers the next PROFILE_DECODE_TOKENS tokens.
Needs MTL_CAPTURE_ENABLED=1. A capture snapshots every buffer it touches, weights included,
so only PROFILE_PROMPT_COUNT prompts are captured.
Traces default to /tmp/llm-traces. Override with PROFILE_DIR. Open a bundle in Xcode.
"""

import os
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import speculative_generate_step

from long_prompts import long_prompts
from run_mlx import SPECULATIVE_TOKENS, generate, load_models, prompt_ids

TRACE_ROOT = os.environ.get("PROFILE_DIR", "/tmp/llm-traces")
PROFILE_PROMPT_COUNT = 1
PROFILE_DECODE_TOKENS = 16


def main() -> None:
    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        raise SystemExit("Metal capture needs MTL_CAPTURE_ENABLED=1 set before Python starts.")
    trace_dir = Path(TRACE_ROOT) / "mlx"
    trace_dir.mkdir(parents=True, exist_ok=True)
    # Metal refuses to overwrite a bundle, so each run gets its own names.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    model, draft_model, tokenizer = load_models()
    prompts = long_prompts()
    generate(model, draft_model, prompt_ids(tokenizer, prompts[0]), 1)
    for prompt in prompts[:PROFILE_PROMPT_COUNT]:
        steps = speculative_generate_step(
            prompt_ids(tokenizer, prompt),
            model,
            draft_model,
            num_draft_tokens=SPECULATIVE_TOKENS,
            max_tokens=1 + PROFILE_DECODE_TOKENS,
        )
        mx.metal.start_capture(str(trace_dir / f"{prompt.prompt_id}_prefill_{stamp}.gputrace"))
        next(steps)
        mx.metal.stop_capture()
        mx.metal.start_capture(str(trace_dir / f"{prompt.prompt_id}_decode_{stamp}.gputrace"))
        for _ in steps:
            pass
        mx.metal.stop_capture()
        print(f"{prompt.prompt_id} profiled")
    print(f"traces in {trace_dir}")
    print("open <trace>.gputrace in Xcode")


if __name__ == "__main__":
    main()
