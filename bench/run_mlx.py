"""MLX long-context check on Apple silicon: Qwen3-4B 4-bit plus a Qwen3-0.6B draft model.
mlx-lm has no paged KV or batching. This is the single-stream baseline for run_vllm_metal.py.
Run in the uv environment. Greedy with no EOS stop, so every prompt decodes MAX_NEW_TOKENS.
"""

import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.generate import speculative_generate_step

from long_prompts import LongPrompt, long_prompts
from metrics import (
    SAMPLE_INTERVAL_S,
    BenchRow,
    decode_tok_s,
    format_row,
    format_summary,
    mib,
    prefill_tok_s,
    row_dict,
    start_gpu_sampler,
    stop_gpu_sampler,
    summarize,
    tpot_s,
    unified_used_bytes,
)

TARGET_MODEL = "mlx-community/Qwen3-4B-4bit"
DRAFT_MODEL = "mlx-community/Qwen3-0.6B-4bit"
MAX_NEW_TOKENS = 128
SPECULATIVE_TOKENS = 3
RESULT_PATH = Path(__file__).resolve().parent / "results" / "mlx_qwen3_4b_4bit_draft.json"


def _acceptance(from_draft: tuple[bool, ...], max_tokens: int) -> tuple[float | None, float | None]:
    # Rebuild speculative_generate_step rounds: each drafts min(remaining, k) tokens,
    # yields the accepted drafts, then one target token unless max_tokens was reached.
    # Same definitions as vLLM: length = 1 + accepted / rounds, rate = accepted / drafted.
    accepted = drafted = rounds = produced = 0
    round_open = False
    for from_drafter in from_draft:
        if not round_open:
            rounds += 1
            drafted += min(max_tokens - produced, SPECULATIVE_TOKENS)
            round_open = True
        produced += 1
        if from_drafter:
            accepted += 1
        else:
            round_open = False
    if rounds == 0 or drafted == 0:
        return None, None
    return 1.0 + accepted / rounds, accepted / drafted


def generate(
    model: nn.Module, draft_model: nn.Module, prompt_ids: mx.array, max_tokens: int
) -> tuple[float, float, tuple[bool, ...]]:
    # Tokens arrive as Python values, so each yield is already evaluated on the GPU.
    started = time.perf_counter()
    first_s = last_s = 0.0
    from_draft: list[bool] = []
    for _, _, from_drafter in speculative_generate_step(
        prompt_ids,
        model,
        draft_model,
        num_draft_tokens=SPECULATIVE_TOKENS,
        max_tokens=max_tokens,
    ):
        last_s = time.perf_counter() - started
        if not from_draft:
            first_s = last_s
        from_draft.append(bool(from_drafter))
    return first_s, last_s, tuple(from_draft)


def _row(
    prompt: LongPrompt,
    prompt_tokens: int,
    e2e_s: float,
    peak_bytes: int,
    timing: tuple[float, float, tuple[bool, ...]],
) -> BenchRow:
    ttft, last_s, from_draft = timing
    output_tokens = len(from_draft)
    tpot = tpot_s(ttft, last_s, output_tokens)
    accept_length, accept_rate = _acceptance(from_draft, MAX_NEW_TOKENS)
    return BenchRow(
        prompt_id=prompt.prompt_id,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        e2e_s=e2e_s,
        ttft_s=ttft,
        tpot_s=tpot,
        prefill_tok_s=prefill_tok_s(prompt_tokens, ttft),
        decode_tok_s=decode_tok_s(tpot),
        peak_gpu_mib=mib(peak_bytes),
        accept_length=accept_length,
        accept_rate=accept_rate,
    )


def load_models() -> tuple[nn.Module, nn.Module, object]:
    model, tokenizer = load(TARGET_MODEL)
    draft_model, _ = load(DRAFT_MODEL)
    return model, draft_model, tokenizer


def prompt_ids(tokenizer: object, prompt: LongPrompt) -> mx.array:
    # rendered() already carries the Qwen3 chat markers.
    return mx.array(tokenizer.encode(prompt.rendered(), add_special_tokens=False))


def run(prompts: tuple[LongPrompt, ...]) -> tuple[BenchRow, ...]:
    model, draft_model, tokenizer = load_models()
    # First Metal dispatch compiles shaders. Keep that out of the first row.
    generate(model, draft_model, prompt_ids(tokenizer, prompts[0]), 1)
    rows: list[BenchRow] = []
    for prompt in prompts:
        ids = prompt_ids(tokenizer, prompt)
        stop, thread, peak = start_gpu_sampler(SAMPLE_INTERVAL_S, unified_used_bytes)
        started = time.perf_counter()
        try:
            timing = generate(model, draft_model, ids, MAX_NEW_TOKENS)
        finally:
            e2e_s = time.perf_counter() - started
            peak_bytes = stop_gpu_sampler(stop, thread, peak)
        rows.append(_row(prompt, ids.size, e2e_s, peak_bytes, timing))
        print(format_row(rows[-1]))
    return tuple(rows)


def write_report(rows: tuple[BenchRow, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    payload = {"summary": summary, "rows": [row_dict(row) for row in rows]}
    path.write_text(json.dumps(payload, indent=2))
    print(format_summary(summary))


def main() -> None:
    rows = run(long_prompts())
    write_report(rows, RESULT_PATH)
    print(f"wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
