"""SGLang long-context check: Qwen3-14B AWQ plus the SGLang EAGLE3 draft head.
dtype half for AWQ. The head was trained on the bf16 target. Radix cache is off.
"""

import json
import time
from pathlib import Path

import sglang as sgl

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
)

TARGET_MODEL = "Qwen/Qwen3-14B-AWQ"
SPECULATOR_MODEL = "thoughtworks/Qwen3-14B-Eagle3"
CONTEXT_LENGTH = 12288
MAX_NEW_TOKENS = 128
MEM_FRACTION_STATIC = 0.85
SPECULATIVE_STEPS = 3
SPECULATIVE_TOPK = 1
SPECULATIVE_DRAFT_TOKENS = 4
RESULT_PATH = Path(__file__).resolve().parent / "results" / "sglang_qwen3_14b_awq_eagle3.json"


def _number(meta: dict[str, object], key: str) -> float | None:
    value = meta.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _int(meta: dict[str, object], key: str) -> int:
    value = meta.get(key)
    if isinstance(value, int):
        return value
    return 0


def _row(prompt: LongPrompt, e2e_s: float, peak_bytes: int, output: dict[str, object]) -> BenchRow:
    meta = output.get("meta_info")
    info: dict[str, object] = meta if isinstance(meta, dict) else {}
    prompt_tokens = _int(info, "prompt_tokens")
    output_tokens = _int(info, "completion_tokens")
    ttft = _number(info, "first_token_latency")
    ttft_s = ttft if ttft is not None else 0.0
    engine_e2e = _number(info, "e2e_latency")
    span = engine_e2e if engine_e2e is not None else e2e_s
    tpot = tpot_s(ttft_s, span, output_tokens)
    return BenchRow(
        prompt_id=prompt.prompt_id,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        e2e_s=span,
        ttft_s=ttft_s,
        tpot_s=tpot,
        prefill_tok_s=prefill_tok_s(prompt_tokens, ttft_s),
        decode_tok_s=decode_tok_s(tpot),
        peak_gpu_mib=mib(peak_bytes),
        accept_length=_number(info, "spec_accept_length"),
        accept_rate=_number(info, "spec_accept_rate"),
    )


def run(prompts: tuple[LongPrompt, ...]) -> tuple[BenchRow, ...]:
    engine = sgl.Engine(
        model_path=TARGET_MODEL,
        dtype="half",
        quantization="awq",
        context_length=CONTEXT_LENGTH,
        mem_fraction_static=MEM_FRACTION_STATIC,
        max_running_requests=1,
        disable_radix_cache=True,
        enable_metrics=True,
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
    rows: list[BenchRow] = []
    try:
        for prompt in prompts:
            stop, thread, peak = start_gpu_sampler(SAMPLE_INTERVAL_S)
            started = time.perf_counter()
            try:
                output = engine.generate(prompt=prompt.rendered(), sampling_params=params)
            finally:
                e2e_s = time.perf_counter() - started
                peak_bytes = stop_gpu_sampler(stop, thread, peak)
            if not isinstance(output, dict):
                raise TypeError("expected one SGLang result dict for a single prompt")
            rows.append(_row(prompt, e2e_s, peak_bytes, output))
            print(format_row(rows[-1]))
    finally:
        engine.shutdown()
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
