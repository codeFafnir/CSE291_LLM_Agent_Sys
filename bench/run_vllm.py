"""vLLM long-context check: Qwen3-14B AWQ plus the Red Hat EAGLE3 speculator.
float16 because a 24GB RTX 6000 is Turing. The head was trained on the bf16 target.
"""

import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

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
SPECULATOR_MODEL = "RedHatAI/Qwen3-14B-speculator.eagle3"
MAX_MODEL_LEN = 12288
MAX_NEW_TOKENS = 128
GPU_MEMORY_UTILIZATION = 0.90
SPECULATIVE_TOKENS = 3
RESULT_PATH = Path(__file__).resolve().parent / "results" / "vllm_qwen3_14b_awq_eagle3.json"


def _clocks(output: RequestOutput, output_tokens: int, e2e_s: float) -> tuple[float, float | None]:
    metrics = output.metrics
    if metrics is None or metrics.first_token_latency <= 0.0:
        return e2e_s, None
    ttft = float(metrics.first_token_latency)
    if metrics.last_token_ts > metrics.first_token_ts and output_tokens > 1:
        decode_s = metrics.last_token_ts - metrics.first_token_ts
        return ttft, decode_s / (output_tokens - 1)
    return ttft, tpot_s(ttft, e2e_s, output_tokens)


def _row(prompt: LongPrompt, e2e_s: float, peak_bytes: int, output: RequestOutput) -> BenchRow:
    prompt_tokens = len(output.prompt_token_ids or ())
    completion = output.outputs[0]
    output_tokens = len(completion.token_ids)
    ttft, tpot = _clocks(output, output_tokens, e2e_s)
    spec = completion.spec_decode_metrics
    accept_length = None
    accept_rate = None
    if spec is not None:
        payload = spec.to_dict()
        accept_length = float(payload["mean_acceptance_length"])
        accept_rate = float(payload["draft_acceptance_rate"])
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


def run(prompts: tuple[LongPrompt, ...]) -> tuple[BenchRow, ...]:
    engine = LLM(
        model=TARGET_MODEL,
        dtype="float16",
        quantization="awq",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_num_seqs=1,
        enable_prefix_caching=False,
        # RequestOutput.metrics (TTFT, token timestamps) is None while stats are off.
        disable_log_stats=False,
        spec_method="eagle3",
        spec_model=SPECULATOR_MODEL,
        spec_tokens=SPECULATIVE_TOKENS,
        per_request_spec_decode_metrics="summary",
    )
    params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=MAX_NEW_TOKENS,
        ignore_eos=True,
    )
    rows: list[BenchRow] = []
    for prompt in prompts:
        stop, thread, peak = start_gpu_sampler(SAMPLE_INTERVAL_S)
        started = time.perf_counter()
        try:
            outputs = engine.generate([prompt.rendered()], params, use_tqdm=False)
        finally:
            e2e_s = time.perf_counter() - started
            peak_bytes = stop_gpu_sampler(stop, thread, peak)
        rows.append(_row(prompt, e2e_s, peak_bytes, outputs[0]))
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
