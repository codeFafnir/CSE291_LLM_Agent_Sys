"""vllm-metal long-context check on Apple silicon: Qwen3-4B 4-bit plus a Qwen3-0.6B draft model.
vllm-metal has no EAGLE3 path, so the drafter is a small model. Same prompts and rows as run_vllm.py.
Run with the Homebrew vllm-metal Python, not the uv environment.
"""

import time
from pathlib import Path

from vllm import LLM, SamplingParams

from long_prompts import LongPrompt, long_prompts
from metrics import (
    SAMPLE_INTERVAL_S,
    BenchRow,
    format_row,
    start_gpu_sampler,
    stop_gpu_sampler,
    unified_used_bytes,
)
from run_vllm import _row, write_report

TARGET_MODEL = "mlx-community/Qwen3-4B-4bit"
DRAFT_MODEL = "mlx-community/Qwen3-0.6B-4bit"
# Prompts are at most 5.8K tokens with output. 8192 keeps both KV pools resident on 24GB.
MAX_MODEL_LEN = 8192
MAX_NEW_TOKENS = 128
GPU_MEMORY_UTILIZATION = 0.40
SPECULATIVE_TOKENS = 3
RESULT_PATH = Path(__file__).resolve().parent / "results" / "vllm_metal_qwen3_4b_4bit_draft.json"


def greedy_params(max_tokens: int) -> SamplingParams:
    # vllm-metal only speculates on greedy requests.
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        ignore_eos=True,
    )


def build_engine(**overrides: object) -> LLM:
    settings: dict[str, object] = {
        "model": TARGET_MODEL,
        "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "max_num_seqs": 1,
        "enable_prefix_caching": False,
        # RequestOutput.metrics (TTFT, token timestamps) is None while stats are off.
        "disable_log_stats": False,
        # vllm-metal speculation runs only with async scheduling off.
        "async_scheduling": False,
        "spec_method": "draft_model",
        "spec_model": DRAFT_MODEL,
        "spec_tokens": SPECULATIVE_TOKENS,
    }
    settings.update(overrides)
    return LLM(**settings)


def run(prompts: tuple[LongPrompt, ...]) -> tuple[BenchRow, ...]:
    engine = build_engine(per_request_spec_decode_metrics="summary")
    # First Metal dispatch compiles shaders. Keep that out of the first row.
    engine.generate([prompts[0].rendered()], greedy_params(1), use_tqdm=False)
    rows: list[BenchRow] = []
    for prompt in prompts:
        stop, thread, peak = start_gpu_sampler(SAMPLE_INTERVAL_S, unified_used_bytes)
        started = time.perf_counter()
        try:
            outputs = engine.generate([prompt.rendered()], greedy_params(MAX_NEW_TOKENS), use_tqdm=False)
        finally:
            e2e_s = time.perf_counter() - started
            peak_bytes = stop_gpu_sampler(stop, thread, peak)
        rows.append(_row(prompt, e2e_s, peak_bytes, outputs[0]))
        print(format_row(rows[-1]))
    return tuple(rows)


def main() -> None:
    rows = run(long_prompts())
    write_report(rows, RESULT_PATH)
    print(f"wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
