"""SGLang long-context check: Qwen3-14B AWQ plus the SGLang EAGLE3 draft head.
dtype half for AWQ. The head was trained on the bf16 target. Radix cache is off.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import sglang as sgl

from long_prompts import LongPrompt, long_prompts

TARGET_MODEL = "Qwen/Qwen3-14B-AWQ"
SPECULATOR_MODEL = "thoughtworks/Qwen3-14B-Eagle3"
CONTEXT_LENGTH = 12288
MAX_NEW_TOKENS = 128
MEM_FRACTION_STATIC = 0.85
SPECULATIVE_STEPS = 3
SPECULATIVE_TOPK = 1
SPECULATIVE_DRAFT_TOKENS = 4
RESULT_PATH = Path(__file__).resolve().parent / "results" / "sglang_qwen3_14b_awq_eagle3.json"


@dataclass(frozen=True)
class RunRow:
    prompt_id: str
    prompt_tokens: int
    output_tokens: int
    latency_s: float
    accept_length: float | None
    accept_rate: float | None


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


def _row(prompt: LongPrompt, started: float, finished: float, output: dict[str, object]) -> RunRow:
    meta = output.get("meta_info")
    info: dict[str, object] = meta if isinstance(meta, dict) else {}
    return RunRow(
        prompt_id=prompt.prompt_id,
        prompt_tokens=_int(info, "prompt_tokens"),
        output_tokens=_int(info, "completion_tokens"),
        latency_s=finished - started,
        accept_length=_number(info, "spec_accept_length"),
        accept_rate=_number(info, "spec_accept_rate"),
    )


def run(prompts: tuple[LongPrompt, ...]) -> tuple[RunRow, ...]:
    engine = sgl.Engine(
        model_path=TARGET_MODEL,
        dtype="half",
        quantization="awq",
        context_length=CONTEXT_LENGTH,
        mem_fraction_static=MEM_FRACTION_STATIC,
        max_running_requests=1,
        disable_radix_cache=True,
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
    rows: list[RunRow] = []
    try:
        for prompt in prompts:
            started = time.perf_counter()
            output = engine.generate(prompt=prompt.rendered(), sampling_params=params)
            finished = time.perf_counter()
            if not isinstance(output, dict):
                raise TypeError("expected one SGLang result dict for a single prompt")
            rows.append(_row(prompt, started, finished, output))
            print(
                f"{prompt.prompt_id} prompt_tokens={rows[-1].prompt_tokens} "
                f"latency_s={rows[-1].latency_s:.2f} accept_length={rows[-1].accept_length}"
            )
    finally:
        engine.shutdown()
    return tuple(rows)


def write_rows(rows: tuple[RunRow, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(row) for row in rows], indent=2))


def main() -> None:
    rows = run(long_prompts())
    write_rows(rows, RESULT_PATH)
    print(f"wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
