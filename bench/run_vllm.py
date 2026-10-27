"""vLLM long-context check: Qwen3-14B AWQ plus the Red Hat EAGLE3 speculator.
float16 because a 24GB RTX 6000 is Turing. The head was trained on the bf16 target.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

from long_prompts import LongPrompt, long_prompts

TARGET_MODEL = "Qwen/Qwen3-14B-AWQ"
SPECULATOR_MODEL = "RedHatAI/Qwen3-14B-speculator.eagle3"
MAX_MODEL_LEN = 12288
MAX_NEW_TOKENS = 128
GPU_MEMORY_UTILIZATION = 0.90
SPECULATIVE_TOKENS = 3
RESULT_PATH = Path(__file__).resolve().parent / "results" / "vllm_qwen3_14b_awq_eagle3.json"


@dataclass(frozen=True)
class RunRow:
    prompt_id: str
    prompt_tokens: int
    output_tokens: int
    latency_s: float
    first_token_s: float
    accept_length: float | None
    accept_rate: float | None


def _row(prompt: LongPrompt, started: float, finished: float, output: RequestOutput) -> RunRow:
    prompt_token_ids = output.prompt_token_ids or ()
    prompt_tokens = len(prompt_token_ids)
    completion = output.outputs[0]
    output_tokens = len(completion.token_ids)
    metrics = output.metrics
    first_token_s = float(metrics.first_token_latency) if metrics is not None else 0.0
    spec = completion.spec_decode_metrics
    accept_length = None
    accept_rate = None
    if spec is not None:
        payload = spec.to_dict()
        accept_length = float(payload["mean_acceptance_length"])
        accept_rate = float(payload["draft_acceptance_rate"])
    return RunRow(
        prompt_id=prompt.prompt_id,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        latency_s=finished - started,
        first_token_s=first_token_s,
        accept_length=accept_length,
        accept_rate=accept_rate,
    )


def run(prompts: tuple[LongPrompt, ...]) -> tuple[RunRow, ...]:
    engine = LLM(
        model=TARGET_MODEL,
        dtype="float16",
        quantization="awq",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_num_seqs=1,
        enable_prefix_caching=False,
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
    rows: list[RunRow] = []
    for prompt in prompts:
        started = time.perf_counter()
        outputs = engine.generate([prompt.rendered()], params, use_tqdm=False)
        finished = time.perf_counter()
        rows.append(_row(prompt, started, finished, outputs[0]))
        print(
            f"{prompt.prompt_id} prompt_tokens={rows[-1].prompt_tokens} "
            f"latency_s={rows[-1].latency_s:.2f} accept_length={rows[-1].accept_length}"
        )
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
