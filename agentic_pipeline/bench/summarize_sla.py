"""Turn server_metrics.jsonl plus a Harbor job dir into summary.json, or capacity_log.jsonl into capacity_summary.json.

  summarize_sla.py run --engine vllm --run-dir <dir>
  summarize_sla.py capacity --run-dir <dir>
"""

import argparse
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from capacity_types import CapacityRow

Engine = Literal["vllm", "sglang"]
QUANTILES: tuple[float, ...] = (0.50, 0.95, 0.99)
CONTEXT_ERROR_MARKERS: tuple[str, ...] = (
    "ContextLengthExceeded",
    "context length",
    "maximum context",
    "context_length_exceeded",
)
MODELS: dict[Engine, tuple[str, str, int]] = {
    "vllm": ("Qwen/Qwen3-14B-AWQ", "RedHatAI/Qwen3-14B-speculator.eagle3", 3),
    "sglang": ("Qwen/Qwen3-14B-AWQ", "thoughtworks/Qwen3-14B-Eagle3", 3),
}
KV_DTYPE = "float16"


class Sample(BaseModel):
    model_config = ConfigDict(frozen=True)
    ts: float
    gpu_used_mib: float
    series: dict[str, float]


class Latency(BaseModel):
    mean: float | None
    p50: float | None
    p95: float | None
    p99: float | None


class SpecStats(BaseModel):
    mean_acceptance_length: float | None
    draft_acceptance_rate: float | None
    accepted_per_position: list[float]


class TaskScore(BaseModel):
    trials: int
    solved: int
    mean_reward: float | None
    context_length_errors: int


class PromptTokens(BaseModel):
    p99: float | None
    max_bucket_upper: float | None


class Summary(BaseModel):
    engine: Engine
    run_id: str
    config: dict[str, str]
    wall_clock_s: float
    requests_completed: float
    output_tokens: float
    requests_per_s: float
    output_tokens_per_s: float
    peak_gpu_mib: float
    ttft_s: Latency
    tpot_s: Latency
    e2e_s: Latency
    spec: SpecStats
    prefix_cache_hit_rate: float | None
    preemptions: float
    prompt_tokens: PromptTokens
    task_score: TaskScore | None


def load_samples(path: Path) -> list[Sample]:
    return [Sample.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def delta(first: Sample, last: Sample, key: str) -> float:
    return last.series.get(key, 0.0) - first.series.get(key, 0.0)


def gauge_mean(samples: list[Sample], key: str) -> float | None:
    values = [s.series[key] for s in samples if s.series.get(key, 0.0) > 0.0]
    return sum(values) / len(values) if values else None


def _le(key: str) -> float:
    bound = key.split("le=", 1)[1].rstrip("}")
    return float("inf") if bound == "+Inf" else float(bound)


def bucket_deltas(first: Sample, last: Sample, name: str) -> list[tuple[float, float]]:
    prefix = f"{name}_bucket{{le="
    keys = {key for key in last.series if key.startswith(prefix)}
    return sorted((_le(key), delta(first, last, key)) for key in keys)


def quantile(buckets: list[tuple[float, float]], q: float) -> float | None:
    # Linear interpolation inside the bucket that holds the q-th observation.
    if not buckets or buckets[-1][1] <= 0.0:
        return None
    target = q * buckets[-1][1]
    lower_bound, lower_count = 0.0, 0.0
    for upper_bound, count in buckets:
        if count >= target:
            if upper_bound == float("inf"):
                return lower_bound
            span = count - lower_count
            if span <= 0.0:
                return upper_bound
            return lower_bound + (upper_bound - lower_bound) * (target - lower_count) / span
        lower_bound, lower_count = upper_bound, count
    return lower_bound


def latency(first: Sample, last: Sample, name: str) -> Latency:
    count = delta(first, last, f"{name}_count")
    total = delta(first, last, f"{name}_sum")
    buckets = bucket_deltas(first, last, name)
    p50, p95, p99 = (quantile(buckets, q) for q in QUANTILES)
    return Latency(mean=total / count if count > 0.0 else None, p50=p50, p95=p95, p99=p99)


def max_bucket_upper(buckets: list[tuple[float, float]]) -> float | None:
    previous = 0.0
    top: float | None = None
    for upper_bound, count in buckets:
        if count > previous:
            top = upper_bound
        previous = count
    return top


def vllm_spec(first: Sample, last: Sample) -> SpecStats:
    drafts = delta(first, last, "vllm:spec_decode_num_drafts_total")
    draft_tokens = delta(first, last, "vllm:spec_decode_num_draft_tokens_total")
    accepted = delta(first, last, "vllm:spec_decode_num_accepted_tokens_total")
    prefix = "vllm:spec_decode_num_accepted_tokens_per_pos_total{position="
    positions = sorted(int(key[len(prefix):].rstrip("}")) for key in last.series if key.startswith(prefix))
    per_position = [
        delta(first, last, f"{prefix}{p}}}") / drafts if drafts > 0.0 else 0.0 for p in positions
    ]
    return SpecStats(
        mean_acceptance_length=1.0 + accepted / drafts if drafts > 0.0 else None,
        draft_acceptance_rate=accepted / draft_tokens if draft_tokens > 0.0 else None,
        accepted_per_position=per_position,
    )


def sglang_spec(samples: list[Sample]) -> SpecStats:
    return SpecStats(
        mean_acceptance_length=gauge_mean(samples, "sglang:spec_accept_length"),
        draft_acceptance_rate=gauge_mean(samples, "sglang:spec_accept_rate"),
        accepted_per_position=[],
    )


def read_env_file(path: Path) -> dict[str, str]:
    pairs = (line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
    return {key: value for key, value in pairs}


class _Verifier(BaseModel):
    rewards: dict[str, float] | None = None


class _Trial(BaseModel):
    verifier_result: _Verifier | None = None
    exception_info: object = None


def task_score(harbor_dir: Path) -> TaskScore | None:
    results = sorted(harbor_dir.glob("*/result.json"))
    if not results:
        return None
    rewards: list[float] = []
    context_errors = 0
    for path in results:
        text = path.read_text()
        trial = _Trial.model_validate_json(text)
        if trial.exception_info is not None and any(m in str(trial.exception_info) for m in CONTEXT_ERROR_MARKERS):
            context_errors += 1
        if trial.verifier_result is not None and trial.verifier_result.rewards:
            rewards.append(trial.verifier_result.rewards.get("reward", 0.0))
    solved = sum(1 for r in rewards if r >= 1.0)
    return TaskScore(
        trials=len(results),
        solved=solved,
        mean_reward=sum(rewards) / len(rewards) if rewards else None,
        context_length_errors=context_errors,
    )


def summarize_run(engine: Engine, run_dir: Path) -> Summary:
    samples = load_samples(run_dir / "server_metrics.jsonl")
    first, last = samples[0], samples[-1]
    wall = last.ts - first.ts
    config = read_env_file(run_dir / "serve_config.env")
    target, speculator, draft_tokens = MODELS[engine]
    config.update(target=target, speculator=speculator, draft_tokens=str(draft_tokens), kv_dtype=KV_DTYPE)
    if engine == "vllm":
        ttft_name, tpot_name, e2e_name = (
            "vllm:time_to_first_token_seconds",
            "vllm:request_time_per_output_token_seconds",
            "vllm:e2e_request_latency_seconds",
        )
        requests = delta(first, last, "vllm:request_success_total")
        tokens = delta(first, last, "vllm:generation_tokens_total")
        queries = delta(first, last, "vllm:prefix_cache_queries_total")
        hit_rate = delta(first, last, "vllm:prefix_cache_hits_total") / queries if queries > 0.0 else None
        preemptions = delta(first, last, "vllm:num_preemptions_total")
        spec = vllm_spec(first, last)
        buckets = bucket_deltas(first, last, "vllm:request_prompt_tokens")
        prompt = PromptTokens(p99=quantile(buckets, 0.99), max_bucket_upper=max_bucket_upper(buckets))
    else:
        ttft_name, tpot_name, e2e_name = (
            "sglang:time_to_first_token_seconds",
            "sglang:inter_token_latency_seconds",
            "sglang:e2e_request_latency_seconds",
        )
        requests = delta(first, last, "sglang:num_requests_total")
        tokens = delta(first, last, "sglang:generation_tokens_total")
        hit_rate = gauge_mean(samples, "sglang:cache_hit_rate")
        preemptions = delta(first, last, "sglang:num_retracted_requests_total")
        spec = sglang_spec(samples)
        prompt = PromptTokens(p99=None, max_bucket_upper=None)
    return Summary(
        engine=engine,
        run_id=run_dir.name,
        config=config,
        wall_clock_s=wall,
        requests_completed=requests,
        output_tokens=tokens,
        requests_per_s=requests / wall if wall > 0.0 else 0.0,
        output_tokens_per_s=tokens / wall if wall > 0.0 else 0.0,
        peak_gpu_mib=max(s.gpu_used_mib for s in samples),
        ttft_s=latency(first, last, ttft_name),
        tpot_s=latency(first, last, tpot_name),
        e2e_s=latency(first, last, e2e_name),
        spec=spec,
        prefix_cache_hit_rate=hit_rate,
        preemptions=preemptions,
        prompt_tokens=prompt,
        task_score=task_score(run_dir / "harbor"),
    )


class CapacityConfigSummary(BaseModel):
    engine: Engine
    max_model_len: int
    max_num_seqs: int
    max_batched_tokens: int
    eviction_policy: str
    preemption_policy: str
    largest_ok_prompt_by_concurrency: dict[int, int]
    first_trouble: str | None


def load_capacity(path: Path) -> list[CapacityRow]:
    return [CapacityRow.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def summarize_capacity(rows: list[CapacityRow]) -> list[CapacityConfigSummary]:
    groups: dict[tuple[Engine, int, int, int, str, str], list[CapacityRow]] = {}
    for row in rows:
        key = (
            row.engine,
            row.max_model_len,
            row.max_num_seqs,
            row.max_batched_tokens,
            row.eviction_policy,
            row.preemption_policy,
        )
        groups.setdefault(key, []).append(row)
    summaries: list[CapacityConfigSummary] = []
    for (engine, model_len, seqs, batched, eviction, preemption), cells in groups.items():
        largest: dict[int, int] = {}
        for cell in cells:
            if cell.outcome == "ok":
                largest[cell.concurrency] = max(largest.get(cell.concurrency, 0), cell.prompt_tokens)
        trouble = [c for c in cells if c.outcome in ("preempted", "oom_runtime", "oom_startup")]
        trouble.sort(key=lambda c: (c.prompt_tokens, c.concurrency))
        first = trouble[0] if trouble else None
        summaries.append(
            CapacityConfigSummary(
                engine=engine,
                max_model_len=model_len,
                max_num_seqs=seqs,
                max_batched_tokens=batched,
                eviction_policy=eviction,
                preemption_policy=preemption,
                largest_ok_prompt_by_concurrency=dict(sorted(largest.items())),
                first_trouble=(
                    f"{first.outcome} at prompt_tokens={first.prompt_tokens} concurrency={first.concurrency}"
                    if first
                    else None
                ),
            )
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--engine", choices=("vllm", "sglang"), required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    capacity = sub.add_parser("capacity")
    capacity.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "run":
        summary = summarize_run(args.engine, args.run_dir)
        out = args.run_dir / "summary.json"
        out.write_text(summary.model_dump_json(indent=2))
    else:
        rows = load_capacity(args.run_dir / "capacity_log.jsonl")
        out = args.run_dir / "capacity_summary.json"
        out.write_text(json.dumps([s.model_dump() for s in summarize_capacity(rows)], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
