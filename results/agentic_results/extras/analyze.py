"""Insights for the nemotron-30 vLLM runs.

Tool duration is the gap from the end of one matched LLM request to the
arrival of the next request in the same trial, on steps that issued bash.
KV recompute is the prefix-cache miss on that next prefill.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

_TS = re.compile(r'"ts":\s*([0-9.]+)')
_HITS = re.compile(r'"vllm:prefix_cache_hits_total":\s*([0-9.eE+-]+)')
_QUERIES = re.compile(r'"vllm:prefix_cache_queries_total":\s*([0-9.eE+-]+)')
_PREEMPT = re.compile(r'"vllm:num_preemptions_total":\s*([0-9.eE+-]+)')
_KV = re.compile(r'"vllm:kv_cache_usage_perc":\s*([0-9.eE+-]+)')

MEDIUM_TASKS = frozenset(
    {
        "software_engineering_task_13509",
        "software_engineering_task_2621",
        "debugging_task_12771",
        "security_task_11984",
        "system_administration_task_12762",
        "software_engineering_task_15464",
        "debugging_task_18959",
        "system_administration_task_2208",
        "model_training_task_18693",
        "file_operations_task_4095",
    }
)

RUNS = (
    ("c1", "c1/baseline-vllm-nemotron-30-c1-b8-t8192-l20000-lru-20261104T051307Z"),
    ("c4", "c4/baseline-vllm-nemotron-30-c4-b8-t8192-l20000-lru-20261104T072031Z"),
)

RECOMPUTE_HIT_FRACTION = 0.5


@dataclass(frozen=True)
class AgentStep:
    task: str
    trial_name: str
    step_id: int
    ts: float
    prompt_tokens: int
    completion_tokens: int
    bash_calls: int
    complete_calls: int
    requested_wait_s: float
    window_start: float
    window_end: float


@dataclass(frozen=True)
class LlmRequest:
    request_id: str
    arrival_ts: float
    e2e_s: float
    prefill_s: float
    queue_s: float
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class MetricSample:
    ts: float
    hits: float
    queries: float
    preemptions: float
    kv_usage: float


@dataclass(frozen=True)
class TaskRow:
    task: str
    passed: bool
    exception_type: str
    seconds: float
    n_requests: int
    peak_tokens: int


@dataclass(frozen=True)
class MatchedStep:
    step: AgentStep
    request: LlmRequest


@dataclass(frozen=True)
class ToolEpisode:
    run: str
    task: str
    difficulty: str
    trial_name: str
    step_id: int
    bash_calls: int
    requested_wait_s: float
    measured_wait_s: float
    next_prompt_tokens: int
    next_prefill_s: float
    hit_tokens: float
    query_tokens: float
    hit_fraction: float
    exclusive: bool
    preempt_during_wait: float
    recomputed: bool


@dataclass(frozen=True)
class TaskTools:
    run: str
    task: str
    difficulty: str
    passed: bool
    exception_type: str
    tool_calls: int
    bash_calls: int
    complete_calls: int
    episodes: int
    mean_measured_wait_s: float
    task_seconds: float


def parse_iso(value: str) -> float:
    text = value.replace("Z", "+00:00")
    return datetime.fromisoformat(text).timestamp()


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("percentile of empty list")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean of empty list")
    return sum(values) / len(values)


def difficulty_of(task: str) -> str:
    if task in MEDIUM_TASKS:
        return "medium"
    return "easy"


def load_steps(harbor: Path) -> list[AgentStep]:
    steps: list[AgentStep] = []
    for result_path in sorted(harbor.glob("*/result.json")):
        result = json.loads(result_path.read_text())
        trajectory_path = result_path.parent / "agent" / "trajectory.json"
        if not trajectory_path.exists():
            continue
        trajectory = json.loads(trajectory_path.read_text())
        window_start = parse_iso(result["started_at"])
        window_end = parse_iso(result["finished_at"])
        task = str(result["task_name"])
        trial_name = str(result["trial_name"])
        for raw in trajectory["steps"]:
            if raw.get("source") != "agent":
                continue
            metrics = raw["metrics"]
            bash_calls = 0
            complete_calls = 0
            requested = 0.0
            for call in raw.get("tool_calls", []):
                name = call["function_name"]
                if name == "bash_command":
                    bash_calls += 1
                    requested += float(call["arguments"]["duration"])
                elif name == "mark_task_complete":
                    complete_calls += 1
            steps.append(
                AgentStep(
                    task=task,
                    trial_name=trial_name,
                    step_id=int(raw["step_id"]),
                    ts=parse_iso(raw["timestamp"]),
                    prompt_tokens=int(metrics["prompt_tokens"]),
                    completion_tokens=int(metrics["completion_tokens"]),
                    bash_calls=bash_calls,
                    complete_calls=complete_calls,
                    requested_wait_s=requested,
                    window_start=window_start,
                    window_end=window_end,
                )
            )
    return steps


def load_requests(path: Path) -> list[LlmRequest]:
    rows: list[LlmRequest] = []
    with path.open() as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                LlmRequest(
                    request_id=raw["request_id"],
                    arrival_ts=float(raw["arrival_ts"]),
                    e2e_s=float(raw["e2e_s"]),
                    prefill_s=float(raw["prefill_s"]),
                    queue_s=float(raw["queue_s"]),
                    prompt_tokens=int(float(raw["prompt_tokens"])),
                    completion_tokens=int(float(raw["completion_tokens"])),
                )
            )
    return rows


def load_tasks(path: Path) -> dict[str, TaskRow]:
    tasks: dict[str, TaskRow] = {}
    with path.open() as handle:
        for raw in csv.DictReader(handle):
            tasks[raw["task"]] = TaskRow(
                task=raw["task"],
                passed=raw["passed"] == "True",
                exception_type=raw["exception_type"],
                seconds=float(raw["seconds"]),
                n_requests=int(raw["n_requests"]),
                peak_tokens=int(float(raw["peak_tokens"])) if raw["peak_tokens"] else 0,
            )
    return tasks


def _field(pattern: re.Pattern[str], line: str) -> float:
    found = pattern.search(line)
    if found is None:
        raise ValueError(f"missing {pattern.pattern}")
    return float(found.group(1))


def load_samples(path: Path) -> list[MetricSample]:
    samples: list[MetricSample] = []
    with path.open() as handle:
        for line in handle:
            samples.append(
                MetricSample(
                    ts=_field(_TS, line),
                    hits=_field(_HITS, line),
                    queries=_field(_QUERIES, line),
                    preemptions=_field(_PREEMPT, line),
                    kv_usage=_field(_KV, line),
                )
            )
    samples.sort(key=lambda sample: sample.ts)
    return samples


def match_steps(steps: list[AgentStep], requests: list[LlmRequest]) -> list[MatchedStep]:
    """Greedy one-to-one match on token counts inside the trial window.

    The step timestamp is logged just after the request finishes, so the
    request must arrive before that timestamp and end near it.
    """
    candidates: list[tuple[float, int, int]] = []
    for step_index, step in enumerate(steps):
        for request_index, request in enumerate(requests):
            if request.prompt_tokens != step.prompt_tokens:
                continue
            if request.completion_tokens != step.completion_tokens:
                continue
            if request.arrival_ts < step.window_start - 1.0:
                continue
            if request.arrival_ts > step.window_end + 1.0:
                continue
            lag = step.ts - (request.arrival_ts + request.e2e_s)
            if lag < -2.0 or lag > 30.0:
                continue
            candidates.append((abs(lag), step_index, request_index))
    candidates.sort()
    used_steps: set[int] = set()
    used_requests: set[int] = set()
    chosen: dict[int, int] = {}
    for _lag, step_index, request_index in candidates:
        if step_index in used_steps or request_index in used_requests:
            continue
        used_steps.add(step_index)
        used_requests.add(request_index)
        chosen[step_index] = request_index
    return [
        MatchedStep(step=steps[step_index], request=requests[request_index])
        for step_index, request_index in sorted(chosen.items())
    ]


def bracket(samples: list[MetricSample], times: list[float], t0: float, t1: float) -> tuple[MetricSample, MetricSample] | None:
    right_index = bisect.bisect_left(times, t1)
    left_index = bisect.bisect_right(times, t0) - 1
    if left_index < 0 or right_index >= len(samples):
        return None
    left = samples[left_index]
    right = samples[right_index]
    if right.ts < left.ts:
        return None
    return left, right


def overlaps(start: float, end: float, other_start: float, other_end: float) -> bool:
    return start < other_end and other_start < end


def episodes_for(
    run: str,
    matched: list[MatchedStep],
    samples: list[MetricSample],
) -> list[ToolEpisode]:
    by_trial: dict[str, list[MatchedStep]] = {}
    for item in matched:
        by_trial.setdefault(item.step.trial_name, []).append(item)
    episodes: list[ToolEpisode] = []
    times = [sample.ts for sample in samples]
    spans = [(item.request.request_id, item.request.arrival_ts, item.request.arrival_ts + item.request.prefill_s) for item in matched]
    for trial_name, items in by_trial.items():
        ordered = sorted(items, key=lambda item: item.step.step_id)
        for current, nxt in zip(ordered, ordered[1:]):
            if current.step.bash_calls == 0:
                continue
            llm_end = current.request.arrival_ts + current.request.e2e_s
            measured = nxt.request.arrival_ts - llm_end
            if measured < 0.0:
                continue
            prefill_end = nxt.request.arrival_ts + nxt.request.prefill_s
            window = bracket(samples, times, nxt.request.arrival_ts, prefill_end)
            if window is None:
                continue
            left, right = window
            query_tokens = right.queries - left.queries
            hit_tokens = right.hits - left.hits
            if query_tokens <= 0.0:
                continue
            hit_fraction = hit_tokens / query_tokens
            others = 0
            for request_id, span_start, span_end in spans:
                if request_id == nxt.request.request_id:
                    continue
                if overlaps(left.ts, right.ts, span_start, span_end):
                    others += 1
            exclusive = others == 0
            wait_window = bracket(samples, times, llm_end, nxt.request.arrival_ts)
            preempt = 0.0
            if wait_window is not None:
                preempt = wait_window[1].preemptions - wait_window[0].preemptions
            recomputed = exclusive and hit_fraction < RECOMPUTE_HIT_FRACTION
            episodes.append(
                ToolEpisode(
                    run=run,
                    task=current.step.task,
                    difficulty=difficulty_of(current.step.task),
                    trial_name=trial_name,
                    step_id=current.step.step_id,
                    bash_calls=current.step.bash_calls,
                    requested_wait_s=current.step.requested_wait_s,
                    measured_wait_s=measured,
                    next_prompt_tokens=nxt.request.prompt_tokens,
                    next_prefill_s=nxt.request.prefill_s,
                    hit_tokens=hit_tokens,
                    query_tokens=query_tokens,
                    hit_fraction=hit_fraction,
                    exclusive=exclusive,
                    preempt_during_wait=preempt,
                    recomputed=recomputed,
                )
            )
    return episodes


def task_tool_rows(
    run: str,
    steps: list[AgentStep],
    episodes: list[ToolEpisode],
    tasks: dict[str, TaskRow],
) -> list[TaskTools]:
    by_task_steps: dict[str, list[AgentStep]] = {}
    for step in steps:
        by_task_steps.setdefault(step.task, []).append(step)
    by_task_episodes: dict[str, list[ToolEpisode]] = {}
    for episode in episodes:
        by_task_episodes.setdefault(episode.task, []).append(episode)
    rows: list[TaskTools] = []
    for task, task_steps in sorted(by_task_steps.items()):
        task_episodes = by_task_episodes.get(task, [])
        waits = [episode.measured_wait_s for episode in task_episodes]
        info = tasks.get(task)
        rows.append(
            TaskTools(
                run=run,
                task=task,
                difficulty=difficulty_of(task),
                passed=info.passed if info else False,
                exception_type=info.exception_type if info else "",
                tool_calls=sum(step.bash_calls + step.complete_calls for step in task_steps),
                bash_calls=sum(step.bash_calls for step in task_steps),
                complete_calls=sum(step.complete_calls for step in task_steps),
                episodes=len(task_episodes),
                mean_measured_wait_s=mean(waits) if waits else 0.0,
                task_seconds=info.seconds if info else 0.0,
            )
        )
    return rows


def write_rows(path: Path, rows: list[ToolEpisode] | list[TaskTools]) -> None:
    if not rows:
        path.write_text("")
        return
    payload = [asdict(row) for row in rows]
    fieldnames = list(payload[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload)


def summarize_group(label: str, episodes: list[ToolEpisode], tools: list[TaskTools]) -> list[str]:
    lines = [f"## {label}"]
    calls = [float(row.tool_calls) for row in tools]
    bash = [float(row.bash_calls) for row in tools]
    lines.append(f"tasks {len(tools)}  tool_calls_mean {mean(calls):.2f}  p50 {percentile(calls, 0.5):.1f}  max {max(calls):.0f}")
    lines.append(f"bash_calls_mean {mean(bash):.2f}  p50 {percentile(bash, 0.5):.1f}")
    for name in ("easy", "medium"):
        subset = [float(row.tool_calls) for row in tools if row.difficulty == name]
        if subset:
            lines.append(f"  {name} n {len(subset)} tool_calls_mean {mean(subset):.2f}")
    passed = [float(row.tool_calls) for row in tools if row.passed]
    failed = [float(row.tool_calls) for row in tools if not row.passed]
    if passed:
        lines.append(f"  passed n {len(passed)} tool_calls_mean {mean(passed):.2f}")
    if failed:
        lines.append(f"  failed n {len(failed)} tool_calls_mean {mean(failed):.2f}")
    context = [float(row.tool_calls) for row in tools if row.exception_type == "ContextLengthExceededError"]
    if context:
        lines.append(f"  context_errors n {len(context)} tool_calls_mean {mean(context):.2f}")
    waits = [episode.measured_wait_s for episode in episodes]
    per_call = [episode.measured_wait_s / episode.bash_calls for episode in episodes if episode.bash_calls > 0]
    requested = [episode.requested_wait_s for episode in episodes]
    lines.append(
        f"episodes {len(episodes)}  measured_episode_s mean {mean(waits):.3f} p50 {percentile(waits, 0.5):.3f} p95 {percentile(waits, 0.95):.3f}"
    )
    lines.append(
        f"measured_per_bash_call_s mean {mean(per_call):.3f} p50 {percentile(per_call, 0.5):.3f}"
    )
    lines.append(
        f"requested_wait_s mean {mean(requested):.3f} p50 {percentile(requested, 0.5):.3f}"
    )
    exclusive = [episode for episode in episodes if episode.exclusive]
    shared = len(episodes) - len(exclusive)
    lines.append(f"exclusive_prefills {len(exclusive)}  shared_prefills {shared}")
    if exclusive:
        hits = [episode.hit_fraction for episode in exclusive]
        recomputed = [episode for episode in exclusive if episode.recomputed]
        preempts = [episode for episode in exclusive if episode.preempt_during_wait > 0.0]
        lines.append(
            f"exclusive_hit_fraction mean {mean(hits):.3f} p50 {percentile(hits, 0.5):.3f}"
        )
        lines.append(
            f"recompute_episodes {len(recomputed)} / {len(exclusive)}  ({len(recomputed) / len(exclusive):.3f})"
        )
        lines.append(f"preempt_during_exclusive_wait {len(preempts)}")
        long_waits = [episode for episode in exclusive if episode.measured_wait_s >= 5.0]
        long_recompute = [episode for episode in long_waits if episode.recomputed]
        if long_waits:
            lines.append(
                f"waits_ge_5s {len(long_waits)}  recomputed {len(long_recompute)}"
            )
    return lines


def run_one(results_root: Path, label: str, relative: str) -> tuple[list[AgentStep], list[MatchedStep], list[ToolEpisode], list[TaskTools]]:
    root = results_root / relative
    steps = load_steps(root / "harbor")
    requests = load_requests(root / "requests.csv")
    tasks = load_tasks(root / "tasks.csv")
    samples = load_samples(root / "server_metrics.jsonl")
    matched = match_steps(steps, requests)
    episode_rows = episodes_for(label, matched, samples)
    tool_rows = task_tool_rows(label, steps, episode_rows, tasks)
    return steps, matched, episode_rows, tool_rows


def analyze(results_root: Path, out_dir: Path) -> str:
    all_episodes: list[ToolEpisode] = []
    all_tools: list[TaskTools] = []
    lines: list[str] = []
    for label, relative in RUNS:
        steps, matched, episode_rows, tool_rows = run_one(results_root, label, relative)
        all_episodes.extend(episode_rows)
        all_tools.extend(tool_rows)
        lines.append(
            f"{label} agent_steps {len(steps)} matched {len(matched)} match_rate {len(matched) / len(steps):.3f}"
        )
        lines.extend(summarize_group(label, episode_rows, tool_rows))
        lines.append("")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "tool_episodes.csv", all_episodes)
    write_rows(out_dir / "task_tools.csv", all_tools)
    text = "\n".join(lines) + "\n"
    (out_dir / "summary.txt").write_text(text)
    return text


def main(results_root: Path, out_dir: Path) -> None:
    sys.stdout.write(analyze(results_root, out_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    main(Path(args.results_root), Path(args.out_dir))
