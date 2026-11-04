"""Stack the nemotron-30 c1 and c4 metrics into team CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

RUNS = (
    ("c1", "c1/baseline-vllm-nemotron-30-c1-b8-t8192-l20000-lru-20261104T051307Z"),
    ("c4", "c4/baseline-vllm-nemotron-30-c4-b8-t8192-l20000-lru-20261104T072031Z"),
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def with_run(run: str, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{"run": run, **row} for row in rows]


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean of empty list")
    return sum(values) / len(values)


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def num(text: str) -> float:
    return float(text)


def analysis_rows(tools: list[dict[str, str]], episodes: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run in ("c1", "c4"):
        task_rows = [row for row in tools if row["run"] == run]
        episode_rows = [row for row in episodes if row["run"] == run]
        calls = [num(row["tool_calls"]) for row in task_rows]
        bash = [num(row["bash_calls"]) for row in task_rows]
        waits = [num(row["measured_wait_s"]) for row in episode_rows]
        per_call = [num(row["measured_wait_s"]) / num(row["bash_calls"]) for row in episode_rows]
        requested = [num(row["requested_wait_s"]) for row in episode_rows]
        exclusive = [row for row in episode_rows if row["exclusive"] == "True"]
        hits = [num(row["hit_fraction"]) for row in exclusive]
        recomputed = [row for row in exclusive if row["recomputed"] == "True"]
        early = [row for row in recomputed if int(row["step_id"]) <= 4]
        late = [row for row in recomputed if int(row["step_id"]) > 4]
        task_seconds = sum(num(row["task_seconds"]) for row in task_rows)
        rows.append(
            {
                "run": run,
                "tasks_with_trajectory": str(len(task_rows)),
                "tool_calls_mean": f"{mean(calls):.4f}",
                "tool_calls_p50": f"{percentile(calls, 0.5):.4f}",
                "tool_calls_max": f"{max(calls):.0f}",
                "bash_calls_mean": f"{mean(bash):.4f}",
                "bash_calls_p50": f"{percentile(bash, 0.5):.4f}",
                "easy_tool_calls_mean": f"{mean([num(row['tool_calls']) for row in task_rows if row['difficulty'] == 'easy']):.4f}",
                "medium_tool_calls_mean": f"{mean([num(row['tool_calls']) for row in task_rows if row['difficulty'] == 'medium']):.4f}",
                "passed_tool_calls_mean": f"{mean([num(row['tool_calls']) for row in task_rows if row['passed'] == 'True']):.4f}",
                "context_error_tool_calls_mean": f"{mean([num(row['tool_calls']) for row in task_rows if row['exception_type'] == 'ContextLengthExceededError']):.4f}",
                "tool_episodes": str(len(episode_rows)),
                "measured_episode_s_mean": f"{mean(waits):.4f}",
                "measured_episode_s_p50": f"{percentile(waits, 0.5):.4f}",
                "measured_episode_s_p95": f"{percentile(waits, 0.95):.4f}",
                "measured_per_bash_call_s_mean": f"{mean(per_call):.4f}",
                "measured_per_bash_call_s_p50": f"{percentile(per_call, 0.5):.4f}",
                "requested_wait_s_mean": f"{mean(requested):.4f}",
                "requested_wait_s_p50": f"{percentile(requested, 0.5):.4f}",
                "tool_wait_s_sum": f"{sum(waits):.4f}",
                "task_seconds_sum": f"{task_seconds:.4f}",
                "tool_wait_fraction_of_task_time": f"{sum(waits) / task_seconds:.4f}",
                "exclusive_prefills": str(len(exclusive)),
                "shared_prefills": str(len(episode_rows) - len(exclusive)),
                "exclusive_hit_fraction_mean": f"{mean(hits):.4f}",
                "exclusive_hit_fraction_p50": f"{percentile(hits, 0.5):.4f}",
                "recompute_episodes": str(len(recomputed)),
                "recompute_fraction_of_exclusive": f"{len(recomputed) / len(exclusive):.4f}",
                "recompute_steps_1_to_4": str(len(early)),
                "recompute_steps_after_4": str(len(late)),
            }
        )
    return rows


def join_tasks(results_root: Path, tools: list[dict[str, str]]) -> list[dict[str, str]]:
    tool_by_key = {(row["run"], row["task"]): row for row in tools}
    joined: list[dict[str, str]] = []
    for run, relative in RUNS:
        for row in read_rows(results_root / relative / "tasks.csv"):
            extra = tool_by_key.get((run, row["task"]), {})
            joined.append(
                {
                    "run": run,
                    "task": row["task"],
                    "difficulty": extra.get("difficulty", ""),
                    "passed": row["passed"],
                    "reward": row["reward"],
                    "exception_type": row["exception_type"],
                    "over_context": row["over_context"],
                    "seconds": row["seconds"],
                    "peak_tokens": row["peak_tokens"],
                    "n_requests": row["n_requests"],
                    "tool_calls": extra.get("tool_calls", ""),
                    "bash_calls": extra.get("bash_calls", ""),
                    "complete_calls": extra.get("complete_calls", ""),
                    "tool_episodes": extra.get("episodes", ""),
                    "mean_measured_wait_s": extra.get("mean_measured_wait_s", ""),
                }
            )
    return joined


def stack_named(results_root: Path, name: str) -> tuple[list[str], list[dict[str, str]]]:
    stacked: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for run, relative in RUNS:
        rows = read_rows(results_root / relative / name)
        if not fieldnames:
            fieldnames = ["run", *rows[0].keys()]
        stacked.extend(with_run(run, rows))
    return fieldnames, stacked


def profile_rows(results_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run, relative in RUNS:
        root = results_root / relative
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if "torch_profile" not in path.parts and path.name != "timeline.perfetto.json":
                continue
            kind = "perfetto_timeline"
            if path.suffix == ".gz" and "rank0" in path.name:
                kind = "tracelens_rank0_kernel_trace"
            elif path.suffix == ".gz":
                kind = "tracelens_async_llm_trace"
            elif path.name == "profile_prompt.json":
                kind = "profiled_prompt"
            elif path.name.endswith("profiler_out.txt"):
                kind = "profiler_summary_text"
            rows.append(
                {
                    "run": run,
                    "kind": kind,
                    "pass": path.parent.name,
                    "bytes": str(path.stat().st_size),
                    "path": str(path),
                }
            )
    return rows


def manifest_rows(out_dir: Path) -> list[dict[str, str]]:
    notes = {
        "run_metrics.csv": "One row per run. Server config, latency percentiles, throughput, KV, speculative-decoding acceptance.",
        "analysis_summary.csv": "One row per run. Tool-call counts, measured tool waits, prefix-cache hit fraction, recompute counts.",
        "tasks.csv": "One row per task per run. Outcome plus tool-call counts. Empty tool columns mean the agent never wrote a trajectory.",
        "requests.csv": "One row per LLM request. Queue, prefill, TTFT, decode, TPOT, end-to-end. cached_tokens is empty for these vLLM runs.",
        "timeseries.csv": "One row per second. Cumulative prefix hits and preemptions, KV usage, GPU memory.",
        "tool_episodes.csv": "One row per bash episode. Measured wait is next-request arrival minus this request's end. recomputed means the next exclusive prefill hit under half its prompt tokens.",
        "profile_files.csv": "Local profiler files. TraceLens reads the rank0 pt.trace.json.gz files.",
    }
    rows: list[dict[str, str]] = []
    for name, note in notes.items():
        path = out_dir / name
        with path.open() as handle:
            count = sum(1 for _ in handle) - 1
        rows.append({"file": name, "rows": str(count), "contents": note})
    return rows


def export(results_root: Path, analysis_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tools = read_rows(analysis_dir / "task_tools.csv")
    episodes = read_rows(analysis_dir / "tool_episodes.csv")
    run_fields, run_rows = stack_named(results_root, "run_metrics.csv")
    request_fields, request_rows = stack_named(results_root, "requests.csv")
    series_fields, series_rows = stack_named(results_root, "timeseries.csv")
    summary = analysis_rows(tools, episodes)
    tasks = join_tasks(results_root, tools)
    profiles = profile_rows(results_root)
    write_rows(out_dir / "run_metrics.csv", run_rows, run_fields)
    write_rows(out_dir / "analysis_summary.csv", summary, list(summary[0].keys()))
    write_rows(out_dir / "tasks.csv", tasks, list(tasks[0].keys()))
    write_rows(out_dir / "requests.csv", request_rows, request_fields)
    write_rows(out_dir / "timeseries.csv", series_rows, series_fields)
    write_rows(out_dir / "tool_episodes.csv", episodes, list(episodes[0].keys()))
    write_rows(out_dir / "profile_files.csv", profiles, list(profiles[0].keys()))
    write_rows(out_dir / "manifest.csv", manifest_rows(out_dir), ["file", "rows", "contents"])


def main(results_root: Path, analysis_dir: Path, out_dir: Path) -> None:
    export(results_root, analysis_dir, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    main(Path(args.results_root), Path(args.analysis_dir), Path(args.out_dir))
