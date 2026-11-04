"""Write every per-run CSV from a finished run dir and append one row to the master all_runs.csv.

  run_csv.py --engine vllm --run-dir <dir> --window-start <epoch> --window-end <epoch> \
    --raw-source /data/cse291-results/otel/spans.jsonl --all-runs-csv /data/cse291-results/all_runs.csv \
    --max-context 20000 --job nemotron-smoke --concurrency 1 --engine-version <v> --repo-sha <sha> \
    --utc-start <iso> --profile-dir <dir>

Reads summary.json, server_metrics.jsonl, gpu_dmon.log and harbor/. Writes requests.csv, tasks.csv,
timeseries.csv, run_metrics.csv and traces/{requests_raw.jsonl,timeline.perfetto.json}.
vLLM --raw-source is the OTLP spans.jsonl; SGLang --raw-source is the sglang-request-metrics dir.
"""

import argparse
import csv
import json
import math
from pathlib import Path

from request_records import (
    GAUGES,
    ParsedRequest,
    RequestRecord,
    TrialWindow,
    attach_peaks,
    attach_trials,
    parse_sglang_records,
    parse_vllm_spans,
    perfetto_trace,
    read_lines,
)
from smoke_report import TaskReport, report_trial
from summarize_sla import Engine, Sample, Summary, load_samples

Cell = str | int | float | None
Row = dict[str, Cell]
STAT_NAMES: tuple[str, ...] = ("mean", "p50", "p90", "p95", "p99", "max")
SPEC_POSITIONS = 3


def percentile(sorted_values: list[float], q: float) -> float:
    rank = (len(sorted_values) - 1) * q
    low, high = math.floor(rank), math.ceil(rank)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (rank - low)


def stats(prefix: str, values: list[float]) -> Row:
    if not values:
        return {f"{prefix}_{name}": None for name in STAT_NAMES}
    ordered = sorted(values)
    return {
        f"{prefix}_mean": sum(ordered) / len(ordered),
        f"{prefix}_p50": percentile(ordered, 0.50),
        f"{prefix}_p90": percentile(ordered, 0.90),
        f"{prefix}_p95": percentile(ordered, 0.95),
        f"{prefix}_p99": percentile(ordered, 0.99),
        f"{prefix}_max": ordered[-1],
    }


def present(values: list[float | None]) -> list[float]:
    return [v for v in values if v is not None]


def dmon_columns(path: Path, names: tuple[str, ...]) -> dict[str, list[float]]:
    # nvidia-smi dmon: first '# gpu ...' line names the columns; '-' means no value.
    header: list[str] = []
    columns: dict[str, list[float]] = {name: [] for name in names}
    for line in path.read_text().splitlines():
        if line.startswith("# gpu"):
            header = line.lstrip("# ").split()
        elif not line.startswith("#") and header:
            cells = line.split()
            for name in names:
                cell = cells[header.index(name)]
                if cell != "-":
                    columns[name].append(float(cell))
    return columns


def write_csv(path: Path, rows: list[Row], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_all_runs(path: Path, row: Row) -> None:
    # A changed column set rewrites the master file under the union of old and new columns.
    existing: list[dict[str, str]] = []
    header: list[str] = []
    if path.exists():
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            existing = list(reader)
    merged = header + [name for name in row if name not in header]
    if merged == header:
        with path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=merged).writerow(row)
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged, restval="")
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow(row)


def requests_rows(records: list[RequestRecord]) -> list[Row]:
    return [r.model_dump() for r in records]


def tasks_rows(reports: list[TaskReport], records: list[RequestRecord]) -> list[Row]:
    return [
        {
            "task": r.task,
            "passed": r.passed,
            "reward": r.reward,
            "peak_tokens": r.peak_tokens,
            "seconds": r.seconds,
            "over_context": r.over_context,
            "exception_type": r.exception_type,
            "n_requests": sum(1 for rec in records if rec.trial == r.task),
        }
        for r in reports
    ]


def timeseries_rows(samples: list[Sample], engine: Engine) -> list[Row]:
    g = GAUGES[engine]
    return [
        {
            "ts": s.ts,
            "gpu_used_mib": s.gpu_used_mib,
            "running": s.series.get(g.running),
            "waiting": s.series.get(g.waiting),
            "kv_usage": s.series.get(g.kv_usage),
            "prompt_tokens_total": s.series.get(g.prompt_tokens_total),
            "generation_tokens_total": s.series.get(g.generation_tokens_total),
            "preemptions_total": s.series.get(g.preemptions_total),
            "prefix_hits_total": s.series.get(g.prefix_hits_total) if g.prefix_hits_total else None,
        }
        for s in samples
    ]


def spec_columns(accepted_per_position: list[float]) -> Row:
    return {
        f"spec_accept_pos{i + 1}": accepted_per_position[i] if i < len(accepted_per_position) else None
        for i in range(SPEC_POSITIONS)
    }


def run_row(
    summary: Summary,
    records: list[RequestRecord],
    reports: list[TaskReport],
    samples: list[Sample],
    window: tuple[float, float],
    identity: Row,
    dmon_dir: Path,
    profile_dir: str,
    timeline_path: str,
) -> Row:
    g = GAUGES[summary.engine]
    config = summary.config
    wall = window[1] - window[0]
    prompt_total = sum(r.prompt_tokens for r in records)
    output_total = sum(r.completion_tokens for r in records)
    inside = [s for s in samples if window[0] <= s.ts <= window[1]]
    kv = [s.series.get(g.kv_usage, 0.0) for s in inside]
    waiting = [s.series.get(g.waiting, 0.0) for s in inside]
    running = [s.series.get(g.running, 0.0) for s in inside]
    dmon = dmon_columns(dmon_dir / "gpu_dmon.log", ("sm", "fb"))
    score = summary.task_score
    seconds = present([r.seconds for r in reports])
    prompts = sorted(float(r.prompt_tokens) for r in records)
    completions = sorted(float(r.completion_tokens) for r in records)
    row: Row = {
        **identity,
        "model": config["target"],
        "speculator": config["speculator"],
        "max_model_len": config["MAX_MODEL_LEN"],
        "max_num_seqs": config["MAX_NUM_SEQS"],
        "max_batched_tokens": config["MAX_BATCHED_TOKENS"],
        "eviction_policy": config["EVICTION_POLICY"],
        "preemption_policy": config["PREEMPTION_POLICY"],
        "kv_dtype": config["kv_dtype"],
        "trials": score.trials if score else None,
        "solved": score.solved if score else None,
        "mean_reward": score.mean_reward if score else None,
        "context_length_errors": score.context_length_errors if score else None,
        "task_seconds_p50": percentile(sorted(seconds), 0.5) if seconds else None,
        "task_seconds_max": max(seconds) if seconds else None,
        "wall_s": wall,
        "requests": len(records),
        "requests_per_s": len(records) / wall,
        "prompt_tokens_total": prompt_total,
        "output_tokens_total": output_total,
        "input_tok_per_s": prompt_total / wall,
        "output_tok_per_s": output_total / wall,
        **stats("ttft_s", present([r.ttft_s for r in records])),
        **stats("tpot_s", present([r.tpot_s for r in records])),
        **stats("e2e_s", [r.e2e_s for r in records]),
        **stats("queue_s", present([r.queue_s for r in records])),
        **stats("prefill_s", present([r.prefill_s for r in records])),
        **stats("decode_s", present([r.decode_s for r in records])),
        "prompt_tokens_p50": percentile(prompts, 0.50) if prompts else None,
        "prompt_tokens_p95": percentile(prompts, 0.95) if prompts else None,
        "prompt_tokens_p99": percentile(prompts, 0.99) if prompts else None,
        "prompt_tokens_max": prompts[-1] if prompts else None,
        "completion_tokens_p50": percentile(completions, 0.50) if completions else None,
        "completion_tokens_p99": percentile(completions, 0.99) if completions else None,
        "completion_tokens_max": completions[-1] if completions else None,
        "peak_gpu_mib": max([summary.peak_gpu_mib, *dmon["fb"]]),
        "mean_gpu_util_pct": sum(dmon["sm"]) / len(dmon["sm"]) if dmon["sm"] else None,
        "peak_kv_usage": max(kv) if kv else None,
        "mean_kv_usage": sum(kv) / len(kv) if kv else None,
        "max_running": max(running) if running else None,
        "max_waiting": max(waiting) if waiting else None,
        "mean_waiting": sum(waiting) / len(waiting) if waiting else None,
        "preemptions": summary.preemptions,
        "prefix_cache_hit_rate": summary.prefix_cache_hit_rate,
        "spec_mean_accept_len": summary.spec.mean_acceptance_length,
        "spec_accept_rate": summary.spec.draft_acceptance_rate,
        **spec_columns(summary.spec.accepted_per_position),
        "profile_dir": profile_dir,
        "timeline_path": timeline_path,
    }
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("vllm", "sglang"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--window-start", type=float, required=True)
    parser.add_argument("--window-end", type=float, required=True)
    parser.add_argument("--raw-source", type=Path, required=True)
    parser.add_argument("--all-runs-csv", type=Path, required=True)
    parser.add_argument("--max-context", type=int, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--engine-version", required=True)
    parser.add_argument("--repo-sha", required=True)
    parser.add_argument("--utc-start", required=True)
    parser.add_argument("--profile-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine: Engine = args.engine
    run_dir: Path = args.run_dir
    window = (args.window_start, args.window_end)
    summary = Summary.model_validate_json((run_dir / "summary.json").read_text())
    samples = load_samples(run_dir / "server_metrics.jsonl")

    lines = read_lines(args.raw_source)
    parsed: list[ParsedRequest] = (
        parse_vllm_spans(lines, window) if engine == "vllm" else parse_sglang_records(lines, window)
    )
    traces = run_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    (traces / "requests_raw.jsonl").write_text("".join(p.raw + "\n" for p in parsed))

    reports = [report_trial(p.parent, args.max_context) for p in sorted((run_dir / "harbor").glob("*/result.json"))]
    trials = [
        TrialWindow(task=r.task, start_ts=r.started_ts, end_ts=r.finished_ts)
        for r in reports
        if r.started_ts is not None and r.finished_ts is not None
    ]
    records = sorted(
        attach_peaks(attach_trials([p.record for p in parsed], trials), samples, GAUGES[engine]),
        key=lambda r: r.arrival_ts,
    )

    request_fields = list(RequestRecord.model_fields)
    write_csv(run_dir / "requests.csv", requests_rows(records), request_fields)
    task_rows = tasks_rows(reports, records)
    write_csv(
        run_dir / "tasks.csv",
        task_rows,
        ["task", "passed", "reward", "peak_tokens", "seconds", "over_context", "exception_type", "n_requests"],
    )
    ts_rows = timeseries_rows(samples, engine)
    write_csv(run_dir / "timeseries.csv", ts_rows, list(ts_rows[0]))
    timeline = traces / "timeline.perfetto.json"
    timeline.write_text(json.dumps(perfetto_trace(records, samples, GAUGES[engine])))

    identity: Row = {
        "run_id": run_dir.name,
        "utc_start": args.utc_start,
        "engine": engine,
        "engine_version": args.engine_version,
        "repo_sha": args.repo_sha,
        "job": args.job,
        "concurrency": args.concurrency,
    }
    row = run_row(summary, records, reports, samples, window, identity, run_dir, args.profile_dir, str(timeline))
    write_csv(run_dir / "run_metrics.csv", [row], list(row))
    append_all_runs(args.all_runs_csv, row)
    print(f"wrote {len(records)} requests, {len(reports)} tasks, run_metrics.csv; appended {args.all_runs_csv}")


if __name__ == "__main__":
    main()
