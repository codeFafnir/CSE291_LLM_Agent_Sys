"""Turn one finished SPEED-Bench run dir into the metric CSVs and append one row to speed_all_runs.csv.

  report.py --run-dir <dir> --all-runs-csv <RESULTS_ROOT>/speed_all_runs.csv --concurrency 1 \
    --repo-sha <sha> --utc-start <iso>

Reads responses.jsonl, server_metrics.jsonl, gpu_dmon.log and serve_config.env from the run dir.
Writes run_metrics.csv (one wide row), tier_metrics.csv, category_metrics.csv and timeseries.csv.
"""

import argparse
import csv
from pathlib import Path

from models import RequestRow, ResponseRecord, Sample, SpecDelta
from server_metrics import (
    GENERATION_TOKENS,
    KV_USAGE,
    PREEMPTIONS,
    PREFIX_HITS,
    PREFIX_QUERIES,
    PROMPT_TOKENS,
    RUNNING,
    SPEC_POSITIONS,
    WAITING,
    accept_length,
    accept_per_position,
    accept_rate,
    delta,
    load_samples,
    spec_delta,
)
from stats import percentile, present, stats

Cell = str | int | float | bool | None
Row = dict[str, Cell]


def load_responses(path: Path) -> list[ResponseRecord]:
    return [ResponseRecord.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def read_env_file(path: Path) -> dict[str, str]:
    pairs = (line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
    return {key: value for key, value in pairs}


def dmon_columns(path: Path, names: tuple[str, ...]) -> dict[str, list[float]]:
    # nvidia-smi dmon: the first '# gpu ...' line names the columns; '-' means no value.
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


def append_csv(path: Path, row: Row) -> None:
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


def spec_columns(spec: SpecDelta | None) -> Row:
    per_position = accept_per_position(spec) if spec is not None else [None] * SPEC_POSITIONS
    return {
        "spec_mean_accept_len": accept_length(spec) if spec is not None else None,
        "spec_accept_rate": accept_rate(spec) if spec is not None else None,
        **{f"spec_accept_pos{i + 1}": per_position[i] for i in range(SPEC_POSITIONS)},
    }


def accepted_per_position(row: RequestRow) -> tuple[float, ...]:
    # Per-position acceptance is stored as a fraction of drafts; turn it back into a count.
    fractions = (row.spec_accept_pos1, row.spec_accept_pos2, row.spec_accept_pos3)
    return tuple((f or 0.0) * (row.spec_drafts or 0.0) for f in fractions)


def group_spec(records: list[RequestRow]) -> SpecDelta | None:
    # Per-request spec counters exist only at concurrency 1.
    with_spec = [r for r in records if r.spec_drafts is not None]
    if not with_spec:
        return None
    return SpecDelta(
        drafts=sum(r.spec_drafts or 0.0 for r in with_spec),
        draft_tokens=sum(r.spec_draft_tokens or 0.0 for r in with_spec),
        accepted=sum(r.spec_accepted or 0.0 for r in with_spec),
        accepted_per_position=[sum(accepted_per_position(r)[i] for r in with_spec) for i in range(SPEC_POSITIONS)],
    )


def latency_columns(records: list[ResponseRecord]) -> Row:
    ok = [r.row for r in records if r.row.error is None]
    gaps = [gap for r in records if r.row.error is None for gap in r.chunk_gaps_s]
    return {
        **stats("ttft_s", present([r.ttft_s for r in ok])),
        **stats("tpot_s", present([r.tpot_s for r in ok])),
        **stats("itl_s", gaps),
        **stats("e2e_s", [r.e2e_s for r in ok]),
        **stats("queue_s", present([r.queue_s for r in ok])),
        **stats("prefill_s", present([r.prefill_s for r in ok])),
        **stats("decode_s", present([r.decode_s for r in ok])),
    }


def group_row(key: str, name: str, records: list[ResponseRecord]) -> Row:
    rows = [r.row for r in records]
    spec = group_spec(rows)
    return {
        key: name,
        "n_requests": len(rows),
        "n_prompts": len({r.question_id for r in rows}),
        "errors": sum(1 for r in rows if r.error is not None),
        "prompt_tokens_total": sum(r.prompt_tokens for r in rows),
        "output_tokens_total": sum(r.completion_tokens for r in rows),
        **latency_columns(records),
        **spec_columns(spec),
    }


def grouped_rows(records: list[ResponseRecord], key: str) -> list[Row]:
    names = sorted({getattr(r.row, key) for r in records})
    return [group_row(key, name, [r for r in records if getattr(r.row, key) == name]) for name in names]


def timeseries_rows(samples: list[Sample]) -> list[Row]:
    return [
        {
            "ts": s.ts,
            "gpu_used_mib": s.gpu_used_mib,
            "running": s.series.get(RUNNING),
            "waiting": s.series.get(WAITING),
            "kv_usage": s.series.get(KV_USAGE),
            "prompt_tokens_total": s.series.get(PROMPT_TOKENS),
            "generation_tokens_total": s.series.get(GENERATION_TOKENS),
            "preemptions_total": s.series.get(PREEMPTIONS),
            "prefix_queries_total": s.series.get(PREFIX_QUERIES),
            "prefix_hits_total": s.series.get(PREFIX_HITS),
        }
        for s in samples
    ]


def run_row(
    identity: Row, config: dict[str, str], records: list[ResponseRecord], samples: list[Sample], dmon: dict[str, list[float]]
) -> Row:
    rows = [r.row for r in records]
    first, last = samples[0], samples[-1]
    wall = max(r.start_ts + r.e2e_s for r in rows) - min(r.start_ts for r in rows)
    prompt_total = sum(r.prompt_tokens for r in rows)
    output_total = sum(r.completion_tokens for r in rows)
    prompts = sorted(float(r.prompt_tokens) for r in rows)
    completions = sorted(float(r.completion_tokens) for r in rows)
    kv = present([s.series.get(KV_USAGE) for s in samples])
    waiting = present([s.series.get(WAITING) for s in samples])
    running = present([s.series.get(RUNNING) for s in samples])
    queries = delta(first.series, last.series, PREFIX_QUERIES)
    return {
        **identity,
        **{key.lower(): value for key, value in config.items() if key != "RUN_ID"},
        "prompts": len({r.question_id for r in rows}),
        "requests": len(rows),
        "errors": sum(1 for r in rows if r.error is not None),
        "over_context_prompts": len({r.question_id for r in rows if r.over_context}),
        "finish_length": sum(1 for r in rows if r.finish_reason == "length"),
        "wall_s": wall,
        "requests_per_s": len(rows) / wall,
        "prompt_tokens_total": prompt_total,
        "output_tokens_total": output_total,
        "input_tok_per_s": prompt_total / wall,
        "output_tok_per_s": output_total / wall,
        **latency_columns(records),
        "prompt_tokens_p50": percentile(prompts, 0.50),
        "prompt_tokens_p95": percentile(prompts, 0.95),
        "prompt_tokens_p99": percentile(prompts, 0.99),
        "prompt_tokens_max": prompts[-1],
        "completion_tokens_p50": percentile(completions, 0.50),
        "completion_tokens_p95": percentile(completions, 0.95),
        "completion_tokens_p99": percentile(completions, 0.99),
        "completion_tokens_max": completions[-1],
        "peak_gpu_mib_client": max(r.peak_gpu_mib for r in rows),
        "peak_gpu_mib_dmon": max(dmon["fb"]) if dmon["fb"] else None,
        "peak_gpu_mib_scrape": max(s.gpu_used_mib for s in samples),
        "mean_gpu_util_pct": sum(dmon["sm"]) / len(dmon["sm"]) if dmon["sm"] else None,
        "peak_kv_usage": max(kv) if kv else None,
        "mean_kv_usage": sum(kv) / len(kv) if kv else None,
        "max_running": max(running) if running else None,
        "max_waiting": max(waiting) if waiting else None,
        "mean_waiting": sum(waiting) / len(waiting) if waiting else None,
        "preemptions": delta(first.series, last.series, PREEMPTIONS),
        "prefix_cache_hit_rate": delta(first.series, last.series, PREFIX_HITS) / queries if queries > 0.0 else None,
        **spec_columns(spec_delta(first.series, last.series)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--all-runs-csv", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--repo-sha", required=True)
    parser.add_argument("--utc-start", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir: Path = args.run_dir
    records = load_responses(run_dir / "responses.jsonl")
    samples = load_samples(run_dir / "server_metrics.jsonl")
    if not records or not samples:
        raise ValueError(f"{run_dir}: {len(records)} responses and {len(samples)} metric samples; nothing to report")
    config = read_env_file(run_dir / "serve_config.env")
    dmon = dmon_columns(run_dir / "gpu_dmon.log", ("sm", "fb"))
    identity: Row = {
        "run_id": run_dir.name,
        "utc_start": args.utc_start,
        "engine": "vllm",
        "repo_sha": args.repo_sha,
        "concurrency": args.concurrency,
    }
    row = run_row(identity, config, records, samples, dmon)
    write_csv(run_dir / "run_metrics.csv", [row], list(row))
    tier_rows = grouped_rows(records, "tier")
    write_csv(run_dir / "tier_metrics.csv", tier_rows, list(tier_rows[0]))
    category_rows = grouped_rows(records, "category")
    write_csv(run_dir / "category_metrics.csv", category_rows, list(category_rows[0]))
    ts_rows = timeseries_rows(samples)
    write_csv(run_dir / "timeseries.csv", ts_rows, list(ts_rows[0]))
    append_csv(args.all_runs_csv, row)
    print(f"wrote run_metrics.csv, tier_metrics.csv, category_metrics.csv, timeseries.csv; appended {args.all_runs_csv}")


if __name__ == "__main__":
    main()
