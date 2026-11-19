"""One comparison row per metric for the LRU baseline and Continuum runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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

METRICS = (
    "trials",
    "solved",
    "mean_reward",
    "context_length_errors",
    "wall_s",
    "output_tok_per_s",
    "prefix_cache_hit_rate",
    "peak_kv_usage",
    "mean_kv_usage",
    "preemptions",
    "spec_accept_rate",
    "ttft_s_p50",
    "ttft_s_p95",
    "queue_s_p95",
    "peak_gpu_mib",
)


def read_metrics(run_dir: Path) -> dict[str, str]:
    path = run_dir / "run_metrics.csv"
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"{path} has {len(rows)} rows")
    return rows[0]


def context_split(run_dir: Path) -> dict[str, str]:
    easy = 0
    medium = 0
    with (run_dir / "tasks.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["exception_type"] != "ContextLengthExceededError":
                continue
            if row["task"] in MEDIUM_TASKS:
                medium += 1
            else:
                easy += 1
    return {"easy_context_errors": str(easy), "medium_context_errors": str(medium)}


def pin_count(path: Path) -> str:
    if not path.exists():
        return ""
    payload = json.loads(path.read_text())
    count = 0
    for events in payload.values():
        for event in events:
            if "pinned_time" in event:
                count += 1
    return str(count)


def main(
    baseline_c1: Path,
    baseline_c4: Path,
    continuum_c1: Path,
    continuum_c4: Path,
    timestamps_c1: Path,
    timestamps_c4: Path,
    out: Path,
) -> None:
    runs = {
        "baseline_c1": baseline_c1,
        "baseline_c4": baseline_c4,
        "continuum_c1": continuum_c1,
        "continuum_c4": continuum_c4,
    }
    loaded = {name: read_metrics(run_dir) for name, run_dir in runs.items()}
    splits = {name: context_split(run_dir) for name, run_dir in runs.items()}
    rows: list[dict[str, str]] = []
    for metric in METRICS:
        rows.append({"metric": metric, **{name: loaded[name].get(metric, "") for name in runs}})
    for metric in ("easy_context_errors", "medium_context_errors"):
        rows.append({"metric": metric, **{name: splits[name][metric] for name in runs}})
    rows.append(
        {
            "metric": "pinned_events",
            "baseline_c1": "",
            "baseline_c4": "",
            "continuum_c1": pin_count(timestamps_c1),
            "continuum_c4": pin_count(timestamps_c4),
        }
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["metric", "baseline_c1", "baseline_c4", "continuum_c1", "continuum_c4"],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-c1", type=Path, required=True)
    parser.add_argument("--baseline-c4", type=Path, required=True)
    parser.add_argument("--continuum-c1", type=Path, required=True)
    parser.add_argument("--continuum-c4", type=Path, required=True)
    parser.add_argument("--timestamps-c1", type=Path, required=True)
    parser.add_argument("--timestamps-c4", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(
        args.baseline_c1,
        args.baseline_c4,
        args.continuum_c1,
        args.continuum_c4,
        args.timestamps_c1,
        args.timestamps_c4,
        args.out,
    )
