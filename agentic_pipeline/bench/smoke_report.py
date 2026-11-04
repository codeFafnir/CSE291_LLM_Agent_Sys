"""Per-task pass/fail, peak tokens, wall time and over-context flag from a Harbor job dir.

  smoke_report.py --harbor-dir <run_dir>/harbor --max-context 20000 --out <run_dir>/smoke_report.json
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from summarize_sla import CONTEXT_ERROR_MARKERS


class _Verifier(BaseModel):
    rewards: dict[str, float] | None = None


class _ExceptionInfo(BaseModel):
    exception_type: str | None = None
    exception_message: str | None = None
    exception_traceback: str | None = None


class _TrialResult(BaseModel):
    task_name: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    verifier_result: _Verifier | None = None
    exception_info: _ExceptionInfo | None = None


class _StepMetrics(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class _Step(BaseModel):
    metrics: _StepMetrics | None = None


class _Trajectory(BaseModel):
    steps: list[_Step]


class TaskReport(BaseModel):
    task: str
    passed: bool
    reward: float | None
    peak_tokens: int | None
    seconds: float | None
    over_context: bool
    exception: str | None
    exception_type: str | None
    started_ts: float | None
    finished_ts: float | None


def epoch(moment: datetime | None) -> float | None:
    if moment is None:
        return None
    return (moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)).timestamp()


def peak_tokens(trajectory_path: Path) -> int | None:
    if not trajectory_path.exists():
        return None
    steps = _Trajectory.model_validate_json(trajectory_path.read_text()).steps
    totals = [
        (s.metrics.prompt_tokens or 0) + (s.metrics.completion_tokens or 0)
        for s in steps
        if s.metrics is not None
    ]
    return max(totals) if totals else None


def report_trial(trial_dir: Path, max_context: int) -> TaskReport:
    result = _TrialResult.model_validate_json((trial_dir / "result.json").read_text())
    reward = result.verifier_result.rewards.get("reward") if result.verifier_result and result.verifier_result.rewards else None
    peak = peak_tokens(trial_dir / "agent" / "trajectory.json")
    exception = None if result.exception_info is None else str(result.exception_info)
    context_error = exception is not None and any(m in exception for m in CONTEXT_ERROR_MARKERS)
    seconds = (result.finished_at - result.started_at).total_seconds() if result.started_at and result.finished_at else None
    return TaskReport(
        task=result.task_name,
        passed=reward is not None and reward >= 1.0,
        reward=reward,
        peak_tokens=peak,
        seconds=seconds,
        over_context=context_error or (peak is not None and peak > max_context),
        exception=exception,
        exception_type=result.exception_info.exception_type if result.exception_info else None,
        started_ts=epoch(result.started_at),
        finished_ts=epoch(result.finished_at),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harbor-dir", type=Path, required=True)
    parser.add_argument("--max-context", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = [report_trial(p.parent, args.max_context) for p in sorted(args.harbor_dir.glob("*/result.json"))]
    args.out.write_text("[" + ",".join(r.model_dump_json() for r in reports) + "]\n")
    print(f"{'task':40} {'pass':5} {'peak_tok':>9} {'sec':>8} over_ctx")
    for r in reports:
        flag = "OVER_CONTEXT" if r.over_context else "-"
        print(f"{r.task:40} {str(r.passed):5} {str(r.peak_tokens):>9} {str(round(r.seconds) if r.seconds is not None else None):>8} {flag}")
        if r.exception is not None:
            print(f"  exception: {r.exception[:300]}")


if __name__ == "__main__":
    main()
