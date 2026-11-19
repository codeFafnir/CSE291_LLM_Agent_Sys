"""Check one run_swebench.sh cell against the smoke criteria; exit 1 listing every failed one.

Run in the vllm env: PYTHONPATH=swesmith python swebench/check_smoke.py --run-dir <run dir>
"""

import argparse
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from run import read_jsonl
from session_proxy import RequestLog
from tool_span import ToolSpan

SEARCH_OR_EDIT_TOOLS = {"search_dir", "search_file", "find_file", "str_replace_editor:view", "str_replace_editor:str_replace"}
NON_BASH_TOOLS = {"search_dir", "search_file", "find_file", "submit", "none"}
EDITOR_PREFIX = "str_replace_editor"


class SessionStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    instance_id: str
    status: str


class CellDone(BaseModel):
    model_config = ConfigDict(frozen=True)

    serve_alive: bool
    step_exit_status: dict[str, int]


def is_bash(tool: str) -> bool:
    """tool_name() labels a step by its first token, so a bash step is anything that is not a registered command."""
    return tool not in NON_BASH_TOOLS and not tool.startswith(EDITOR_PREFIX)


def failed_criteria(
    cell: CellDone,
    sessions: list[SessionStatus],
    spans: list[ToolSpan],
    requests: list[RequestLog],
    has_report: bool,
    has_run_metrics: bool,
) -> list[str]:
    completed = {s.instance_id for s in sessions if s.status == "completed"}
    executed_tools = {s.tool for s in spans if s.executed and s.session_id in completed}
    checks = {
        "server alive after the run": cell.serve_alive,
        "run step exit status 0": cell.step_exit_status.get("run") == 0,
        "at least one completed session": bool(completed),
        "bash command executed in a completed session": any(is_bash(t) for t in executed_tools),
        "search or editor tool executed in a completed session": bool(executed_tools & SEARCH_OR_EDIT_TOOLS),
        "proxy logged a 200 for a completed session": any(r.status == 200 and r.session_id in completed for r in requests),
        "report/ written": has_report,
        "run_metrics.csv written": has_run_metrics,
    }
    return [name for name, ok in checks.items() if not ok]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run_dir = parse_args().run_dir
    failed = failed_criteria(
        CellDone.model_validate_json((run_dir / "cell_done.json").read_text()),
        read_jsonl(run_dir / "sessions.jsonl", SessionStatus),
        read_jsonl(run_dir / "tool_spans.jsonl", ToolSpan),
        read_jsonl(run_dir / "proxy_requests.jsonl", RequestLog),
        (run_dir / "report").is_dir(),
        (run_dir / "run_metrics.csv").is_file(),
    )
    for name in failed:
        print(f"FAIL: {name}", file=sys.stderr)
    if failed:
        sys.exit(1)
    print(f"smoke criteria met: {run_dir}")


if __name__ == "__main__":
    main()
