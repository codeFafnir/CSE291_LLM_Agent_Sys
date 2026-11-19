from check_smoke import CellDone, SessionStatus, failed_criteria, is_bash
from session_proxy import RequestLog
from tool_span import ToolSpan


def span(session: str, tool: str, executed: bool) -> ToolSpan:
    return ToolSpan(
        session_id=session, step=1, tool=tool, action_head=tool, executed=executed,
        t_step_start=0.0, t_tool_start=0.0, t_tool_end=1.0, execution_time_s=1.0,
        timed_out=False, step_exit_status=None, observation_chars=10,
    )


def request(session: str, status: int) -> RequestLog:
    return RequestLog(
        session_id=session, turn=1, request_id=f"{session}-t001", path="/v1/chat/completions",
        t_send=0.0, t_first_byte=0.1, t_end=0.2, status=status, error=None, max_tokens=2048,
        request_bytes=10, prompt_tokens=5, completion_tokens=2, cached_tokens=None,
    )


CELL = CellDone(serve_alive=True, step_exit_status={"run": 0, "run_csv": 0})
SESSIONS = [SessionStatus(instance_id="a", status="completed"), SessionStatus(instance_id="b", status="no_trajectory")]
SPANS = [span("a", "grep", True), span("a", "search_dir", True), span("a", "submit", True)]
REQUESTS = [request("a", 200)]


def test_all_criteria_met() -> None:
    assert failed_criteria(CELL, SESSIONS, SPANS, REQUESTS, True, True) == []


def test_is_bash() -> None:
    assert is_bash("grep") and is_bash("python")
    assert not is_bash("submit") and not is_bash("search_dir") and not is_bash("str_replace_editor:view")


def test_spans_of_incomplete_sessions_do_not_count() -> None:
    spans = [span("b", "grep", True), span("b", "find_file", True), span("a", "submit", True)]
    failed = failed_criteria(CELL, SESSIONS, spans, REQUESTS, True, True)
    assert failed == [
        "bash command executed in a completed session",
        "search or editor tool executed in a completed session",
    ]


def test_unexecuted_spans_do_not_count() -> None:
    spans = [span("a", "grep", False), span("a", "search_dir", True)]
    assert failed_criteria(CELL, SESSIONS, spans, REQUESTS, True, True) == ["bash command executed in a completed session"]


def test_no_completed_session_fails_dependent_criteria() -> None:
    sessions = [SessionStatus(instance_id="a", status="crashed")]
    failed = failed_criteria(CellDone(serve_alive=False, step_exit_status={"run": 1}), sessions, SPANS, REQUESTS, False, False)
    assert len(failed) == 8


def test_non_200_proxy_status_fails() -> None:
    failed = failed_criteria(CELL, SESSIONS, SPANS, [request("a", 400)], True, True)
    assert failed == ["proxy logged a 200 for a completed session"]
