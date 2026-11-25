import asyncio
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer
from pydantic import BaseModel, ConfigDict, JsonValue

from poisson import arrivals
from replay import (
    PlannedRequest,
    PlannedSession,
    ReplayPlan,
    ReplayTurn,
    load_sessions,
    plan_replay,
    run_replay,
    tool_gap_s,
)
from session_proxy import RecordedTurn, RequestLog, write_record

CHAT_ROUTE = "/s/{session}/v1/chat/completions"
GAP_S = 0.05


class Arrival(BaseModel):
    session_id: str
    t_start: float
    t_end: float
    body: dict[str, JsonValue]


class Tracker(BaseModel):
    arrivals: list[Arrival]
    active: set[str]
    peak: int


class Upstream(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    server: TestServer
    tracker: Tracker


def messages_of(session_id: str, turn: int) -> list[JsonValue]:
    return [{"role": "user", "content": f"{session_id} turn {turn}"}]


def write_corpus(corpus: Path, record_dir: Path, sessions: dict[str, list[tuple[int, int | None, int]]], tool_gap: float) -> None:
    """sessions: session_id -> [(turn, completion_tokens, status)]; each turn takes 1s, next send follows tool_gap later."""
    logs: list[RequestLog] = []
    for session_id, turns in sessions.items():
        for turn, completion_tokens, status in turns:
            t_send = 100.0 + turn * (1.0 + tool_gap)
            request: JsonValue = {"model": "m", "messages": messages_of(session_id, turn), "max_tokens": 2048, "max_completion_tokens": 2048}
            write_record(
                record_dir,
                RecordedTurn(
                    session_id=session_id, turn=turn, request=request, completion_tokens=completion_tokens, status=status
                ),
            )
            logs.append(
                RequestLog(
                    session_id=session_id,
                    turn=turn,
                    request_id=f"{session_id}-t{turn:03d}",
                    path="/v1/chat/completions",
                    t_send=t_send,
                    t_first_byte=t_send + 0.5,
                    t_end=t_send + 1.0,
                    status=status,
                    error=None,
                    max_tokens=2048,
                    request_bytes=10,
                    prompt_tokens=10,
                    completion_tokens=completion_tokens,
                    cached_tokens=None,
                )
            )
    (corpus / "proxy_requests.jsonl").write_text("".join(log.model_dump_json() + "\n" for log in logs))


def turn_of(session_id: str, turn: int, completion_tokens: int, t_send: float, t_end: float) -> ReplayTurn:
    return ReplayTurn(
        session_id=session_id,
        turn=turn,
        request={"model": "m", "messages": messages_of(session_id, turn)},
        completion_tokens=completion_tokens,
        t_send=t_send,
        t_end=t_end,
    )


def synthetic_sessions(n: int, turns: int) -> list[list[ReplayTurn]]:
    return [
        [turn_of(f"s{i:02d}", t, 10 + t, 10.0 * t, 10.0 * t + 1.0) for t in range(1, turns + 1)] for i in range(n)
    ]


def plan_of(n: int, turns: int, concurrency: int, gap_s: float, offsets: list[float]) -> ReplayPlan:
    return ReplayPlan(
        concurrency=concurrency,
        sessions=[
            PlannedSession(
                session_id=f"s{i:02d}",
                start_offset_s=offsets[i],
                requests=[
                    PlannedRequest(
                        turn=t,
                        body={"messages": messages_of(f"s{i:02d}", t), "max_tokens": 10 + t, "ignore_eos": True},
                        gap_s=gap_s if t < turns else 0.0,
                    )
                    for t in range(1, turns + 1)
                ],
            )
            for i in range(n)
        ],
    )


@pytest_asyncio.fixture
async def upstream() -> AsyncIterator[Upstream]:
    tracker = Tracker(arrivals=[], active=set(), peak=0)

    async def chat(request: web.Request) -> web.Response:
        session_id = request.match_info["session"]
        body = await request.json()
        t_start = time.monotonic()
        tracker.active.add(session_id)
        tracker.peak = max(tracker.peak, len(tracker.active))
        await asyncio.sleep(0.02)
        if body.get("last") is True:
            tracker.active.discard(session_id)
        tracker.arrivals.append(Arrival(session_id=session_id, t_start=t_start, t_end=time.monotonic(), body=body))
        return web.json_response({"usage": {"prompt_tokens": 5, "completion_tokens": body["max_tokens"]}})

    app = web.Application()
    app.router.add_post(CHAT_ROUTE, chat)
    server = TestServer(app)
    await server.start_server()
    yield Upstream(server=server, tracker=tracker)
    await server.close()


def base_of(upstream: Upstream) -> str:
    return str(upstream.server.make_url("")).rstrip("/")


def mark_last(plan: ReplayPlan) -> ReplayPlan:
    """Tag each session's last request body so the fake upstream can tell when a session leaves flight."""
    return ReplayPlan(
        concurrency=plan.concurrency,
        sessions=[
            s.model_copy(
                update={
                    "requests": [
                        r.model_copy(update={"body": {**r.body, "last": i == len(s.requests) - 1}})
                        for i, r in enumerate(s.requests)
                    ]
                }
            )
            for s in plan.sessions
        ],
    )


def test_tool_gap_s() -> None:
    assert tool_gap_s(10.0, 12.5) == pytest.approx(2.5)
    assert tool_gap_s(10.0, 9.0) == 0.0


def test_load_sessions_order_skips_and_join(tmp_path: Path) -> None:
    write_corpus(
        tmp_path,
        tmp_path,
        {
            "b": [(2, 30, 200), (1, 20, 200)],
            "a": [(1, 10, 200), (2, None, 200), (3, 15, 400), (4, 40, 200)],
        },
        tool_gap=2.0,
    )
    sessions = load_sessions(tmp_path)
    assert [[t.session_id for t in s] for s in sessions] == [["a", "a"], ["b", "b"]]
    assert [t.turn for t in sessions[0]] == [1, 4]
    assert [t.turn for t in sessions[1]] == [1, 2]
    assert sessions[0][0].t_send == pytest.approx(100.0 + 3.0)
    assert sessions[0][0].t_end == pytest.approx(100.0 + 3.0 + 1.0)
    assert sessions[0][0].completion_tokens == 10


def test_load_sessions_live_run_layout(tmp_path: Path) -> None:
    write_corpus(tmp_path, tmp_path / "recorded", {"a": [(0, 10, 200), (1, 20, 200)]}, tool_gap=1.0)
    (tmp_path / "sweagent" / "x").mkdir(parents=True)
    sessions = load_sessions(tmp_path)
    assert [[t.turn for t in s] for s in sessions] == [[0, 1]]
    assert [t.session_id for t in sessions[0]] == ["a", "a"]


def test_load_sessions_empty_session_raises(tmp_path: Path) -> None:
    write_corpus(tmp_path, tmp_path, {"a": [(1, 10, 200)], "b": [(1, None, 200), (2, 5, 500)]}, tool_gap=1.0)
    with pytest.raises(ValueError, match="b"):
        load_sessions(tmp_path)


def test_plan_gap_scale_and_last_gap_zero() -> None:
    sessions = [[turn_of("s", 1, 10, 0.0, 1.0), turn_of("s", 2, 20, 4.0, 5.0), turn_of("s", 3, 30, 5.0, 6.0)]]
    plan = plan_replay(sessions, rate=1.0, count=1, seed=0, concurrency=1, gap_scale=0.5)
    assert [r.gap_s for r in plan.sessions[0].requests] == [pytest.approx(1.5), 0.0, 0.0]
    unscaled = plan_replay(sessions, rate=1.0, count=1, seed=0, concurrency=1, gap_scale=1.0)
    assert [r.gap_s for r in unscaled.sessions[0].requests] == [pytest.approx(3.0), 0.0, 0.0]


def test_plan_body_forces_completion_and_keeps_messages() -> None:
    sessions = synthetic_sessions(2, 3)
    before = sessions[0][1].model_dump()
    plan = plan_replay(sessions, rate=1.0, count=2, seed=0, concurrency=2, gap_scale=1.0)
    request = plan.sessions[0].requests[1]
    assert request.body["max_tokens"] == 12
    assert request.body["ignore_eos"] is True
    assert request.body["messages"] == messages_of("s00", 2)
    assert sessions[0][1].model_dump() == before


def test_plan_takes_first_count_sorted_with_poisson_offsets() -> None:
    sessions = list(reversed(synthetic_sessions(5, 2)))
    plan = plan_replay(sessions, rate=0.5, count=3, seed=7, concurrency=2, gap_scale=1.0)
    assert [s.session_id for s in plan.sessions] == ["s00", "s01", "s02"]
    assert [s.start_offset_s for s in plan.sessions] == arrivals(0.5, 3, 7)


def test_plan_deterministic_from_same_corpus_and_seed(tmp_path: Path) -> None:
    write_corpus(tmp_path, tmp_path, {f"s{i}": [(1, 10, 200), (2, 20, 200)] for i in range(4)}, tool_gap=1.5)
    first = plan_replay(load_sessions(tmp_path), 0.3, 3, 11, 2, 1.0)
    second = plan_replay(load_sessions(tmp_path), 0.3, 3, 11, 2, 1.0)
    assert first == second
    assert first.model_dump_json() == second.model_dump_json()
    assert first != plan_replay(load_sessions(tmp_path), 0.3, 3, 12, 2, 1.0)


def test_plan_validation() -> None:
    sessions = synthetic_sessions(2, 2)
    with pytest.raises(ValueError, match="gap_scale"):
        plan_replay(sessions, 1.0, 2, 0, 1, -0.1)
    with pytest.raises(ValueError, match="concurrency"):
        plan_replay(sessions, 1.0, 2, 0, 0, 1.0)
    with pytest.raises(ValueError, match="count"):
        plan_replay(sessions, 1.0, 3, 0, 1, 1.0)


@pytest.mark.asyncio
async def test_turn_order_preserved_and_results(upstream: Upstream) -> None:
    plan = plan_of(n=2, turns=4, concurrency=2, gap_s=0.0, offsets=[0.0, 0.0])
    async with ClientSession() as client:
        results = await run_replay(plan, base_of(upstream), client)
    for session_id in ("s00", "s01"):
        sent = [a.body["max_tokens"] for a in upstream.tracker.arrivals if a.session_id == session_id]
        assert sent == [11, 12, 13, 14]
        got = [(r.turn, r.status, r.completion_tokens) for r in results if r.session_id == session_id]
        assert got == [(1, 200, 11), (2, 200, 12), (3, 200, 13), (4, 200, 14)]


@pytest.mark.asyncio
async def test_gap_sleep_within_tolerance(upstream: Upstream) -> None:
    plan = plan_of(n=1, turns=3, concurrency=1, gap_s=GAP_S, offsets=[0.0])
    async with ClientSession() as client:
        await run_replay(plan, base_of(upstream), client)
    arrived = upstream.tracker.arrivals
    gaps = [b.t_start - a.t_end for a, b in zip(arrived, arrived[1:])]
    assert len(gaps) == 2
    assert all(abs(g - GAP_S) < 0.05 for g in gaps)


@pytest.mark.asyncio
async def test_outgoing_body_matches_recorded(upstream: Upstream, tmp_path: Path) -> None:
    write_corpus(tmp_path, tmp_path, {"a": [(1, 17, 200), (2, 23, 200)]}, tool_gap=0.0)
    plan = plan_replay(load_sessions(tmp_path), rate=1000.0, count=1, seed=0, concurrency=1, gap_scale=0.0)
    async with ClientSession() as client:
        await run_replay(plan, base_of(upstream), client)
    bodies = [a.body for a in upstream.tracker.arrivals]
    assert [b["max_tokens"] for b in bodies] == [17, 23]
    assert all("max_completion_tokens" not in b for b in bodies)
    assert all(b["ignore_eos"] is True for b in bodies)
    assert [b["messages"] for b in bodies] == [messages_of("a", 1), messages_of("a", 2)]
    assert all(b["model"] == "m" for b in bodies)


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [1, 2, 3])
async def test_inflight_cap_never_exceeded(upstream: Upstream, concurrency: int) -> None:
    plan = mark_last(plan_of(n=7, turns=2, concurrency=concurrency, gap_s=0.02, offsets=[0.0] * 7))
    async with ClientSession() as client:
        results = await run_replay(plan, base_of(upstream), client)
    assert len(results) == 14
    assert upstream.tracker.peak == concurrency

