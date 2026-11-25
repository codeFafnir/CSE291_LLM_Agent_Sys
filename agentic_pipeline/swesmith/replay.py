"""Replay recorded SWE-agent sessions against a session proxy with Poisson arrivals and an in-flight cap.

The corpus is a live run dir: sessions are <corpus>/recorded/<session>/tNNN.json (RecordedTurn; <corpus>/<session>/ when
there is no recorded/ dir) and timing is <corpus>/proxy_requests.jsonl (RequestLog). Each replayed request keeps the recorded messages, forces the recorded
completion length (max_tokens + ignore_eos, max_completion_tokens dropped) and waits the recorded tool gap (scaled by --gap-scale) before the next turn.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout
from pydantic import BaseModel, ConfigDict, JsonValue

from poisson import arrivals
from session_proxy import RecordedTurn, RequestLog

OK_STATUS = 200
RECORDED_DIR = "recorded"
REQUEST_TIMEOUT_S = 960


class ReplayTurn(BaseModel):
    model_config = ConfigDict(frozen=True)
    session_id: str
    turn: int
    request: dict[str, JsonValue]
    completion_tokens: int
    t_send: float
    t_end: float


class PlannedRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    turn: int
    body: dict[str, JsonValue]
    gap_s: float


class PlannedSession(BaseModel):
    model_config = ConfigDict(frozen=True)
    session_id: str
    start_offset_s: float
    requests: list[PlannedRequest]


class ReplayPlan(BaseModel):
    model_config = ConfigDict(frozen=True)
    concurrency: int
    sessions: list[PlannedSession]


class ReplayResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    session_id: str
    turn: int
    status: int
    completion_tokens: int | None


class ReplayArgs(BaseModel):
    model_config = ConfigDict(frozen=True)
    corpus: Path
    proxy_base: str
    rate: float
    count: int
    seed: int
    concurrency: int
    gap_scale: float
    out: Path


def read_request_logs(path: Path) -> dict[tuple[str, int], RequestLog]:
    logs = [RequestLog.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]
    return {(log.session_id, log.turn): log for log in logs}


def replay_turn_of(record: RecordedTurn, completion_tokens: int, logs: dict[tuple[str, int], RequestLog]) -> ReplayTurn:
    log = logs.get((record.session_id, record.turn))
    if log is None:
        raise ValueError(f"no proxy_requests.jsonl row for session={record.session_id} turn={record.turn}")
    if not isinstance(record.request, dict):
        raise ValueError(f"recorded request is not a JSON object: session={record.session_id} turn={record.turn}")
    return ReplayTurn(
        session_id=record.session_id,
        turn=record.turn,
        request=record.request,
        completion_tokens=completion_tokens,
        t_send=log.t_send,
        t_end=log.t_end,
    )


def load_session(session_dir: Path, logs: dict[tuple[str, int], RequestLog]) -> list[ReplayTurn]:
    records = [RecordedTurn.model_validate_json(p.read_text()) for p in session_dir.glob("t*.json")]
    turns = [
        replay_turn_of(r, r.completion_tokens, logs)
        for r in sorted(records, key=lambda r: r.turn)
        if r.status == OK_STATUS and r.completion_tokens is not None
    ]
    if not turns:
        raise ValueError(f"session has no replayable turns (status 200 with completion_tokens): {session_dir}")
    return turns


def load_sessions(corpus: Path) -> list[list[ReplayTurn]]:
    logs = read_request_logs(corpus / "proxy_requests.jsonl")
    record_root = corpus / RECORDED_DIR if (corpus / RECORDED_DIR).is_dir() else corpus
    session_dirs = sorted(d for d in record_root.iterdir() if d.is_dir() and any(d.glob("t*.json")))
    return [load_session(d, logs) for d in session_dirs]


def tool_gap_s(this_t_end: float, next_t_send: float) -> float:
    return max(0.0, next_t_send - this_t_end)


def planned_request_of(turn: ReplayTurn, gap_s: float) -> PlannedRequest:
    kept = {k: v for k, v in turn.request.items() if k != "max_completion_tokens"}
    body = {**kept, "max_tokens": turn.completion_tokens, "ignore_eos": True}
    return PlannedRequest(turn=turn.turn, body=body, gap_s=gap_s)


def planned_session_of(turns: list[ReplayTurn], start_offset_s: float, gap_scale: float) -> PlannedSession:
    gaps = [tool_gap_s(a.t_end, b.t_send) * gap_scale for a, b in zip(turns, turns[1:])] + [0.0]
    return PlannedSession(
        session_id=turns[0].session_id,
        start_offset_s=start_offset_s,
        requests=[planned_request_of(t, g) for t, g in zip(turns, gaps)],
    )


def plan_replay(
    sessions: list[list[ReplayTurn]], rate: float, count: int, seed: int, concurrency: int, gap_scale: float
) -> ReplayPlan:
    if gap_scale < 0:
        raise ValueError(f"gap_scale must be >= 0, got gap_scale={gap_scale}")
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got concurrency={concurrency}")
    if count > len(sessions):
        raise ValueError(f"count={count} exceeds the corpus size of {len(sessions)} sessions")
    chosen = sorted(sessions, key=lambda s: s[0].session_id)[:count]
    offsets = arrivals(rate, count, seed)
    return ReplayPlan(
        concurrency=concurrency,
        sessions=[planned_session_of(turns, offset, gap_scale) for turns, offset in zip(chosen, offsets)],
    )


def completion_tokens_of(status: int, payload: bytes) -> int | None:
    if status != OK_STATUS:
        return None
    body = json.loads(payload)
    usage = body.get("usage") if isinstance(body, dict) else None
    value = usage.get("completion_tokens") if isinstance(usage, dict) else None
    return value if isinstance(value, int) else None


async def post_turn(client: ClientSession, proxy_base: str, session_id: str, request: PlannedRequest) -> ReplayResult:
    url = f"{proxy_base.rstrip('/')}/s/{session_id}/v1/chat/completions"
    async with client.post(url, json=request.body) as response:
        payload = await response.read()
        status = response.status
    return ReplayResult(
        session_id=session_id, turn=request.turn, status=status, completion_tokens=completion_tokens_of(status, payload)
    )


async def replay_session(client: ClientSession, proxy_base: str, session: PlannedSession) -> list[ReplayResult]:
    results: list[ReplayResult] = []
    for request in session.requests:
        results.append(await post_turn(client, proxy_base, session.session_id, request))
        await asyncio.sleep(request.gap_s)
    return results


async def run_replay(plan: ReplayPlan, proxy_base: str, client: ClientSession) -> list[ReplayResult]:
    slots = asyncio.Semaphore(plan.concurrency)
    t0 = time.monotonic()

    async def run_admitted(session: PlannedSession) -> list[ReplayResult]:
        try:
            return await replay_session(client, proxy_base, session)
        finally:
            slots.release()

    tasks: list[asyncio.Task[list[ReplayResult]]] = []
    for session in plan.sessions:
        await asyncio.sleep(max(0.0, t0 + session.start_offset_s - time.monotonic()))
        await slots.acquire()
        tasks.append(asyncio.create_task(run_admitted(session)))
    return [result for session_results in await asyncio.gather(*tasks) for result in session_results]


async def run_cli(args: ReplayArgs) -> list[ReplayResult]:
    plan = plan_replay(load_sessions(args.corpus), args.rate, args.count, args.seed, args.concurrency, args.gap_scale)
    async with ClientSession(timeout=ClientTimeout(total=REQUEST_TIMEOUT_S)) as client:
        return await run_replay(plan, args.proxy_base, client)


def parse_args() -> ReplayArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--proxy-base", required=True, help="e.g. http://127.0.0.1:8100")
    parser.add_argument("--rate", type=float, required=True, help="sessions per second")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--gap-scale", type=float, required=True, help="multiplier on recorded tool gaps; 1 = unchanged")
    parser.add_argument("--out", type=Path, required=True)
    return ReplayArgs.model_validate(vars(parser.parse_args()))


def main() -> None:
    args = parse_args()
    results = asyncio.run(run_cli(args))
    args.out.write_text("".join(r.model_dump_json() + "\n" for r in results))


if __name__ == "__main__":
    main()
