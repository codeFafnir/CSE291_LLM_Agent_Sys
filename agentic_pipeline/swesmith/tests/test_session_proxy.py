import asyncio
import json
import socket
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from pydantic import BaseModel

from session_proxy import ERROR_EXCERPT_CHARS, RequestLog, make_app

pytestmark = pytest.mark.asyncio

CHAT_PATH = "/v1/chat/completions"
PLAIN_PATH = "/metrics"
PLAIN_TEXT = "# HELP vllm_up 1\nvllm_up 1\n"
OVERFLOW_TEXT = "maximum context length exceeded " * 40


class Seen(BaseModel):
    path: str
    session_id: str | None
    request_id: str | None
    body: bytes


def usage_body(cached: int | None, with_details: bool) -> dict[str, object]:
    usage: dict[str, object] = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    if with_details:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return {"id": "chatcmpl-x", "choices": [], "usage": usage}


async def fake_chat(request: web.Request) -> web.Response:
    raw = await request.read()
    seen: list[Seen] = request.app["seen"]
    seen.append(
        Seen(
            path=request.path,
            session_id=request.headers.get("X-Session-ID"),
            request_id=request.headers.get("X-Request-Id"),
            body=raw,
        )
    )
    body = json.loads(raw)
    await asyncio.sleep(body.get("delay_s", 0))
    case = body.get("case", "ok")
    if case == "overflow":
        return web.Response(text=OVERFLOW_TEXT, status=400, content_type="text/plain")
    if case == "no_usage":
        return web.json_response({"id": "chatcmpl-x", "choices": []})
    if case == "no_details":
        return web.json_response(usage_body(None, False))
    return web.json_response(usage_body(7, True))


async def fake_plain(_: web.Request) -> web.Response:
    return web.Response(text=PLAIN_TEXT, status=200, content_type="text/plain")


@pytest_asyncio.fixture
async def upstream() -> AsyncIterator[TestServer]:
    app = web.Application()
    app["seen"] = []
    app.router.add_post(CHAT_PATH, fake_chat)
    app.router.add_get(PLAIN_PATH, fake_plain)
    server = TestServer(app)
    await server.start_server()
    yield server
    await server.close()


@pytest_asyncio.fixture
async def proxy(upstream: TestServer, tmp_path: Path) -> AsyncIterator[TestClient]:
    app = make_app(str(upstream.make_url("")).rstrip("/"), tmp_path / "requests.jsonl")
    client = TestClient(TestServer(app))
    await client.start_server()
    yield client
    await client.close()


def seen_of(upstream: TestServer) -> list[Seen]:
    return upstream.app["seen"]


def logs_of(tmp_path: Path) -> list[RequestLog]:
    lines = (tmp_path / "requests.jsonl").read_text().splitlines()
    return [RequestLog.model_validate_json(line) for line in lines]


def post(proxy: TestClient, session: str, body: dict[str, object]) -> object:
    return proxy.post(f"/s/{session}{CHAT_PATH}", data=json.dumps(body), headers={"Content-Type": "application/json"})


async def test_forwards_with_session_headers_and_unchanged_response(
    proxy: TestClient, upstream: TestServer, tmp_path: Path
) -> None:
    resp = await post(proxy, "sessA", {"case": "ok", "max_tokens": 64})
    assert resp.status == 200
    assert resp.content_type == "application/json"
    assert await resp.json() == usage_body(7, True)
    [seen] = seen_of(upstream)
    assert seen.path == CHAT_PATH
    assert seen.session_id == "sessA"
    assert seen.request_id == "sessA-t001"
    assert json.loads(seen.body) == {"case": "ok", "max_tokens": 64}
    [log] = logs_of(tmp_path)
    assert log.request_id == "sessA-t001"
    assert log.session_id == "sessA"
    assert log.turn == 1
    assert log.path == CHAT_PATH
    assert log.max_tokens == 64
    assert log.request_bytes == len(seen.body)


async def test_log_timing_and_usage(proxy: TestClient, tmp_path: Path) -> None:
    await post(proxy, "s", {"case": "ok", "delay_s": 0.02})
    [log] = logs_of(tmp_path)
    assert log.t_first_byte is not None
    assert log.t_send <= log.t_first_byte <= log.t_end
    assert log.status == 200
    assert log.error is None
    assert (log.prompt_tokens, log.completion_tokens, log.cached_tokens) == (100, 20, 7)


@pytest.mark.parametrize("case", ["no_details", "no_usage"])
async def test_cached_tokens_none_when_absent(proxy: TestClient, tmp_path: Path, case: str) -> None:
    resp = await post(proxy, "s", {"case": case})
    assert resp.status == 200
    [log] = logs_of(tmp_path)
    assert log.cached_tokens is None
    if case == "no_usage":
        assert log.prompt_tokens is None
        assert log.completion_tokens is None
    else:
        assert (log.prompt_tokens, log.completion_tokens) == (100, 20)


async def test_context_overflow_400_logged_and_returned_unchanged(proxy: TestClient, tmp_path: Path) -> None:
    resp = await post(proxy, "s", {"case": "overflow"})
    assert resp.status == 400
    assert resp.content_type == "text/plain"
    assert await resp.text() == OVERFLOW_TEXT
    [log] = logs_of(tmp_path)
    assert log.status == 400
    assert log.error == OVERFLOW_TEXT[:ERROR_EXCERPT_CHARS]
    assert len(OVERFLOW_TEXT) > ERROR_EXCERPT_CHARS
    assert log.prompt_tokens is None
    assert log.completion_tokens is None
    assert log.cached_tokens is None


async def test_upstream_connection_failure_returns_502_and_is_logged(tmp_path: Path) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        dead_port = sock.getsockname()[1]
    app = make_app(f"http://127.0.0.1:{dead_port}", tmp_path / "requests.jsonl")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        resp = await post(client, "s", {"case": "ok"})
        assert resp.status == 502
    finally:
        await client.close()
    [log] = logs_of(tmp_path)
    assert log.status == 502
    assert log.error is not None
    assert log.t_first_byte is None
    assert log.prompt_tokens is None


async def test_healthz(proxy: TestClient) -> None:
    resp = await proxy.get("/healthz")
    assert resp.status == 200
    assert await resp.text() == "ok"


async def test_turns_increment_per_session_under_concurrency(
    proxy: TestClient, upstream: TestServer, tmp_path: Path
) -> None:
    sessions = ["a", "b", "c"]
    per_session = 8
    bodies = [{"case": "ok", "delay_s": 0.005 * (i % 3)} for i in range(per_session)]
    responses = await asyncio.gather(*(post(proxy, s, b) for s in sessions for b in bodies))
    assert all(r.status == 200 for r in responses)
    logs = logs_of(tmp_path)
    assert len(logs) == len(sessions) * per_session
    for session in sessions:
        turns = sorted(log.turn for log in logs if log.session_id == session)
        assert turns == list(range(1, per_session + 1))
        ids = {s.request_id for s in seen_of(upstream) if s.session_id == session}
        assert ids == {f"{session}-t{turn:03d}" for turn in turns}
    assert all(log.request_id == f"{log.session_id}-t{log.turn:03d}" for log in logs)


async def test_log_is_valid_jsonl_one_line_per_request(proxy: TestClient, tmp_path: Path) -> None:
    count = 40
    await asyncio.gather(*(post(proxy, f"s{i % 5}", {"case": "ok", "delay_s": 0.001 * (i % 4)}) for i in range(count)))
    lines = (tmp_path / "requests.jsonl").read_text().splitlines()
    assert len(lines) == count
    assert len({json.loads(line)["request_id"] for line in lines}) == count


async def test_streaming_request_rejected_with_400(proxy: TestClient, upstream: TestServer, tmp_path: Path) -> None:
    resp = await post(proxy, "s", {"case": "ok", "stream": True})
    assert resp.status == 400
    assert "streaming is not supported" in await resp.text()
    assert seen_of(upstream) == []
    assert not (tmp_path / "requests.jsonl").exists()
    ok = await post(proxy, "s", {"case": "ok", "stream": False})
    assert ok.status == 200
    assert seen_of(upstream)[0].request_id == "s-t001"


async def test_non_json_200_on_non_chat_path_passed_through_and_logged(proxy: TestClient, tmp_path: Path) -> None:
    resp = await proxy.get(f"/s/s{PLAIN_PATH}")
    assert resp.status == 200
    assert resp.content_type == "text/plain"
    assert await resp.text() == PLAIN_TEXT
    [log] = logs_of(tmp_path)
    assert log.path == PLAIN_PATH
    assert log.status == 200
    assert log.error is None
    assert log.prompt_tokens is None
    assert log.completion_tokens is None
    assert log.cached_tokens is None
