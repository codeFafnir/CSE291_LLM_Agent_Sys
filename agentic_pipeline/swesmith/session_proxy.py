"""Reverse proxy from /s/<session>/<path> to vLLM /<path> that tags sessions and logs every request.

It adds X-Session-ID (read by the Continuum scheduler) and X-Request-Id=<session>-t<turn>, and appends
one RequestLog JSON line per request: send / first-byte / end times, status, token usage, cached tokens.
Failed requests (context overflow, upstream error) are logged with the error text and returned unchanged.
Responses are buffered, so requests with "stream": true are rejected with a 400 (streaming unsupported).
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from aiohttp import ClientError, ClientSession, ClientTimeout, web
from pydantic import BaseModel, ConfigDict

UPSTREAM_TIMEOUT_S = 900
ERROR_EXCERPT_CHARS = 500
HOP_BY_HOP = {"host", "content-length", "accept-encoding", "connection", "transfer-encoding"}


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True)
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int | None


class RequestLog(BaseModel):
    model_config = ConfigDict(frozen=True)
    session_id: str
    turn: int
    request_id: str
    path: str
    t_send: float
    t_first_byte: float | None
    t_end: float
    status: int
    error: str | None
    max_tokens: int | None
    request_bytes: int
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_tokens: int | None


def request_id_of(session_id: str, turn: int) -> str:
    return f"{session_id}-t{turn:03d}"


def max_tokens_of(raw: bytes) -> int | None:
    if not raw:
        return None
    body = json.loads(raw)
    value = body.get("max_tokens") if isinstance(body, dict) else None
    return value if isinstance(value, int) else None


def is_stream_request(raw: bytes) -> bool:
    if not raw:
        return False
    body = json.loads(raw)
    return isinstance(body, dict) and body.get("stream") is True


def is_chat_path(path: str) -> bool:
    return path.endswith("/chat/completions")


def is_json(payload: bytes) -> bool:
    try:
        json.loads(payload)
    except ValueError:
        return False
    return True


def usage_of(payload: bytes) -> Usage | None:
    body = json.loads(payload)
    usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict):
        return None
    details = usage.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    return Usage(
        prompt_tokens=int(usage["prompt_tokens"]),
        completion_tokens=int(usage["completion_tokens"]),
        cached_tokens=cached if isinstance(cached, int) else None,
    )


def error_text(status: int, payload: bytes) -> str | None:
    if status < 400:
        return None
    return payload.decode(errors="replace")[:ERROR_EXCERPT_CHARS]


def log_of(
    session_id: str,
    turn: int,
    path: str,
    t_send: float,
    t_first_byte: float | None,
    t_end: float,
    status: int,
    payload: bytes,
    raw: bytes,
) -> RequestLog:
    parsable = is_chat_path(path) or is_json(payload)
    usage = usage_of(payload) if status < 400 and payload and parsable else None
    return RequestLog(
        session_id=session_id,
        turn=turn,
        request_id=request_id_of(session_id, turn),
        path=path,
        t_send=t_send,
        t_first_byte=t_first_byte,
        t_end=t_end,
        status=status,
        error=error_text(status, payload),
        max_tokens=max_tokens_of(raw),
        request_bytes=len(raw),
        prompt_tokens=usage.prompt_tokens if usage else None,
        completion_tokens=usage.completion_tokens if usage else None,
        cached_tokens=usage.cached_tokens if usage else None,
    )


class ProxyState:
    def __init__(self, upstream: str, log_path: Path) -> None:
        self.upstream = upstream
        self.log_path = log_path
        self.turns: dict[str, int] = {}
        self.lock = asyncio.Lock()

    async def next_turn(self, session_id: str) -> int:
        async with self.lock:
            turn = self.turns.get(session_id, 0) + 1
            self.turns[session_id] = turn
            return turn

    async def append(self, record: RequestLog) -> None:
        async with self.lock:
            with self.log_path.open("a") as handle:
                handle.write(record.model_dump_json() + "\n")


async def handle(request: web.Request) -> web.Response:
    state: ProxyState = request.app["state"]
    session_id = request.match_info["session"]
    tail = request.match_info["tail"]
    raw = await request.read()
    if is_stream_request(raw):
        return web.Response(text="streaming is not supported by session_proxy", status=400)
    turn = await state.next_turn(session_id)
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
    headers["X-Session-ID"] = session_id
    headers["X-Request-Id"] = request_id_of(session_id, turn)
    path = f"/{tail}"
    t_send = time.time()
    t_first_byte: float | None = None
    try:
        async with request.app["client"].request(
            request.method, state.upstream + path, data=raw or None, headers=headers
        ) as upstream:
            t_first_byte = time.time()
            payload = await upstream.read()
            status = upstream.status
            content_type = upstream.headers.get("Content-Type", "application/json")
    except (ClientError, asyncio.TimeoutError) as error:
        payload = f"{type(error).__name__}: {error}".encode()
        status, content_type = 502, "text/plain"
    t_end = time.time()
    await state.append(log_of(session_id, turn, path, t_send, t_first_byte, t_end, status, payload, raw))
    return web.Response(body=payload, status=status, content_type=content_type.split(";")[0])


async def health(_: web.Request) -> web.Response:
    return web.Response(text="ok")


async def open_client(app: web.Application) -> None:
    app["client"] = ClientSession(timeout=ClientTimeout(total=UPSTREAM_TIMEOUT_S))


async def close_client(app: web.Application) -> None:
    await app["client"].close()


def make_app(upstream: str, log_path: Path) -> web.Application:
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["state"] = ProxyState(upstream, log_path)
    app.on_startup.append(open_client)
    app.on_cleanup.append(close_client)
    app.router.add_get("/healthz", health)
    app.router.add_route("*", "/s/{session}/{tail:.*}", handle)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, help="e.g. http://127.0.0.1:8000")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    web.run_app(make_app(args.upstream, args.log_path), host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
