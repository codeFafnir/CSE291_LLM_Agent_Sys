"""Forward Harbor chat requests to vLLM and stamp a stable job_id.

The Continuum fork reads job_id from the top-level chat JSON field and
copies it into sampling_params.extra_args. The id is a hash of the first
user message, so turns of one trial share a job and trials do not.
"""

from __future__ import annotations

import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

UPSTREAM = "http://127.0.0.1:8000"
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 8001
TIMEOUT_S = 900


def first_user_text(body: dict[str, Any]) -> str:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        return json.dumps(content, sort_keys=True)
    return ""


def job_id_of(body: dict[str, Any]) -> str:
    text = first_user_text(body)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self._forward(b"")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        if self.path.split("?", 1)[0] == "/v1/chat/completions" and raw:
            body = json.loads(raw)
            body["job_id"] = job_id_of(body)
            raw = json.dumps(body).encode()
        self._forward(raw)

    def _forward(self, raw: bytes) -> None:
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"host", "content-length", "accept-encoding"}
        }
        request = Request(
            UPSTREAM + self.path,
            data=raw if self.command == "POST" else None,
            headers=headers,
            method=self.command,
        )
        try:
            upstream = urlopen(request, timeout=TIMEOUT_S)
        except HTTPError as error:
            payload = error.read()
            self.send_response(error.code)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        payload = upstream.read()
        self.send_response(upstream.status)
        content_type = upstream.headers.get("Content-Type")
        if content_type is not None:
            self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: object) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), ProxyHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
