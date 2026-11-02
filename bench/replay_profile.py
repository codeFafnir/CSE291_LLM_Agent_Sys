"""Replay one saved terminus-2 chat request against a running server, cold then warm, one profiler window each.

  replay_profile.py --engine vllm --messages-json turn.json --out-dir /data/cse291-results/profile-vllm/replay

turn.json is either the chat `messages` list or an object with a `messages` key.
vLLM must be started with PROFILE_DIR and VLLM_SERVER_DEV_MODE=1 set (the latter exposes /reset_prefix_cache). Traces land in the server's profiler dir (vLLM) or --out-dir (SGLang).
"""

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

MODEL = "Qwen/Qwen3-14B-AWQ"
MAX_TOKENS = 256
REQUEST_TIMEOUT_S = 900.0
Engine = Literal["vllm", "sglang"]

PORT: dict[Engine, int] = {"vllm": 8000, "sglang": 30000}
FLUSH_PATH: dict[Engine, str] = {"vllm": "/reset_prefix_cache", "sglang": "/flush_cache"}


class Message(BaseModel):
    role: str
    content: str


class Turn(BaseModel):
    messages: list[Message]


def load_turn(path: Path) -> Turn:
    raw = json.loads(path.read_text())
    return Turn.model_validate(raw if isinstance(raw, dict) else {"messages": raw})


def post(url: str, body: dict[str, object] | None, timeout_s: float) -> bytes:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return response.read()


def start_body(engine: Engine, out_dir: Path, label: str) -> dict[str, object] | None:
    if engine == "sglang":
        return {
            "output_dir": str(out_dir / label),
            "profile_by_stage": True,
            "activities": ["CPU", "GPU"],
            "with_stack": True,
        }
    return None


def profiled_request(engine: Engine, base: str, turn: Turn, out_dir: Path, label: str) -> None:
    post(f"{base}/start_profile", start_body(engine, out_dir, label), 60.0)
    body = {"model": MODEL, "messages": [m.model_dump() for m in turn.messages], "max_tokens": MAX_TOKENS}
    post(f"{base}/v1/chat/completions", body, REQUEST_TIMEOUT_S)
    post(f"{base}/stop_profile", None, 600.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    parser.add_argument("--messages-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    engine: Engine = args.engine
    base = f"http://127.0.0.1:{PORT[engine]}"
    turn = load_turn(args.messages_json)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    post(base + FLUSH_PATH[engine], None, 60.0)
    profiled_request(engine, base, turn, args.out_dir, "cold")
    profiled_request(engine, base, turn, args.out_dir, "warm")


if __name__ == "__main__":
    main()
