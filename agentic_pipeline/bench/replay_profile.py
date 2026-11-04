"""Replay one saved terminus-2 chat request against a running server, cold then warm, one profiler window each.

  replay_profile.py --engine vllm --messages-json turn.json --out-dir /data/cse291-results/profile-vllm/replay

turn.json is either the chat `messages` list or an object with a `messages` key.
Two unprofiled warmup requests run first. serve_vllm.sh already sets the profiler dir and VLLM_SERVER_DEV_MODE=1 (/reset_prefix_cache).
vLLM traces are moved from --server-trace-dir into --out-dir/<cold|warm>; SGLang writes straight to --out-dir/<cold|warm>.
SGLang's per-stage profiler needs num_steps and stops itself after that many prefill and decode steps.
"""

import argparse
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

MODEL = "Qwen/Qwen3-14B-AWQ"
MAX_TOKENS = 256
STAGE_STEPS = 8
WARMUP_REQUESTS = 2
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
            "num_steps": STAGE_STEPS,
            "activities": ["CPU", "GPU"],
            "with_stack": True,
        }
    return None


def chat(base: str, turn: Turn) -> None:
    body = {"model": MODEL, "messages": [m.model_dump() for m in turn.messages], "max_tokens": MAX_TOKENS}
    post(f"{base}/v1/chat/completions", body, REQUEST_TIMEOUT_S)


def collect_vllm_traces(server_trace_dir: Path, before: set[str], out_dir: Path, label: str) -> None:
    # vLLM names traces by time inside one dir, so each pass's new files move into its label dir.
    (out_dir / label).mkdir(parents=True, exist_ok=True)
    for path in server_trace_dir.iterdir():
        if path.name not in before:
            shutil.move(str(path), out_dir / label / path.name)


def profiled_request(
    engine: Engine, base: str, turn: Turn, out_dir: Path, label: str, server_trace_dir: Path
) -> None:
    before = {p.name for p in server_trace_dir.iterdir()} if engine == "vllm" else set()
    post(f"{base}/start_profile", start_body(engine, out_dir, label), 60.0)
    chat(base, turn)
    if engine == "vllm":
        post(f"{base}/stop_profile", None, 600.0)
        collect_vllm_traces(server_trace_dir, before, out_dir, label)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    parser.add_argument("--messages-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--server-trace-dir", type=Path, required=True, help="vLLM torch_profile_dir; unused for SGLang")
    args = parser.parse_args()

    engine: Engine = args.engine
    base = f"http://127.0.0.1:{PORT[engine]}"
    turn = load_turn(args.messages_json)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    args.server_trace_dir.mkdir(parents=True, exist_ok=True)

    # Warmup first (graphs, allocator, kernels), then flush so the cold pass starts with an empty prefix cache.
    for _ in range(WARMUP_REQUESTS):
        chat(base, turn)
    post(base + FLUSH_PATH[engine], None, 60.0)
    profiled_request(engine, base, turn, args.out_dir, "cold", args.server_trace_dir)
    profiled_request(engine, base, turn, args.out_dir, "warm", args.server_trace_dir)


if __name__ == "__main__":
    main()
