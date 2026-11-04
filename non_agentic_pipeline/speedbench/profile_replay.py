"""Pick one representative prompt per tier and torch-profile it on a running vLLM server.

  profile_replay.py pick --subset <speed_subset.jsonl> --out-dir <dir>
  profile_replay.py replay --messages-json <dir>/profile_easy.json --out-dir <dir>/torch_profile/easy \
    --server-trace-dir <RESULTS_ROOT>/<SERVE_RUN_ID>/torch_profile

pick writes profile_<tier>.json (turn-1 chat messages of the prompt whose first_turn_tokens is the tier median).
replay sends 2 unprofiled warmup requests, resets the prefix cache, then runs a cold and a warm profiled pass.
Each pass is wrapped in /start_profile and /stop_profile; vLLM writes traces into --server-trace-dir and the
new files are moved into <out-dir>/cold and <out-dir>/warm. serve_vllm.sh sets VLLM_SERVER_DEV_MODE=1 for /reset_prefix_cache.
"""

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

from pydantic import BaseModel

from models import ChatMessage, SubsetRow, Tier

MODEL = "Qwen/Qwen3-14B-AWQ"
BASE_URL = "http://127.0.0.1:8000"
MAX_TOKENS = 256
WARMUP_REQUESTS = 2
REQUEST_TIMEOUT_S = 900.0
CONTROL_TIMEOUT_S = 60.0
STOP_TIMEOUT_S = 600.0
TIERS: tuple[Tier, ...] = ("easy", "medium")


class ProfilePrompt(BaseModel):
    question_id: str
    tier: Tier
    first_turn_tokens: int
    messages: list[ChatMessage]


def load_subset(path: Path) -> list[SubsetRow]:
    return [SubsetRow.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def median_row(rows: list[SubsetRow]) -> SubsetRow:
    ordered = sorted(rows, key=lambda r: r.first_turn_tokens)
    return ordered[(len(ordered) - 1) // 2]


def pick_prompts(rows: list[SubsetRow]) -> list[ProfilePrompt]:
    prompts: list[ProfilePrompt] = []
    for tier in TIERS:
        tier_rows = [r for r in rows if r.tier == tier]
        if not tier_rows:
            raise ValueError(f"no {tier} prompts in the subset")
        row = median_row(tier_rows)
        prompts.append(
            ProfilePrompt(
                question_id=row.question_id,
                tier=tier,
                first_turn_tokens=row.first_turn_tokens,
                messages=[ChatMessage(role="user", content=row.turns[0])],
            )
        )
    return prompts


def post(url: str, body: dict[str, object] | None, timeout_s: float) -> bytes:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return response.read()


def chat(prompt: ProfilePrompt) -> None:
    body: dict[str, object] = {
        "model": MODEL,
        "messages": [m.model_dump() for m in prompt.messages],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    post(f"{BASE_URL}/v1/chat/completions", body, REQUEST_TIMEOUT_S)


def move_new_traces(server_trace_dir: Path, before: set[str], out_dir: Path, label: str) -> None:
    # vLLM names traces by time inside one dir, so each pass's new files move into its label dir.
    target = out_dir / label
    target.mkdir(parents=True, exist_ok=True)
    for path in server_trace_dir.iterdir():
        if path.name not in before:
            shutil.move(str(path), target / path.name)


def profiled_pass(prompt: ProfilePrompt, out_dir: Path, label: str, server_trace_dir: Path) -> None:
    before = {p.name for p in server_trace_dir.iterdir()}
    post(f"{BASE_URL}/start_profile", None, CONTROL_TIMEOUT_S)
    chat(prompt)
    post(f"{BASE_URL}/stop_profile", None, STOP_TIMEOUT_S)
    move_new_traces(server_trace_dir, before, out_dir, label)


def run_pick(subset: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for prompt in pick_prompts(load_subset(subset)):
        path = out_dir / f"profile_{prompt.tier}.json"
        path.write_text(prompt.model_dump_json(indent=2))
        print(f"wrote {path} ({prompt.question_id}, {prompt.first_turn_tokens} tokens)")


def run_replay(messages_json: Path, out_dir: Path, server_trace_dir: Path) -> None:
    prompt = ProfilePrompt.model_validate_json(messages_json.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    server_trace_dir.mkdir(parents=True, exist_ok=True)
    # Warmup first (graphs, allocator, kernels), then reset so the cold pass starts with an empty prefix cache.
    for _ in range(WARMUP_REQUESTS):
        chat(prompt)
    post(f"{BASE_URL}/reset_prefix_cache", None, CONTROL_TIMEOUT_S)
    profiled_pass(prompt, out_dir, "cold", server_trace_dir)
    profiled_pass(prompt, out_dir, "warm", server_trace_dir)
    print(f"traces in {out_dir}/cold and {out_dir}/warm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pick = sub.add_parser("pick")
    pick.add_argument("--subset", type=Path, required=True)
    pick.add_argument("--out-dir", type=Path, required=True)
    replay = sub.add_parser("replay")
    replay.add_argument("--messages-json", type=Path, required=True)
    replay.add_argument("--out-dir", type=Path, required=True)
    replay.add_argument("--server-trace-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "pick":
        run_pick(args.subset, args.out_dir)
    else:
        run_replay(args.messages_json, args.out_dir, args.server_trace_dir)


if __name__ == "__main__":
    main()
