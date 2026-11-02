"""Rebuild a late-turn chat request from a Harbor terminus-2 trajectory.json for replay_profile.py.

  trajectory_to_turn.py --trajectory agent/trajectory.json --max-chars 30000 --out replay_turn.json

Keeps the task prompt, then agent/observation pairs in order until max-chars is reached.
"""

import argparse
import json
from pathlib import Path

from pydantic import BaseModel


class ToolCall(BaseModel):
    arguments: dict[str, str | float]


class Result(BaseModel):
    content: str


class Observation(BaseModel):
    results: list[Result]


class Step(BaseModel):
    source: str
    message: str
    tool_calls: list[ToolCall] = []
    observation: Observation | None = None


class Trajectory(BaseModel):
    steps: list[Step]


class Message(BaseModel):
    role: str
    content: str


def agent_text(step: Step) -> str:
    keystrokes = "".join(str(call.arguments.get("keystrokes", "")) for call in step.tool_calls)
    return f"{step.message}\nKeystrokes: {keystrokes}"


def build_messages(trajectory: Trajectory, max_chars: int) -> list[Message]:
    first, *rest = trajectory.steps
    messages = [Message(role="user", content=first.message)]
    size = len(first.message)
    for step in rest:
        if step.source != "agent" or step.observation is None:
            continue
        pair = [
            Message(role="assistant", content=agent_text(step)),
            Message(role="user", content="\n".join(r.content for r in step.observation.results)),
        ]
        size += sum(len(m.content) for m in pair)
        if size > max_chars:
            break
        messages.extend(pair)
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--max-chars", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    trajectory = Trajectory.model_validate_json(args.trajectory.read_text())
    messages = build_messages(trajectory, args.max_chars)
    args.out.write_text(json.dumps({"messages": [m.model_dump() for m in messages]}))
    print(f"wrote {len(messages)} messages to {args.out}")


if __name__ == "__main__":
    main()
