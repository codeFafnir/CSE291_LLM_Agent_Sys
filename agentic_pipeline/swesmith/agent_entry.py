"""Run `sweagent run-batch` with a hook that appends one ToolSpan JSON line per agent step.

Usage: agent_entry.py <tool_spans.jsonl> <session_id> -- <sweagent run-batch args...>
Run it from the sweagent env. Timestamps are wall-clock epoch seconds, so they join with the proxy log.
"""

import sys
import time
from collections.abc import Callable
from pathlib import Path

from sweagent.agent import agents
from sweagent.agent.hooks.abstract import AbstractAgentHook
from sweagent.run import run_batch
from sweagent.types import AgentInfo, StepOutput
from swerex.deployment.docker import DockerDeployment

from tool_span import ToolSpan

TIMEOUT_MARKER = "was cancelled because it took more than"
ACTION_HEAD_CHARS = 120
BUILDER_OLD = "FROM  python:3.11.9-slim-bookworm AS builder\nRUN apt-get update"
BUILDER_NEW = "FROM  ubuntu:20.04 AS builder\nENV DEBIAN_FRONTEND=noninteractive\nRUN apt-get update"
WGET_OLD = "    wget \\\n"
WGET_NEW = "    wget \\\n    ca-certificates \\\n"
SWEREX_LINK = "RUN ln -s /root/python3.11/bin/swerex-remote /usr/local/bin/swerex-remote\n"
PYTHON_LINK = "RUN ln -s /root/python3.11/bin/python3 /usr/local/bin/python\n"
LIBC6_STEP = "RUN apt-get update && apt-get install -y \\\n    libc6 \\\n    && rm -rf /var/lib/apt/lists/*\n"


def tool_end_time(
    tool_start: float, tool_end: float | None, timed_out: bool, execution_timeout_s: float, now: float
) -> float:
    """Timeouts end at the cancellation time; other unfinished tools end now; finished tools keep their stamp."""
    if timed_out:
        return tool_start + execution_timeout_s
    return now if tool_end is None else tool_end


def replace_once(text: str, old: str, new: str, name: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"swe-rex glibc Dockerfile: expected exactly one {name} block, found {text.count(old)}: {old!r}")
    return text.replace(old, new)


def patch_glibc_dockerfile(dockerfile: str) -> str:
    """Build python on ubuntu:20.04 (glibc 2.31), drop the final-stage libc6 install, and expose `python` for tool shebangs."""
    dockerfile = replace_once(dockerfile, BUILDER_OLD, BUILDER_NEW, "builder stage")
    dockerfile = replace_once(dockerfile, WGET_OLD, WGET_NEW, "builder wget package")
    dockerfile = replace_once(dockerfile, LIBC6_STEP, "", "final-stage libc6 install")
    return replace_once(dockerfile, SWEREX_LINK, SWEREX_LINK + PYTHON_LINK, "swerex-remote symlink")


def install_glibc_patch() -> None:
    original: Callable[[DockerDeployment], str] = DockerDeployment.glibc_dockerfile.fget

    def patched(self: DockerDeployment) -> str:
        return patch_glibc_dockerfile(original(self))

    DockerDeployment.glibc_dockerfile = property(patched)


def tool_name(action: str) -> str:
    tokens = action.split()
    if not tokens:
        return "none"
    if tokens[0] == "str_replace_editor" and len(tokens) > 1:
        return f"str_replace_editor:{tokens[1]}"
    return tokens[0]


class SpanHook(AbstractAgentHook):
    def __init__(self, out_path: Path, session_id: str, execution_timeout_s: float) -> None:
        self._out_path = out_path
        self._session_id = session_id
        self._execution_timeout_s = execution_timeout_s
        self._step = 0
        self._step_start = 0.0
        self._tool_start: float | None = None
        self._tool_end: float | None = None
        self._started_action = ""

    def on_step_start(self) -> None:
        self._step += 1
        self._step_start = time.time()
        self._tool_start = None
        self._tool_end = None
        self._started_action = ""

    def on_action_started(self, *, step: StepOutput) -> None:
        self._tool_start = time.time()
        self._tool_end = None
        self._started_action = step.action

    def on_action_executed(self, *, step: StepOutput) -> None:
        self._tool_end = time.time()

    def on_step_done(self, *, step: StepOutput, info: AgentInfo) -> None:
        exit_status = None if step.exit_status is None else str(step.exit_status)
        timed_out = TIMEOUT_MARKER in step.observation or (exit_status is not None and "timeout" in exit_status)
        tool_end: float | None = None
        execution_time_s = step.execution_time
        if self._tool_start is not None:
            tool_end = tool_end_time(self._tool_start, self._tool_end, timed_out, self._execution_timeout_s, time.time())
            execution_time_s = tool_end - self._tool_start
        action = step.action if step.action.strip() else self._started_action
        span = ToolSpan(
            session_id=self._session_id,
            step=self._step,
            tool=tool_name(action),
            action_head=action.strip()[:ACTION_HEAD_CHARS],
            executed=self._tool_start is not None,
            t_step_start=self._step_start,
            t_tool_start=self._tool_start,
            t_tool_end=tool_end,
            execution_time_s=execution_time_s,
            timed_out=timed_out,
            step_exit_status=exit_status,
            observation_chars=len(step.observation),
        )
        with self._out_path.open("a") as handle:
            handle.write(span.model_dump_json() + "\n")


def main() -> None:
    out_path, session_id, separator, *sweagent_args = sys.argv[1:]
    if separator != "--":
        raise SystemExit(f"expected '--' after session_id, got {separator!r}")
    original = agents.get_agent_from_config

    def with_hook(config: agents.AgentConfig) -> agents.AbstractAgent:
        agent = original(config)
        agent.add_hook(SpanHook(Path(out_path), session_id, config.tools.execution_timeout))
        return agent

    run_batch.get_agent_from_config = with_hook
    install_glibc_patch()
    run_batch.run_from_cli(sweagent_args)


if __name__ == "__main__":
    main()
