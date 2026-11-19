"""Agent config and SpanHook checks. Run in the sweagent env: PYTHONPATH=. python -m pytest tests/test_agent_config.py"""

import json
import time
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel
from sweagent.agent.agents import DefaultAgentConfig, TemplateConfig
from sweagent.agent.models import GenericAPIModelConfig
from sweagent.run.common import BasicCLI
from sweagent.run.run_batch import RunBatchConfig
from sweagent.tools.parsing import ThoughtActionParser
from sweagent.tools.tools import ToolConfig, ToolFilterConfig
from sweagent.types import StepOutput
from swerex.deployment.docker import DockerDeployment

from agent_entry import (
    BUILDER_OLD,
    LIBC6_STEP,
    PYTHON_LINK,
    SWEREX_LINK,
    TIMEOUT_MARKER,
    SpanHook,
    install_glibc_patch,
    patch_glibc_dockerfile,
)
from run import AgentConfig, resolve_config
from tool_span import ToolSpan

CONFIG_PATH = Path(__file__).resolve().parents[1] / "agent_config.yaml"
TOOL_TIMEOUT_S = 12
MAX_MODEL_LEN = 30000
INSTANCE_ID = "owner__repo.abcd1234.func_pm_remove_assign__xyz"


def base_config() -> dict[str, object]:
    return yaml.safe_load(CONFIG_PATH.read_text())


def resolved_dict(base: dict[str, object]) -> dict[str, object]:
    return resolve_config(AgentConfig.model_validate(base), TOOL_TIMEOUT_S, MAX_MODEL_LEN).model_dump(exclude_unset=True)


def write_instances(tmp_path: Path) -> Path:
    row = {
        "instance_id": INSTANCE_ID,
        "image_name": "swebench/swesmith.x86_64.owner_1776_repo.abcd1234",
        "problem_statement": "bug",
        "repo_name": "testbed",
        "base_commit": INSTANCE_ID,
    }
    path = tmp_path / "instances.jsonl"
    path.write_text(json.dumps(row) + "\n")
    return path


def load_run_config(tmp_path: Path, resolved: dict[str, object]) -> RunBatchConfig:
    config_path = tmp_path / "agent_config.resolved.yaml"
    config_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    args = [
        "--config", str(config_path),
        "--instances.type", "file",
        "--instances.path", str(write_instances(tmp_path)),
        "--output_dir", str(tmp_path / "out"),
        "--agent.model.api_base", "http://127.0.0.1:9/s/x/v1",
        "--num_workers", "1",
    ]
    return BasicCLI(RunBatchConfig, help_text="").get_config(args)


def test_config_loads_via_sweagent_models(tmp_path: Path) -> None:
    resolved = resolved_dict(base_config())
    config = load_run_config(tmp_path, resolved)
    assert isinstance(config.agent, DefaultAgentConfig)
    assert config.agent.model.name == "openai/Qwen/Qwen3-14B-AWQ"
    assert config.agent.model.completion_kwargs["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert config.agent.tools.parse_function.type == "thought_action"


def test_base_config_execution_timeout_in_range() -> None:
    timeout = base_config()["agent"]["tools"]["execution_timeout"]
    assert 10 <= timeout <= 15
    assert base_config()["agent"]["tools"]["env_variables"]["TOOL_TIMEOUT_S"] == timeout


def section_keys(section: dict[str, object], schema: type[BaseModel]) -> set[str]:
    return set(section) - set(schema.model_fields)


def unknown_keys(config: dict[str, object]) -> dict[str, set[str]]:
    """SWE-agent ignores unknown keys in agent/templates/tools (only the model section forbids them), so check here."""
    agent = config["agent"]
    tools = agent["tools"]
    found = {
        "agent": section_keys(agent, DefaultAgentConfig),
        "agent.templates": section_keys(agent["templates"], TemplateConfig),
        "agent.tools": section_keys(tools, ToolConfig),
        "agent.tools.filter": section_keys(tools["filter"], ToolFilterConfig),
        "agent.tools.parse_function": section_keys(tools["parse_function"], ThoughtActionParser),
        "agent.model": section_keys(agent["model"], GenericAPIModelConfig),
    }
    return {name: keys for name, keys in found.items() if keys}


def test_no_key_outside_schema() -> None:
    assert unknown_keys(base_config()) == {}
    assert unknown_keys(resolved_dict(base_config())) == {}


def test_unknown_key_detector_flags_typos() -> None:
    config = base_config()
    config["agent"]["tools"]["execution_timeuot"] = 1
    assert unknown_keys(config) == {"agent.tools": {"execution_timeuot"}}


def test_resolve_config_sets_timeout_and_input_cap() -> None:
    base = base_config()
    resolved = resolved_dict(base)
    tools = resolved["agent"]["tools"]
    model = resolved["agent"]["model"]
    assert tools["execution_timeout"] == TOOL_TIMEOUT_S
    assert tools["env_variables"]["TOOL_TIMEOUT_S"] == TOOL_TIMEOUT_S
    assert model["max_input_tokens"] == MAX_MODEL_LEN - model["completion_kwargs"]["max_tokens"]
    assert "max_input_tokens" not in base["agent"]["model"]


def make_hook(tmp_path: Path) -> tuple[SpanHook, Path]:
    out_path = tmp_path / "spans.jsonl"
    return SpanHook(out_path, "s1", TOOL_TIMEOUT_S), out_path


def read_span(out_path: Path) -> ToolSpan:
    return ToolSpan.model_validate_json(out_path.read_text().splitlines()[-1])


def test_span_hook_cancelled_step_ends_at_configured_timeout(tmp_path: Path) -> None:
    hook, out_path = make_hook(tmp_path)
    step = StepOutput(action="make -j4", observation=f"cmd {TIMEOUT_MARKER} 12 seconds", execution_time=TOOL_TIMEOUT_S + 7.5)
    hook.on_step_start()
    hook.on_action_started(step=step)
    hook.on_action_executed(step=step)
    hook.on_step_done(step=step, info={})
    span = read_span(out_path)
    assert span.timed_out and span.executed and span.tool == "make"
    assert span.t_tool_start is not None and span.t_tool_end is not None
    assert span.t_tool_end == span.t_tool_start + TOOL_TIMEOUT_S
    assert span.execution_time_s == span.t_tool_end - span.t_tool_start


def test_span_hook_final_timeout_step_ends_at_cancellation_not_autosubmit(tmp_path: Path) -> None:
    hook, out_path = make_hook(tmp_path)
    hook.on_step_start()
    hook.on_action_started(step=StepOutput(action="make -j4 all"))
    time.sleep(0.05)
    final = StepOutput(
        action="", observation="Exit due to multiple consecutive command timeouts",
        exit_status="exit_command_timeout", execution_time=0.0,
    )
    hook.on_step_done(step=final, info={})
    span = read_span(out_path)
    assert span.timed_out and span.executed
    assert span.tool == "make" and span.action_head == "make -j4 all"
    assert span.t_tool_start is not None and span.t_tool_end is not None
    assert span.t_tool_end == span.t_tool_start + TOOL_TIMEOUT_S
    assert span.execution_time_s == span.t_tool_end - span.t_tool_start


def test_span_hook_non_timeout_exception_ends_now_and_is_not_timed_out(tmp_path: Path) -> None:
    hook, out_path = make_hook(tmp_path)
    hook.on_step_start()
    hook.on_action_started(step=StepOutput(action="cat big.txt"))
    time.sleep(0.05)
    final = StepOutput(action="", observation="Exit due to runtime error", exit_status="exit_environment_error")
    before = time.time()
    hook.on_step_done(step=final, info={})
    span = read_span(out_path)
    assert span.executed and not span.timed_out
    assert span.t_tool_start is not None and span.t_tool_end is not None
    assert before <= span.t_tool_end <= time.time()
    assert 0.05 <= span.execution_time_s < TOOL_TIMEOUT_S
    assert span.execution_time_s == span.t_tool_end - span.t_tool_start


def test_span_hook_requery_does_not_leave_tool_end_before_start(tmp_path: Path) -> None:
    hook, out_path = make_hook(tmp_path)
    first = StepOutput(action="ls")
    hook.on_step_start()
    hook.on_action_started(step=first)
    hook.on_action_executed(step=first)
    time.sleep(0.02)
    hook.on_action_started(step=StepOutput(action="cat x"))
    time.sleep(0.02)
    hook.on_step_done(step=StepOutput(action="", observation="", exit_status="exit_error"), info={})
    span = read_span(out_path)
    assert span.t_tool_start is not None and span.t_tool_end is not None
    assert span.t_tool_end >= span.t_tool_start
    assert span.tool == "cat"


def test_span_hook_normal_step_uses_measured_duration(tmp_path: Path) -> None:
    hook, out_path = make_hook(tmp_path)
    step = StepOutput(action="ls /testbed", observation="a b", execution_time=0.01)
    hook.on_step_start()
    hook.on_action_started(step=step)
    time.sleep(0.02)
    hook.on_action_executed(step=step)
    hook.on_step_done(step=step, info={})
    span = read_span(out_path)
    assert not span.timed_out and span.executed
    assert span.t_tool_start is not None and span.t_tool_end is not None
    assert span.execution_time_s == span.t_tool_end - span.t_tool_start


def test_span_hook_step_without_tool_is_not_executed(tmp_path: Path) -> None:
    hook, out_path = make_hook(tmp_path)
    hook.on_step_start()
    hook.on_step_done(step=StepOutput(action="", observation="", exit_status="exit_context"), info={})
    span = read_span(out_path)
    assert not span.executed and not span.timed_out
    assert span.t_tool_start is None and span.t_tool_end is None
    assert span.execution_time_s == 0.0


def original_dockerfile() -> str:
    return DockerDeployment(image="x", python_standalone_dir="/root").glibc_dockerfile


def test_patch_glibc_dockerfile_builds_on_ubuntu_2004_without_libc6_step() -> None:
    original = original_dockerfile()
    patched = patch_glibc_dockerfile(original)
    assert "FROM  ubuntu:20.04 AS builder\nENV DEBIAN_FRONTEND=noninteractive\nRUN apt-get update" in patched
    assert "python:3.11.9-slim-bookworm" not in patched
    assert "    wget \\\n    ca-certificates \\\n    gcc \\\n" in patched
    for package in ("wget", "gcc", "make", "zlib1g-dev", "libssl-dev"):
        assert f"    {package} \\\n" in patched
    assert "libc6" not in patched
    assert patched.count(PYTHON_LINK) == 1
    assert SWEREX_LINK + PYTHON_LINK in patched
    assert "/usr/local/bin/python3" not in patched
    assert LIBC6_STEP in original
    tail = original.split(LIBC6_STEP)[1]
    assert patched.endswith("FROM  $BASE_IMAGE\n" + tail.replace(SWEREX_LINK, SWEREX_LINK + PYTHON_LINK))
    assert "COPY --from=builder /root/python3.11 /root/python3.11\n" in patched
    assert "RUN /root/python3.11/bin/pip3 install --no-cache-dir swe-rex\n" in patched
    build = "WORKDIR /build\n" + original.split("WORKDIR /build\n", 1)[1].split("FROM  $BASE_IMAGE")[0]
    assert build in patched and "make -j$(nproc)" in build


def test_patch_glibc_dockerfile_rejects_missing_builder_block() -> None:
    with pytest.raises(ValueError, match="builder stage"):
        patch_glibc_dockerfile(original_dockerfile().replace(BUILDER_OLD, "FROM  ubuntu:22.04 AS builder\nRUN apt-get update"))


def test_patch_glibc_dockerfile_rejects_missing_wget_block() -> None:
    with pytest.raises(ValueError, match="wget"):
        patch_glibc_dockerfile(original_dockerfile().replace("    wget \\\n", ""))


def test_patch_glibc_dockerfile_rejects_missing_libc6_block() -> None:
    with pytest.raises(ValueError, match="libc6"):
        patch_glibc_dockerfile(original_dockerfile().replace(LIBC6_STEP, ""))


def test_patch_glibc_dockerfile_rejects_missing_swerex_link_block() -> None:
    with pytest.raises(ValueError, match="swerex-remote symlink"):
        patch_glibc_dockerfile(original_dockerfile().replace(SWEREX_LINK, ""))


def test_install_glibc_patch_overrides_property(monkeypatch: pytest.MonkeyPatch) -> None:
    original = original_dockerfile()
    monkeypatch.setattr(DockerDeployment, "glibc_dockerfile", DockerDeployment.glibc_dockerfile)
    install_glibc_patch()
    assert original_dockerfile() == patch_glibc_dockerfile(original)
