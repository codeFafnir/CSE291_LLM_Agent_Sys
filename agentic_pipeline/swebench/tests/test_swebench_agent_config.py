"""SWE-bench agent config checks. Run in the sweagent env: PYTHONPATH=../swesmith:. python -m pytest tests/test_swebench_agent_config.py"""

import json
from pathlib import Path

import yaml
from pydantic import BaseModel
from sweagent.agent.agents import DefaultAgentConfig, TemplateConfig
from sweagent.agent.models import GenericAPIModelConfig
from sweagent.run.common import BasicCLI
from sweagent.run.run_batch import RunBatchConfig
from sweagent.tools.parsing import ThoughtActionParser
from sweagent.tools.tools import ToolConfig, ToolFilterConfig

from run import AgentConfig, resolve_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "agent_config.yaml"
TOOL_TIMEOUT_S = 12
MAX_MODEL_LEN = 30000
INSTANCE_ID = "astropy__astropy-12907"
BASE_COMMIT = "d16bfe05a744909de4b27f5875fe0d4ed41ce607"


def base_config() -> dict[str, object]:
    return yaml.safe_load(CONFIG_PATH.read_text())


def resolved_dict(base: dict[str, object]) -> dict[str, object]:
    return resolve_config(AgentConfig.model_validate(base), TOOL_TIMEOUT_S, MAX_MODEL_LEN).model_dump(exclude_unset=True)


def section_keys(section: dict[str, object], schema: type[BaseModel]) -> set[str]:
    return set(section) - set(schema.model_fields)


def test_no_key_outside_schema() -> None:
    agent = base_config()["agent"]
    tools = agent["tools"]
    found = {
        "agent": section_keys(agent, DefaultAgentConfig),
        "agent.templates": section_keys(agent["templates"], TemplateConfig),
        "agent.tools": section_keys(tools, ToolConfig),
        "agent.tools.filter": section_keys(tools["filter"], ToolFilterConfig),
        "agent.tools.parse_function": section_keys(tools["parse_function"], ThoughtActionParser),
        "agent.model": section_keys(agent["model"], GenericAPIModelConfig),
    }
    assert {name: keys for name, keys in found.items() if keys} == {}


def test_caps_and_tool_set() -> None:
    agent = base_config()["agent"]
    assert agent["templates"]["max_observation_length"] == 4000
    assert agent["model"]["per_instance_call_limit"] == 12
    assert agent["model"]["completion_kwargs"]["max_tokens"] == 2048
    assert "history_processors" not in agent
    assert [b["path"] for b in agent["tools"]["bundles"]] == [
        "tools/registry", "tools/edit_anthropic", "tools/search", "tools/submit",
    ]
    assert agent["tools"]["enable_bash_tool"] is True
    assert 10 <= agent["tools"]["execution_timeout"] <= 15


def test_resolved_config_loads_a_verified_instance(tmp_path: Path) -> None:
    config_path = tmp_path / "agent_config.resolved.yaml"
    config_path.write_text(yaml.safe_dump(resolved_dict(base_config()), sort_keys=False))
    row = {
        "instance_id": INSTANCE_ID,
        "image_name": f"ghcr.io/epoch-research/swe-bench.eval.x86_64.{INSTANCE_ID}:latest",
        "problem_statement": "bug",
        "repo_name": "testbed",
        "base_commit": BASE_COMMIT,
    }
    instances = tmp_path / "instances.jsonl"
    instances.write_text(json.dumps(row) + "\n")
    args = [
        "--config", str(config_path),
        "--instances.type", "file",
        "--instances.path", str(instances),
        "--output_dir", str(tmp_path / "out"),
        "--agent.model.api_base", "http://127.0.0.1:9/s/x/v1",
        "--num_workers", "1",
    ]
    config = BasicCLI(RunBatchConfig, help_text="").get_config(args)
    assert isinstance(config.agent, DefaultAgentConfig)
    assert config.agent.templates.max_observation_length == 4000
    assert config.agent.model.per_instance_call_limit == 12
    assert config.agent.model.max_input_tokens == MAX_MODEL_LEN - 2048
    assert config.agent.tools.execution_timeout == TOOL_TIMEOUT_S
