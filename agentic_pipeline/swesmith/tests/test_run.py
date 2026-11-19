"""Tests for run.py: config rules, failure accounting, and an end-to-end dry run with fake agent + upstream."""

import asyncio
import json
import os
import socket
import sys
import threading
import time
from datetime import datetime
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from aiohttp import web
from pydantic import BaseModel, ValidationError

import run
from run import AgentConfig, ContainerInfo, InstanceRow, RunArgs, Session
from session_proxy import RequestLog

SWESMITH_DIR = Path(run.__file__).resolve().parent
OVERFLOW_ID = "overflow__repo.aaaa1111"
CRASH_ID = "crash__repo.bbbb2222"
HANG_ID = "hang__repo.cccc3333"
NOTRAJ_IDS = ["notraj__repo.1111aaaa", "notraj__repo.2222bbbb"]
STUBBORN_ID = "stubborn__repo.ffff6666"
IMAGE = "swebench/swesmith.x86_64.cisco_1776_openh264.cf568c83"
SANITIZED = "swebenchswesmith.x86_64.cisco_1776_openh264.cf568c83"
UUID_A = "123e4567-e89b-42d3-a456-426614174000"
OK_IDS = ["ok1__repo.dddd4444", "ok2__repo.eeee5555"]
OVERFLOW_TEXT = "maximum context length exceeded " * 20

BASE_CONFIG = AgentConfig.model_validate({
    "agent": {
        "tools": {"execution_timeout": 15, "env_variables": {"TOOL_TIMEOUT_S": 15}},
        "model": {"completion_kwargs": {"max_tokens": 2048}},
    }
})
REAL_CONFIG_PATH = SWESMITH_DIR / "agent_config.yaml"

FAKE_AGENT = """
import json, signal, subprocess, sys, time, urllib.error, urllib.request
sys.path.insert(0, {swesmith_dir!r})
from pathlib import Path
from tool_span import ToolSpan

spans_path, instance_id, _, *rest = sys.argv[1:]
api_base = rest[rest.index("--agent.model.api_base") + 1]
out_dir = Path(rest[rest.index("--output_dir") + 1])
if instance_id.startswith("notraj"):
    out_dir.mkdir(parents=True, exist_ok=True)
    statuses = {{"instances_by_exit_status": {{"Uncaught CalledProcessError": [instance_id]}}, "total_cost": 0}}
    (out_dir / "run_batch_exit_statuses.yaml").write_text(json.dumps(statuses))
    (out_dir / "run_batch.log").write_text(
        "Traceback (most recent call last):\\n  File x\\nsubprocess.CalledProcessError: Command 'docker build' failed\\n")
    sys.exit(0)
if instance_id.startswith(("hang", "stubborn")):
    if instance_id.startswith("stubborn"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # inherited by the grandchild too
    grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    Path(spans_path).with_name(instance_id + ".grandchild.pid").write_text(str(grandchild.pid))
    time.sleep(60)
for step in range(1, 4):
    t0 = time.time()
    body = json.dumps({{"max_tokens": 16, "messages": []}}).encode()
    req = urllib.request.Request(api_base + "/chat/completions", data=body, headers={{"Content-Type": "application/json"}})
    try:
        urllib.request.urlopen(req, timeout=10).read()
    except urllib.error.HTTPError:
        pass
    time.sleep(0.3)
    span = ToolSpan(session_id=instance_id, step=step, tool="ls", action_head="ls", executed=True, t_step_start=t0,
                    t_tool_start=t0, t_tool_end=time.time(), execution_time_s=0.3, timed_out=False,
                    step_exit_status=None, observation_chars=5)
    with open(spans_path, "a") as handle:
        handle.write(span.model_dump_json() + "\\n")
if instance_id.startswith("crash"):
    sys.exit(3)
traj_dir = out_dir / instance_id
traj_dir.mkdir(parents=True, exist_ok=True)
info = {{"exit_status": "submitted", "submission": "diff --git", "model_stats": {{"api_calls": 3}}}}
(traj_dir / (instance_id + ".traj")).write_text(json.dumps({{"trajectory": [], "info": info}}))
preds = {{instance_id: {{"instance_id": instance_id, "model_patch": "diff", "model_name_or_path": "fake"}}}}
(out_dir / "preds.json").write_text(json.dumps(preds))
"""


FAKE_DOCKER = """#!{python}
import json, os, sys

state_path = os.environ["FAKE_DOCKER_STATE"]
state = json.load(open(state_path))
args = sys.argv[1:]
if args[0] == "ps":
    ids = [c["id"] for c in state]
    if "--filter" in args:
        key, value = args[args.index("--filter") + 1].removeprefix("label=").split("=", 1)
        ids = [c["id"] for c in state if c["labels"].get(key) == value]
    print("\\n".join(ids))
elif args[0] == "inspect":
    missing = False
    for ref in args[3:]:
        c = next(c for c in state if c["id"] == ref)
        if c.get("ghost"):
            missing = True
            print("Error: No such object: " + ref, file=sys.stderr)
            continue
        print("/" + c["name"], str(c["running"]).lower(), c["created"])
    sys.exit(1 if missing else 0)
elif args[0] == "rm":
    refs = args[2:]
    if os.environ.get("FAKE_DOCKER_RM_FAIL") in refs:
        print("Error response from daemon: boom", file=sys.stderr)
        sys.exit(1)
    state = [c for c in state if c["id"] not in refs and c["name"] not in refs]
    json.dump(state, open(state_path, "w"))
"""


@pytest.fixture(autouse=True)
def fake_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    script = tmp_path / "docker"
    script.write_text(FAKE_DOCKER.format(python=sys.executable))
    script.chmod(0o755)
    state = tmp_path / "docker_state.json"
    state.write_text("[]")
    monkeypatch.setenv("FAKE_DOCKER_STATE", str(state))
    return state


def container(cid: str, name: str, running: bool, created: str, labels: dict[str, str]) -> dict[str, object]:
    return {"id": cid, "name": name, "running": running, "created": created, "labels": labels}


def docker_names(state: Path) -> list[str]:
    return [c["name"] for c in json.loads(state.read_text())]


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_dead(pid: int) -> bool:
    deadline = time.time() + 5
    while time.time() < deadline:
        if not pid_alive(pid):
            return True
        time.sleep(0.1)
    return False


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def fake_chat(request: web.Request) -> web.Response:
    if "overflow" in request.headers["X-Session-ID"]:
        return web.Response(text=OVERFLOW_TEXT, status=400)
    usage = {"prompt_tokens": 10, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 4}}
    return web.json_response({"choices": [], "usage": usage})


@pytest.fixture
def upstream() -> Iterator[str]:
    loop = asyncio.new_event_loop()
    app = web.Application()
    app.router.add_post("/v1/chat/completions", fake_chat)
    runner = web.AppRunner(app)
    port = free_port()

    async def start() -> None:
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", port).start()

    loop.run_until_complete(start())
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    asyncio.run_coroutine_threadsafe(runner.cleanup(), loop).result(timeout=10)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=10)
    loop.close()


def make_args(tmp_path: Path, upstream: str, ids: list[str], concurrency: int) -> RunArgs:
    instances = tmp_path / "instances.jsonl"
    rows = [
        {"instance_id": i, "image_name": "img", "problem_statement": "bug", "image_key": "srt"} for i in ids
    ]
    instances.write_text("".join(json.dumps(r) + "\n" for r in rows))
    config = tmp_path / "agent_config.yaml"
    config.write_text(yaml.safe_dump(BASE_CONFIG.model_dump()))
    entry = tmp_path / "fake_agent_entry.py"
    entry.write_text(FAKE_AGENT.format(swesmith_dir=str(SWESMITH_DIR)))
    return RunArgs(
        instances_jsonl=instances,
        count=len(ids),
        concurrency=concurrency,
        rate=1_000_000.0,
        seed=0,
        run_dir=tmp_path / "run",
        upstream=upstream,
        proxy_port=free_port(),
        max_model_len=30000,
        tool_timeout_s=12,
        agent_config=config,
        sweagent_python=Path(__import__("sys").executable),
        agent_entry=entry,
        docker=str(tmp_path / "docker"),
    )


def read_sessions(run_dir: Path) -> dict[str, Session]:
    lines = (run_dir / "sessions.jsonl").read_text().splitlines()
    sessions = [Session.model_validate_json(line) for line in lines]
    return {s.instance_id: s for s in sessions}


def max_overlap(sessions: list[Session]) -> int:
    events = [(s.t_start, 1) for s in sessions] + [(s.t_end, -1) for s in sessions]
    live = peak = 0
    for _, delta in sorted(events, key=lambda e: (e[0], e[1])):
        live += delta
        peak = max(peak, live)
    return peak


@pytest.mark.parametrize("value,ok", [(9, False), (10, True), (15, True), (16, False)])
def test_check_tool_timeout_bounds(value: int, ok: bool) -> None:
    if ok:
        run.check_tool_timeout(value)
    else:
        with pytest.raises(ValueError, match=f"tool_timeout_s={value}"):
            run.check_tool_timeout(value)


def with_timeout(config: AgentConfig, value: int) -> AgentConfig:
    tools = config.agent.tools.model_copy(update={"execution_timeout": value})
    return config.model_copy(update={"agent": config.agent.model_copy(update={"tools": tools})})


def test_resolve_config_sets_timeout_and_token_cap() -> None:
    resolved = run.resolve_config(BASE_CONFIG, 12, 30000)
    assert resolved.agent.tools.execution_timeout == 12
    assert resolved.agent.tools.env_variables["TOOL_TIMEOUT_S"] == 12
    assert resolved.agent.model.max_input_tokens == 30000 - 2048
    assert BASE_CONFIG.agent.tools.execution_timeout == 15  # input not mutated
    assert BASE_CONFIG.agent.model.max_input_tokens is None
    run.check_resolved_timeout(resolved, 12)


def test_check_resolved_timeout_rejects_mismatch_and_over_15() -> None:
    resolved = run.resolve_config(BASE_CONFIG, 12, 30000)
    with pytest.raises(ValueError, match="resolved execution_timeout=12"):
        run.check_resolved_timeout(resolved, 13)
    with pytest.raises(ValueError, match="resolved execution_timeout=20"):
        run.check_resolved_timeout(with_timeout(resolved, 20), 20)


def legacy_resolve_config(base: object, tool_timeout_s: int, max_model_len: int) -> object:
    """Pre-refactor dict logic, kept verbatim as the equivalence oracle."""
    resolved = json.loads(json.dumps(base))
    agent = resolved["agent"]
    tools = agent["tools"]
    model = agent["model"]
    tools["execution_timeout"] = tool_timeout_s
    tools["env_variables"]["TOOL_TIMEOUT_S"] = tool_timeout_s
    model["max_input_tokens"] = max_model_len - model["completion_kwargs"]["max_tokens"]
    return resolved


@pytest.mark.parametrize("tool_timeout_s,max_model_len", [(10, 30000), (12, 30000), (15, 32768)])
def test_resolve_config_matches_legacy_dict_logic_on_real_config(tool_timeout_s: int, max_model_len: int) -> None:
    raw = yaml.safe_load(REAL_CONFIG_PATH.read_text())
    resolved = run.resolve_config(AgentConfig.model_validate(raw), tool_timeout_s, max_model_len)
    dumped = resolved.model_dump(exclude_unset=True)
    expected = legacy_resolve_config(raw, tool_timeout_s, max_model_len)
    assert dumped == expected
    assert yaml.safe_load(yaml.safe_dump(dumped, sort_keys=False)) == expected  # what run() writes to disk
    assert AgentConfig.model_validate(raw).model_dump(exclude_unset=True) == raw  # unknown keys survive untouched


def test_resolve_config_rejects_config_missing_read_fields() -> None:
    base = BASE_CONFIG.model_dump()
    no_max_tokens = {"agent": {**base["agent"], "model": {"completion_kwargs": {}}}}
    with pytest.raises(ValueError, match="max_tokens"):
        AgentConfig.model_validate(no_max_tokens)
    no_env = {"agent": {**base["agent"], "tools": {"execution_timeout": 15}}}
    with pytest.raises(ValueError, match="env_variables"):
        AgentConfig.model_validate(no_env)
    bad_timeout = {"agent": {**base["agent"], "tools": {"execution_timeout": "soon", "env_variables": {}}}}
    with pytest.raises(ValueError, match="execution_timeout"):
        AgentConfig.model_validate(bad_timeout)
    with pytest.raises(ValueError, match="agent"):
        AgentConfig.model_validate({})


def test_agent_instance_maps_fields() -> None:
    row = InstanceRow(instance_id="a__b.1234", image_name="img:tag", problem_statement="p", image_key="srt")
    assert run.agent_instance(row).model_dump() == {
        "instance_id": "a__b.1234",
        "image_name": "img:tag",
        "problem_statement": "p",
        "repo_name": "testbed",
        "base_commit": "a__b.1234",
    }


def test_agent_instance_uses_row_base_commit_when_present() -> None:
    row = InstanceRow(instance_id="a__b-1", image_name="img", problem_statement="p", image_key="verified", base_commit="f" * 40)
    assert run.agent_instance(row).base_commit == "f" * 40


def test_read_jsonl_validates_rows_and_names_missing_field(tmp_path: Path) -> None:
    path = tmp_path / "instances.jsonl"
    good = {"instance_id": "a", "image_name": "img", "problem_statement": "p", "image_key": "k", "extra": 1}
    path.write_text(json.dumps(good) + "\n\n")
    assert run.read_jsonl(path, InstanceRow) == [InstanceRow(instance_id="a", image_name="img", problem_statement="p", image_key="k")]
    path.write_text(json.dumps({"instance_id": "a"}) + "\n")
    with pytest.raises(ValueError, match="image_name"):
        run.read_jsonl(path, InstanceRow)


def request_log(session_id: str, turn: int, status: int, error: str | None) -> str:
    return RequestLog(
        session_id=session_id, turn=turn, request_id=f"{session_id}-t{turn:03d}", path="/v1/chat/completions",
        t_send=1.0, t_first_byte=1.1, t_end=1.2, status=status, error=error, max_tokens=16, request_bytes=10,
        prompt_tokens=None, completion_tokens=None, cached_tokens=None,
    ).model_dump_json()


def test_proxy_failures(tmp_path: Path) -> None:
    assert run.proxy_failures(tmp_path, "s1") == (0, None)  # no log file yet
    lines = [
        request_log("s1", 1, 200, None),
        request_log("s1", 2, 400, "x" * 300),
        request_log("s1", 3, 502, "second"),
        request_log("s2", 1, 500, "other session"),
    ]
    (tmp_path / "proxy_requests.jsonl").write_text("\n".join(lines) + "\n")
    count, excerpt = run.proxy_failures(tmp_path, "s1")
    assert count == 2
    assert excerpt == "x" * 200
    assert run.proxy_failures(tmp_path, "s3") == (0, None)


def test_sweagent_command_uses_agent_entry(tmp_path: Path) -> None:
    args = make_args(tmp_path, "http://127.0.0.1:1", ["a"], 1)
    cmd = run.sweagent_command(args, tmp_path / "cfg.yaml", tmp_path / "i.jsonl", "a")
    assert cmd[:2] == [str(args.sweagent_python), str(args.agent_entry)]
    assert cmd[3] == "a" and cmd[4] == "--"


def test_end_to_end_dry_run(tmp_path: Path, upstream: str) -> None:
    ids = [OK_IDS[0], CRASH_ID, OVERFLOW_ID, OK_IDS[1]]
    args = make_args(tmp_path, upstream, ids, 2)
    run.run(args)
    out = args.run_dir
    for name in [
        "sessions.jsonl", "sessions.csv", "tool_spans.jsonl", "tool_spans.csv",
        "proxy_requests.jsonl", "preds.json", "agent_config.resolved.yaml",
    ]:
        assert (out / name).exists(), name
    sessions = read_sessions(out)
    assert set(sessions) == set(ids)
    assert sessions[CRASH_ID].status == "crashed" and sessions[CRASH_ID].returncode == 3
    assert sessions[CRASH_ID].exit_status is None and not sessions[CRASH_ID].submitted
    for done in [OK_IDS[0], OK_IDS[1], OVERFLOW_ID]:
        assert sessions[done].status == "completed"
        assert sessions[done].exit_status == "submitted" and sessions[done].api_calls == 3
    assert sessions[OVERFLOW_ID].failed_requests == 3
    assert sessions[OVERFLOW_ID].error_excerpt == OVERFLOW_TEXT[:200]
    assert sessions[OK_IDS[0]].failed_requests == 0 and sessions[OK_IDS[0]].error_excerpt is None
    assert 1 < max_overlap(list(sessions.values())) <= 2
    preds = json.loads((out / "preds.json").read_text())
    assert set(preds) == {OK_IDS[0], OK_IDS[1], OVERFLOW_ID}
    csv_header = (out / "sessions.csv").read_text().splitlines()[0].split(",")
    assert csv_header == list(Session.model_fields)
    spans = (out / "tool_spans.jsonl").read_text().splitlines()
    assert len(spans) == 3 * len(ids)
    proxied = [RequestLog.model_validate_json(line) for line in (out / "proxy_requests.jsonl").read_text().splitlines()]
    assert len(proxied) == 3 * len(ids)
    resolved = yaml.safe_load((out / "agent_config.resolved.yaml").read_text())
    assert resolved["agent"]["tools"]["execution_timeout"] == 12
    assert resolved["agent"]["model"]["max_input_tokens"] == 30000 - 2048


def test_wall_timeout_does_not_abort_cell(tmp_path: Path, upstream: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run, "SESSION_WALL_TIMEOUT_S", 4)
    args = make_args(tmp_path, upstream, [HANG_ID, OK_IDS[0]], 1)
    run.run(args)
    sessions = read_sessions(args.run_dir)
    assert sessions[HANG_ID].status == "wall_timeout"
    assert sessions[OK_IDS[0]].status == "completed"
    assert set(json.loads((args.run_dir / "preds.json").read_text())) == {OK_IDS[0]}


def test_is_swerex_container_matches_only_name_pattern() -> None:
    images = [IMAGE]
    assert run.sanitized_image(IMAGE) == SANITIZED
    assert run.is_swerex_container(f"{SANITIZED}-{UUID_A}", images)
    assert not run.is_swerex_container(f"{SANITIZED}-{UUID_A}-x", images)
    assert not run.is_swerex_container(f"{SANITIZED}-notauuid", images)
    assert not run.is_swerex_container(f"{SANITIZED}{UUID_A}", images)
    assert not run.is_swerex_container(f"other-{SANITIZED}-{UUID_A}", images)
    assert not run.is_swerex_container(f"postgres-{UUID_A}", images)
    assert not run.is_swerex_container(f"{SANITIZED}-{UUID_A.upper()}", images)
    assert not run.is_swerex_container(f"{SANITIZED}-{UUID_A}", [])


def test_swerex_containers_filters_pure() -> None:
    mine = ContainerInfo(name=f"{SANITIZED}-{UUID_A}", running=True, created=1.0)
    other = ContainerInfo(name="mydb", running=True, created=1.0)
    assert run.swerex_containers([other, mine], [IMAGE]) == [mine]


def test_sweagent_command_labels_container_by_session(tmp_path: Path) -> None:
    args = make_args(tmp_path, "http://127.0.0.1:1", ["a"], 1)
    cmd = run.sweagent_command(args, tmp_path / "cfg.yaml", tmp_path / "i.jsonl", "a")
    flag = cmd.index("--instances.deployment.docker_args")
    assert json.loads(cmd[flag + 1]) == ["--label", "cse291.session=a"]


def test_cleanup_only_removes_matching_containers_created_since_window(tmp_path: Path, fake_docker: Path) -> None:
    name = lambda n: f"{SANITIZED}-123e4567-e89b-42d3-a456-42661417400{n}"  # noqa: E731
    since = datetime.fromisoformat("2026-11-17T20:00:00+00:00").timestamp()
    fake_docker.write_text(json.dumps([
        container("c1", name(1), True, "2026-11-17T19:00:00.123456789Z", {}),
        container("c2", name(2), True, "2026-11-17T20:00:05.5Z", {}),
        container("c3", name(3), False, "2026-11-17T20:01:00Z", {}),
        container("c4", "mydb", True, "2026-11-17T20:02:00Z", {}),
        container("c5", f"other-{UUID_A}", True, "2026-11-17T20:02:00Z", {}),
    ]))
    docker = str(tmp_path / "docker")
    assert run.running_swerex_names(docker, [IMAGE]) == [name(1), name(2)]
    assert run.cleanup_swerex_containers(docker, [IMAGE], since) == run.CleanupResult(removed=2, failures=[])
    assert docker_names(fake_docker) == [name(1), "mydb", f"other-{UUID_A}"]
    assert run.cleanup_swerex_containers(docker, [IMAGE], since).removed == 0


def test_remove_session_containers_uses_label(tmp_path: Path, fake_docker: Path) -> None:
    fake_docker.write_text(json.dumps([
        container("c1", "x", True, "2026-11-17T20:00:00Z", {"cse291.session": "s1"}),
        container("c2", "y", True, "2026-11-17T20:00:00Z", {"cse291.session": "s2"}),
        container("c3", "z", True, "2026-11-17T20:00:00Z", {}),
    ]))
    assert run.remove_session_containers(str(tmp_path / "docker"), "s1") == run.CleanupResult(removed=1, failures=[])
    assert docker_names(fake_docker) == ["y", "z"]


@pytest.mark.parametrize("session_id,grace_s", [(HANG_ID, 5), (STUBBORN_ID, 1)])
def test_wall_timeout_kills_process_group_and_session_container(
    tmp_path: Path, upstream: str, fake_docker: Path, monkeypatch: pytest.MonkeyPatch, session_id: str, grace_s: int
) -> None:
    monkeypatch.setattr(run, "SESSION_WALL_TIMEOUT_S", 4)
    monkeypatch.setattr(run, "TERM_GRACE_S", grace_s)
    fake_docker.write_text(json.dumps([
        container("c1", "leaked", True, "2026-11-17T20:00:00Z", {"cse291.session": session_id}),
        container("c2", "fine", True, "2026-11-17T20:00:00Z", {"cse291.session": OK_IDS[0]}),
    ]))
    args = make_args(tmp_path, upstream, [session_id, OK_IDS[0]], 1)
    run.run(args)
    pid = int((args.run_dir / f"{session_id}.grandchild.pid").read_text())
    assert wait_dead(pid), "grandchild survived the group kill"
    assert read_sessions(args.run_dir)[session_id].status == "wall_timeout"
    assert docker_names(fake_docker) == ["fine"]


class Row(BaseModel):
    a: int


def test_parse_jsonl_tolerating_torn_tail() -> None:
    good = '{"a": 1}\n{"a": 2}\n'
    assert run.parse_jsonl_tolerating_torn_tail(good, Row) == [Row(a=1), Row(a=2)]
    assert run.parse_jsonl_tolerating_torn_tail(good + '{"a": 3, "b": "tru', Row) == [Row(a=1), Row(a=2)]
    assert run.parse_jsonl_tolerating_torn_tail("", Row) == []
    with pytest.raises(ValueError, match="json_invalid"):
        run.parse_jsonl_tolerating_torn_tail('{"a": 1\n{"a": 2}\n', Row)  # a bad non-final line is still an error
    with pytest.raises(ValidationError, match="Field required"):
        run.parse_jsonl_tolerating_torn_tail(good + '{"b": 3}', Row)  # complete but wrong-shaped tail is not a torn write


def test_proxy_failures_ignores_torn_last_line(tmp_path: Path) -> None:
    lines = request_log("s1", 1, 400, "bad") + "\n" + request_log("s1", 2, 500, "worse")[:40]
    (tmp_path / "proxy_requests.jsonl").write_text(lines)
    assert run.proxy_failures(tmp_path, "s1") == (1, "bad")


def test_parse_inspect_lines() -> None:
    infos = run.parse_inspect_lines(["/a true 2026-11-17T20:00:05.123456789Z", "/b false 2026-11-17T20:00:00Z"])
    assert [(i.name, i.running) for i in infos] == [("a", True), ("b", False)]
    assert infos[0].created - infos[1].created == pytest.approx(5.123456)


def test_cleanup_tolerates_vanished_container_and_rm_failure(
    tmp_path: Path, fake_docker: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = lambda n: f"{SANITIZED}-123e4567-e89b-42d3-a456-42661417400{n}"  # noqa: E731
    fake_docker.write_text(json.dumps([
        {**container("c1", name(1), True, "2099-01-01T00:00:00Z", {}), "ghost": True},  # gone between ps and inspect
        container("c2", name(2), True, "2099-01-01T00:00:00Z", {}),
        container("c3", name(3), True, "2099-01-01T00:00:00Z", {}),
    ]))
    monkeypatch.setenv("FAKE_DOCKER_RM_FAIL", name(3))
    result = run.cleanup_swerex_containers(str(tmp_path / "docker"), [IMAGE], 0.0)
    assert result.removed == 1
    assert len(result.failures) == 2
    assert "No such object" in result.failures[0] and "boom" in result.failures[1]
    assert all("\n" not in f for f in result.failures)
    log = tmp_path / "container_cleanup.log"
    run.append_cleanup_log(log, result.failures)
    run.append_cleanup_log(log, [])
    assert log.read_text().splitlines() == result.failures


def test_missing_docker_binary_is_tolerated(tmp_path: Path) -> None:
    missing = str(tmp_path / "no-such-docker")
    result = run.remove_session_containers(missing, "s1")
    assert result.removed == 0 and len(result.failures) == 1 and "FileNotFoundError" in result.failures[0]
    assert run.cleanup_swerex_containers(missing, [IMAGE], 0.0).removed == 0


def test_docker_failure_does_not_abort_cell(tmp_path: Path, upstream: str) -> None:
    args = make_args(tmp_path, upstream, [CRASH_ID, OK_IDS[0]], 1)
    run.run(args.model_copy(update={"docker": str(tmp_path / "no-such-docker")}))
    sessions = read_sessions(args.run_dir)
    assert sessions[CRASH_ID].status == "crashed" and sessions[OK_IDS[0]].status == "completed"
    log = (args.run_dir / "container_cleanup.log").read_text().splitlines()
    assert len(log) == 1 and "FileNotFoundError" in log[0]


@pytest.mark.parametrize(
    "returncode,traj_exists,timed_out,expected",
    [
        (0, True, False, "completed"),
        (0, False, False, "no_trajectory"),
        (3, False, False, "crashed"),
        (3, True, False, "crashed"),
        (0, True, True, "wall_timeout"),
        (-9, False, True, "wall_timeout"),
    ],
)
def test_classify_session(returncode: int, traj_exists: bool, timed_out: bool, expected: str) -> None:
    assert run.classify_session(returncode, traj_exists, timed_out) == expected


STATUSES_YAML = """instances_by_exit_status:
    Uncaught CalledProcessError:
    - a__b.1
    submitted:
    - c__d.2
total_cost: 0
"""
TRACE = (
    "Traceback (most recent call last):\n"
    "subprocess.CalledProcessError: Command 'first' failed\n"
    "noise CalledProcessError: not at line start\n"
    "subprocess.CalledProcessError: Command 'docker build -t x .' returned non-zero exit status 1.\n"
)


def test_uncaught_error_text() -> None:
    text = run.uncaught_error_text(STATUSES_YAML, TRACE, "a__b.1")
    assert text == "Uncaught CalledProcessError: Command 'docker build -t x .' returned non-zero exit status 1."
    assert run.uncaught_error_text(STATUSES_YAML, "", "a__b.1") == "Uncaught CalledProcessError"
    assert run.uncaught_error_text(STATUSES_YAML, TRACE, "c__d.2") is None  # normal exit status, not an error
    assert run.uncaught_error_text(STATUSES_YAML, TRACE, "other") is None
    assert run.uncaught_error_text("", TRACE, "a__b.1") is None
    long_log = "CalledProcessError: " + "x" * 500 + "\n"
    assert len(run.uncaught_error_text(STATUSES_YAML, long_log, "a__b.1") or "") == 200


def session_with(status: str) -> Session:
    return Session(
        instance_id="i", image_key="srt", arrival_offset_s=0.0, t_arrival=0.0, t_start=0.0, t_end=1.0, returncode=0,
        status=status, exit_status=None, api_calls=None, submitted=False, failed_requests=0, error_excerpt=None,
    )


def test_check_any_trajectory() -> None:
    run.check_any_trajectory([])
    run.check_any_trajectory([session_with("no_trajectory"), session_with("completed")])
    with pytest.raises(RuntimeError, match=r"all 3 sessions ended without completing.*no_trajectory.*2.*crashed.*1"):
        run.check_any_trajectory([session_with("no_trajectory")] * 2 + [session_with("crashed")])


def test_no_trajectory_session_is_classified_and_cell_continues(tmp_path: Path, upstream: str) -> None:
    args = make_args(tmp_path, upstream, [NOTRAJ_IDS[0], OK_IDS[0]], 1)
    run.run(args)
    sessions = read_sessions(args.run_dir)
    failed = sessions[NOTRAJ_IDS[0]]
    assert failed.status == "no_trajectory" and failed.returncode == 0 and failed.exit_status is None
    assert failed.failed_requests == 0
    assert failed.error_excerpt == "Uncaught CalledProcessError: Command 'docker build' failed"
    assert sessions[OK_IDS[0]].status == "completed" and sessions[OK_IDS[0]].error_excerpt is None


def test_all_sessions_without_trajectory_exit_nonzero_after_writing_outputs(tmp_path: Path, upstream: str) -> None:
    args = make_args(tmp_path, upstream, NOTRAJ_IDS, 2)
    with pytest.raises(RuntimeError, match=r"all 2 sessions ended without completing.*'no_trajectory': 2"):
        run.run(args)
    out = args.run_dir
    assert (out / "proxy_requests.jsonl").read_text() == ""
    assert (out / "tool_spans.jsonl").exists() and (out / "preds.json").exists()
    assert (out / "sessions.csv").exists()
    assert {s.status for s in read_sessions(out).values()} == {"no_trajectory"}


def test_session_error_excerpt_falls_back_on_empty_proxy_text(tmp_path: Path) -> None:
    out_dir = tmp_path / "sweagent" / "a__b.1"
    out_dir.mkdir(parents=True)
    (out_dir / "run_batch_exit_statuses.yaml").write_text(STATUSES_YAML)
    expected = "Uncaught CalledProcessError"
    assert run.session_error_excerpt(tmp_path, "a__b.1", None) == expected
    assert run.session_error_excerpt(tmp_path, "a__b.1", "") == expected
    assert run.session_error_excerpt(tmp_path, "a__b.1", "context too long") == "context too long"
    assert run.session_error_excerpt(tmp_path, "other", "") is None


def test_proxy_failure_without_error_text_falls_back_end_to_end(tmp_path: Path) -> None:
    (tmp_path / "proxy_requests.jsonl").write_text(request_log("a__b.1", 1, 500, None) + "\n")
    count, proxy_error = run.proxy_failures(tmp_path, "a__b.1")
    assert count == 1 and not proxy_error
    out_dir = tmp_path / "sweagent" / "a__b.1"
    out_dir.mkdir(parents=True)
    (out_dir / "run_batch_exit_statuses.yaml").write_text(STATUSES_YAML)
    assert run.session_error_excerpt(tmp_path, "a__b.1", proxy_error) == "Uncaught CalledProcessError"


def test_uncaught_error_text_tolerates_null_instances_by_exit_status() -> None:
    assert run.uncaught_error_text("instances_by_exit_status:\ntotal_cost: 0\n", "", "a__b.1") is None
    assert run.uncaught_error_text("instances_by_exit_status: null\n", "", "a__b.1") is None


def write_traj(run_dir: Path, instance_id: str, text: str) -> None:
    traj = run.trajectory_path(run_dir, instance_id)
    traj.parent.mkdir(parents=True)
    traj.write_text(text)


def test_trajectory_info_reads_info_fields(tmp_path: Path) -> None:
    assert run.trajectory_info(tmp_path, "a") == (None, None, False)  # no .traj
    info = {"exit_status": "submitted", "submission": "diff", "model_stats": {"api_calls": 7, "tokens_sent": 1}}
    write_traj(tmp_path, "a", json.dumps({"trajectory": [], "info": info}))
    assert run.trajectory_info(tmp_path, "a") == ("submitted", 7, True)
    write_traj(tmp_path, "b", json.dumps({"info": {"submission": None}}))
    assert run.trajectory_info(tmp_path, "b") == (None, None, False)
    write_traj(tmp_path, "c", json.dumps({"info": {"submission": "", "model_stats": {}}}))
    assert run.trajectory_info(tmp_path, "c") == (None, None, False)


def test_trajectory_info_partial_file_is_not_an_error(tmp_path: Path) -> None:
    write_traj(tmp_path, "torn", '{"trajectory": [{"a": 1}, {"a"')
    assert run.trajectory_info(tmp_path, "torn") == (None, None, False)
    write_traj(tmp_path, "noinfo", '{"trajectory": []}')
    assert run.trajectory_info(tmp_path, "noinfo") == (None, None, False)


def test_trajectory_info_rejects_wrong_shape(tmp_path: Path) -> None:
    write_traj(tmp_path, "bad", json.dumps({"info": {"exit_status": "x", "model_stats": {"api_calls": "many"}}}))
    with pytest.raises(ValueError, match="api_calls"):
        run.trajectory_info(tmp_path, "bad")


def test_merge_predictions_merges_and_preserves_content(tmp_path: Path) -> None:
    def pred(instance_id: str, patch: str | None) -> dict[str, dict[str, str | int | None]]:
        entry = {"model_name_or_path": "m", "instance_id": instance_id, "model_patch": patch, "extra": 1}
        return {instance_id: entry}

    written = {"a": pred("a", "diff \u00e9\n"), "b": pred("b", None)}
    for instance_id, preds in written.items():
        out = tmp_path / "sweagent" / instance_id
        out.mkdir(parents=True)
        (out / "preds.json").write_text(json.dumps(preds))
    run.merge_predictions(tmp_path, ["a", "b", "missing"])
    expected = {**written["a"], **written["b"]}
    assert (tmp_path / "preds.json").read_text() == json.dumps(expected, indent=2)


def test_merge_predictions_rejects_malformed_pred(tmp_path: Path) -> None:
    out = tmp_path / "sweagent" / "a"
    out.mkdir(parents=True)
    (out / "preds.json").write_text(json.dumps({"a": {"instance_id": "a"}}))
    with pytest.raises(ValueError, match="model_name_or_path"):
        run.merge_predictions(tmp_path, ["a"])
