"""Run SWE-agent sessions against vLLM with Poisson arrivals and an in-flight cap of C.

Starts session_proxy.py, then launches one agent_entry.py subprocess (sweagent run-batch, one
instance) per session. Arrival i is due at t0 + arrivals[i]; it starts when a slot (of C) is free.
Each agent runs in its own process group; on a wall timeout the group is SIGTERMed then SIGKILLed, and the
session's swe-rex container (labelled cse291.session=<id>) is removed; docker failures there are logged to
container_cleanup.log, never fatal. A session without a .traj is "no_trajectory" (run-batch exits 0 on setup
failures). The run exits non-zero (RuntimeError) only if every session ended without completing.
Writes into --run-dir: sessions.jsonl/csv, tool_spans.jsonl/csv, proxy_requests.jsonl, preds.json,
agent_config.resolved.yaml, instances/, sweagent/<id>/ (trajs and logs).
"""

import argparse
import asyncio
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, StrictBool, StrictFloat, StrictInt, StrictStr, TypeAdapter, ValidationError

from poisson import arrivals
from session_proxy import RequestLog
from tool_span import ToolSpan

MIN_TOOL_TIMEOUT_S = 10
MAX_TOOL_TIMEOUT_S = 15
ERROR_EXCERPT_CHARS = 200
TERM_GRACE_S = 5
SESSION_LABEL_KEY = "cse291.session"
INSPECT_FORMAT = "{{.Name}} {{.State.Running}} {{.Created}}"
CLEANUP_LOG_NAME = "container_cleanup.log"
UUID_PATTERN = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
SESSION_WALL_TIMEOUT_S = 1800
PROXY_START_TIMEOUT_S = 30
HERE = Path(__file__).resolve().parent
M = TypeVar("M", bound=BaseModel)
EnvValue = StrictBool | StrictInt | StrictFloat | StrictStr


class Session(BaseModel):
    model_config = ConfigDict(frozen=True)
    instance_id: str
    image_key: str
    arrival_offset_s: float
    t_arrival: float
    t_start: float
    t_end: float
    returncode: int
    status: str
    exit_status: str | None
    api_calls: int | None
    submitted: bool
    failed_requests: int
    error_excerpt: str | None


class CleanupResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    removed: int
    failures: list[str]


class ContainerInfo(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    running: bool
    created: float


class InstanceRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    instance_id: str
    image_key: str
    image_name: str
    problem_statement: str
    base_commit: str | None = None


class AgentInstance(BaseModel):
    model_config = ConfigDict(frozen=True)
    instance_id: str
    image_name: str
    problem_statement: str
    repo_name: str
    base_commit: str


class CompletionKwargs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")
    max_tokens: int


class AgentModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")
    completion_kwargs: CompletionKwargs
    max_input_tokens: int | None = None


class AgentTools(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")
    execution_timeout: int
    env_variables: dict[str, EnvValue]


class AgentSection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")
    tools: AgentTools
    model: AgentModel


class AgentConfig(BaseModel):
    """The agent_config.yaml fields run.py reads or sets; every other key passes through unchanged."""

    model_config = ConfigDict(frozen=True, extra="allow")
    agent: AgentSection


class Prediction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")
    model_name_or_path: str
    instance_id: str
    model_patch: str | None


class TrajModelStats(BaseModel):
    model_config = ConfigDict(frozen=True)
    api_calls: int | None = None


class TrajInfo(BaseModel):
    model_config = ConfigDict(frozen=True, protected_namespaces=())
    exit_status: str | None = None
    submission: str | None = None
    model_stats: TrajModelStats | None = None


class Trajectory(BaseModel):
    model_config = ConfigDict(frozen=True)
    info: TrajInfo | None = None


PREDICTIONS = TypeAdapter(dict[str, Prediction])


class RunArgs(BaseModel):
    model_config = ConfigDict(frozen=True)
    instances_jsonl: Path
    count: int
    concurrency: int
    rate: float
    seed: int
    run_dir: Path
    upstream: str
    proxy_port: int
    max_model_len: int
    tool_timeout_s: int
    agent_config: Path
    sweagent_python: Path
    agent_entry: Path
    docker: str


def check_tool_timeout(tool_timeout_s: int) -> None:
    if not MIN_TOOL_TIMEOUT_S <= tool_timeout_s <= MAX_TOOL_TIMEOUT_S:
        raise ValueError(
            f"tool_timeout_s={tool_timeout_s} outside [{MIN_TOOL_TIMEOUT_S}, {MAX_TOOL_TIMEOUT_S}]"
        )


def resolve_config(base: AgentConfig, tool_timeout_s: int, max_model_len: int) -> AgentConfig:
    """Return the agent config with the tool timeout and input-token cap set for this run."""
    agent = base.agent
    tools = agent.tools.model_copy(update={
        "execution_timeout": tool_timeout_s,
        "env_variables": {**agent.tools.env_variables, "TOOL_TIMEOUT_S": tool_timeout_s},
    })
    model = agent.model.model_copy(update={"max_input_tokens": max_model_len - agent.model.completion_kwargs.max_tokens})
    return base.model_copy(update={"agent": agent.model_copy(update={"tools": tools, "model": model})})


def check_resolved_timeout(resolved: AgentConfig, tool_timeout_s: int) -> None:
    """The timeout SWE-agent will actually use must equal the requested one and stay <= 15 s."""
    actual = resolved.agent.tools.execution_timeout
    if actual != tool_timeout_s or actual > MAX_TOOL_TIMEOUT_S:
        raise ValueError(f"resolved execution_timeout={actual}, requested={tool_timeout_s}, max={MAX_TOOL_TIMEOUT_S}")


def read_jsonl(path: Path, model: type[M]) -> list[M]:
    return [model.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def is_invalid_json(error: ValidationError) -> bool:
    return any(detail["type"] == "json_invalid" for detail in error.errors())


def parse_jsonl_tolerating_torn_tail(text: str, model: type[M]) -> list[M]:
    """Parse JSONL while another process appends: only an unparsable final line (a partial write) is dropped."""
    lines = [line for line in text.splitlines() if line.strip()]
    rows = [model.model_validate_json(line) for line in lines[:-1]]
    if lines:
        try:
            rows.append(model.model_validate_json(lines[-1]))
        except ValidationError as error:
            if not is_invalid_json(error):
                raise
    return rows


def agent_instance(row: InstanceRow) -> AgentInstance:
    """SWE-agent `file` instance. Avoids the swesmith source, which calls the GitHub API per instance.

    SWE-smith rows check out the bug branch named by the instance id; rows with a base_commit use that SHA."""
    return AgentInstance(
        instance_id=row.instance_id,
        image_name=row.image_name,
        problem_statement=row.problem_statement,
        repo_name="testbed",
        base_commit=row.instance_id if row.base_commit is None else row.base_commit,
    )


def sweagent_command(args: RunArgs, config_path: Path, instance_file: Path, instance_id: str) -> list[str]:
    out_dir = args.run_dir / "sweagent" / instance_id
    return [
        str(args.sweagent_python),
        str(args.agent_entry),
        str(args.run_dir / "tool_spans.jsonl"),
        instance_id,
        "--",
        "--config", str(config_path),
        "--instances.type", "file",
        "--instances.path", str(instance_file),
        "--output_dir", str(out_dir),
        "--agent.model.api_base", f"http://127.0.0.1:{args.proxy_port}/s/{instance_id}/v1",
        "--instances.deployment.docker_args", json.dumps(["--label", f"{SESSION_LABEL_KEY}={instance_id}"]),
        "--num_workers", "1",
    ]


def proxy_failures(run_dir: Path, instance_id: str) -> tuple[int, str | None]:
    """Count of this session's proxy requests with status >= 400 and the first error text (truncated)."""
    log_path = run_dir / "proxy_requests.jsonl"
    if not log_path.exists():
        return 0, None
    logs = parse_jsonl_tolerating_torn_tail(log_path.read_text(), RequestLog)
    failed = [log for log in logs if log.session_id == instance_id and log.status >= 400]
    if not failed:
        return 0, None
    return len(failed), (failed[0].error or "")[:ERROR_EXCERPT_CHARS]


def sanitized_image(image: str) -> str:
    """The image part of a swe-rex container name (DockerDeployment._get_container_name)."""
    return "".join(c for c in image if c.isalnum() or c in "-_.")


def is_swerex_container(name: str, images: list[str]) -> bool:
    """swe-rex names containers `<sanitized image>-<uuid4>`."""
    return any(
        name.startswith(prefix := sanitized_image(image) + "-") and UUID_PATTERN.fullmatch(name[len(prefix):])
        for image in images
    )


def swerex_containers(containers: list[ContainerInfo], images: list[str]) -> list[ContainerInfo]:
    return [c for c in containers if is_swerex_container(c.name, images)]


def images_of(instances_jsonl: Path) -> list[str]:
    return sorted({r.image_name for r in read_jsonl(instances_jsonl, InstanceRow)})


def docker_lines(command: list[str]) -> list[str]:
    done = subprocess.run(command, check=True, capture_output=True, text=True)
    return [line for line in done.stdout.splitlines() if line.strip()]


def parse_inspect_lines(lines: list[str]) -> list[ContainerInfo]:
    """Parse `docker inspect --format '{{.Name}} {{.State.Running}} {{.Created}}'` output."""
    rows = [line.split(" ") for line in lines]
    return [
        ContainerInfo(
            name=name.lstrip("/"),
            running=running == "true",
            created=datetime.fromisoformat(re.sub(r"\.(\d{6})\d*", r".\1", created)).timestamp(),
        )
        for name, running, created in rows
    ]


def list_containers(docker: str) -> list[ContainerInfo]:
    ids = docker_lines([docker, "ps", "-aq", "--no-trunc"])
    if not ids:
        return []
    return parse_inspect_lines(docker_lines([docker, "inspect", "--format", INSPECT_FORMAT, *ids]))


def docker_try(command: list[str]) -> tuple[list[str], str | None]:
    """Run docker without raising: (stdout lines, one-line error text or None)."""
    try:
        done = subprocess.run(command, capture_output=True, text=True)
    except OSError as error:
        return [], f"{' '.join(command)} -> {type(error).__name__}: {error}"
    lines = [line for line in done.stdout.splitlines() if line.strip()]
    if done.returncode == 0:
        return lines, None
    return lines, f"{' '.join(command)} -> rc={done.returncode}: {' '.join(done.stderr.split())[:300]}"


def errors_of(*errors: str | None) -> list[str]:
    return [e for e in errors if e is not None]


def list_containers_tolerant(docker: str) -> tuple[list[ContainerInfo], list[str]]:
    """Like list_containers, but containers that vanish between ps and inspect (--rm) are skipped."""
    ids, ps_error = docker_try([docker, "ps", "-aq", "--no-trunc"])
    if not ids:
        return [], errors_of(ps_error)
    lines, inspect_error = docker_try([docker, "inspect", "--format", INSPECT_FORMAT, *ids])
    return parse_inspect_lines(lines), errors_of(ps_error, inspect_error)


def remove_each(docker: str, refs: list[str]) -> CleanupResult:
    """`docker rm -f` one container at a time so one failure does not hide the others."""
    results = [docker_try([docker, "rm", "-f", ref]) for ref in refs]
    failures = errors_of(*(error for _, error in results))
    return CleanupResult(removed=len(refs) - len(failures), failures=failures)


def running_swerex_names(docker: str, images: list[str]) -> list[str]:
    return [c.name for c in swerex_containers(list_containers(docker), images) if c.running]


def cleanup_swerex_containers(docker: str, images: list[str], since_epoch: float) -> CleanupResult:
    """Remove swe-rex containers (by name pattern only) created at or after since_epoch; never raises on docker errors."""
    containers, list_errors = list_containers_tolerant(docker)
    stale = [c.name for c in swerex_containers(containers, images) if c.created >= since_epoch]
    result = remove_each(docker, stale)
    return CleanupResult(removed=result.removed, failures=list_errors + result.failures)


def remove_session_containers(docker: str, instance_id: str) -> CleanupResult:
    ids, ps_error = docker_try([docker, "ps", "-aq", "--no-trunc", "--filter", f"label={SESSION_LABEL_KEY}={instance_id}"])
    result = remove_each(docker, ids)
    return CleanupResult(removed=result.removed, failures=errors_of(ps_error) + result.failures)


def append_cleanup_log(log_path: Path, failures: list[str]) -> None:
    if failures:
        with log_path.open("a") as handle:
            handle.writelines(f"{failure}\n" for failure in failures)


def signal_group(pgid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass  # group already gone


async def terminate_group(process: asyncio.subprocess.Process) -> int:
    """SIGTERM the agent's process group, then SIGKILL it if it outlives the grace period."""
    signal_group(process.pid, signal.SIGTERM)
    try:
        return await asyncio.wait_for(process.wait(), timeout=TERM_GRACE_S)
    except asyncio.TimeoutError:
        signal_group(process.pid, signal.SIGKILL)
        return await process.wait()


def trajectory_path(run_dir: Path, instance_id: str) -> Path:
    return run_dir / "sweagent" / instance_id / instance_id / f"{instance_id}.traj"


def classify_session(returncode: int, traj_exists: bool, timed_out: bool) -> str:
    """sweagent run-batch exits 0 even when an instance fails in setup, so a missing .traj is its own status."""
    if timed_out:
        return "wall_timeout"
    if returncode != 0:
        return "crashed"
    return "completed" if traj_exists else "no_trajectory"


def uncaught_error_text(statuses_yaml: str, log_text: str, instance_id: str) -> str | None:
    """`Uncaught <Class>[: message]` for this instance from run_batch_exit_statuses.yaml (+ run_batch.log), <= 200 chars."""
    by_status = ((yaml.safe_load(statuses_yaml) or {}).get("instances_by_exit_status")) or {}
    keys = [k for k, ids in by_status.items() if k.startswith("Uncaught") and instance_id in ids]
    if not keys:
        return None
    class_name = keys[0].removeprefix("Uncaught").strip()
    matches = [m.group(1) for m in re.finditer(rf"^(?:[\w.]+\.)?{re.escape(class_name)}: (.*)$", log_text, re.MULTILINE)]
    text = f"{keys[0]}: {matches[-1]}" if matches else keys[0]
    return text[:ERROR_EXCERPT_CHARS]


def run_batch_error(run_dir: Path, instance_id: str) -> str | None:
    out_dir = run_dir / "sweagent" / instance_id
    statuses = out_dir / "run_batch_exit_statuses.yaml"
    if not statuses.exists():
        return None
    log = out_dir / "run_batch.log"
    return uncaught_error_text(statuses.read_text(), log.read_text() if log.exists() else "", instance_id)


def session_error_excerpt(run_dir: Path, instance_id: str, proxy_error: str | None) -> str | None:
    """Proxy error text if non-empty, else the run-batch uncaught exception."""
    return proxy_error or run_batch_error(run_dir, instance_id)


def check_any_trajectory(sessions: list[Session]) -> None:
    """Fail the cell when every session ended without completing (partial failures are only counted)."""
    if sessions and all(s.status != "completed" for s in sessions):
        counts: dict[str, int] = {}
        for s in sessions:
            counts[s.status] = counts.get(s.status, 0) + 1
        raise RuntimeError(f"all {len(sessions)} sessions ended without completing: status counts {counts}")


def trajectory_info(run_dir: Path, instance_id: str) -> tuple[str | None, int | None, bool]:
    traj = trajectory_path(run_dir, instance_id)
    if not traj.exists():
        return None, None, False
    try:
        info = Trajectory.model_validate_json(traj.read_text()).info
    except ValidationError as error:
        if not is_invalid_json(error):
            raise
        return None, None, False  # partial .traj left by a killed session
    if info is None:
        return None, None, False
    api_calls = None if info.model_stats is None else info.model_stats.api_calls
    return info.exit_status, api_calls, bool(info.submission)


async def run_session(
    args: RunArgs, config_path: Path, row: InstanceRow, offset_s: float, t0: float, slots: asyncio.Semaphore
) -> Session:
    instance_id = row.instance_id
    delay = t0 + offset_s - time.time()
    if delay > 0:
        await asyncio.sleep(delay)
    t_arrival = time.time()
    async with slots:
        instance_file = args.run_dir / "instances" / f"{instance_id}.jsonl"
        instance_file.write_text(json.dumps(agent_instance(row).model_dump()) + "\n")
        log_path = args.run_dir / "sweagent" / f"{instance_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        t_start = time.time()
        with log_path.open("w") as log:
            process = await asyncio.create_subprocess_exec(
                *sweagent_command(args, config_path, instance_file, instance_id),
                stdout=log, stderr=log, start_new_session=True,
            )
            timed_out = False
            try:
                returncode = await asyncio.wait_for(process.wait(), timeout=SESSION_WALL_TIMEOUT_S)
            except asyncio.TimeoutError:
                returncode = await terminate_group(process)
                timed_out = True
            except asyncio.CancelledError:
                await terminate_group(process)
                raise
        t_end = time.time()
    status = classify_session(returncode, trajectory_path(args.run_dir, instance_id).exists(), timed_out)
    if status != "completed":
        cleanup = await asyncio.to_thread(remove_session_containers, args.docker, instance_id)
        append_cleanup_log(args.run_dir / CLEANUP_LOG_NAME, cleanup.failures)
    exit_status, api_calls, submitted = trajectory_info(args.run_dir, instance_id)
    failed_requests, proxy_error = proxy_failures(args.run_dir, instance_id)
    error_excerpt = session_error_excerpt(args.run_dir, instance_id, proxy_error)
    session = Session(
        instance_id=instance_id,
        image_key=row.image_key,
        arrival_offset_s=offset_s,
        t_arrival=t_arrival,
        t_start=t_start,
        t_end=t_end,
        returncode=returncode,
        status=status,
        exit_status=exit_status,
        api_calls=api_calls,
        submitted=submitted,
        failed_requests=failed_requests,
        error_excerpt=error_excerpt,
    )
    with (args.run_dir / "sessions.jsonl").open("a") as handle:
        handle.write(session.model_dump_json() + "\n")
    return session


def wait_for_proxy(port: int) -> None:
    deadline = time.time() + PROXY_START_TIMEOUT_S
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1)
            return
        except OSError:
            time.sleep(0.3)
    raise RuntimeError(f"session proxy did not answer on port {port} within {PROXY_START_TIMEOUT_S}s")


def merge_predictions(run_dir: Path, instance_ids: list[str]) -> None:
    merged: dict[str, Prediction] = {}
    for instance_id in instance_ids:
        preds = run_dir / "sweagent" / instance_id / "preds.json"
        if preds.exists():
            merged.update(PREDICTIONS.validate_json(preds.read_text()))
    (run_dir / "preds.json").write_text(json.dumps(PREDICTIONS.dump_python(merged), indent=2))


def write_csv(path: Path, rows: list[BaseModel]) -> None:
    if not rows:
        return
    fields = list(type(rows[0]).model_fields)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(r.model_dump() for r in rows)


def write_tables(run_dir: Path) -> None:
    sessions = read_jsonl(run_dir / "sessions.jsonl", Session)
    write_csv(run_dir / "sessions.csv", list(sessions))
    spans_path = run_dir / "tool_spans.jsonl"
    spans = read_jsonl(spans_path, ToolSpan) if spans_path.exists() else []
    write_csv(run_dir / "tool_spans.csv", list(spans))


async def run_all(args: RunArgs, config_path: Path, rows: list[InstanceRow]) -> list[Session]:
    offsets = arrivals(args.rate, len(rows), args.seed)
    slots = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    return list(await asyncio.gather(*(run_session(args, config_path, r, o, t0, slots) for r, o in zip(rows, offsets))))


def run(args: RunArgs) -> None:
    check_tool_timeout(args.tool_timeout_s)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "instances").mkdir(exist_ok=True)
    rows = read_jsonl(args.instances_jsonl, InstanceRow)[: args.count]
    base = AgentConfig.model_validate(yaml.safe_load(args.agent_config.read_text()))
    resolved = resolve_config(base, args.tool_timeout_s, args.max_model_len)
    check_resolved_timeout(resolved, args.tool_timeout_s)
    config_path = args.run_dir / "agent_config.resolved.yaml"
    config_path.write_text(yaml.safe_dump(resolved.model_dump(exclude_unset=True), sort_keys=False))
    proxy = subprocess.Popen(
        [
            sys.executable, str(HERE / "session_proxy.py"),
            "--upstream", args.upstream,
            "--port", str(args.proxy_port),
            "--log-path", str(args.run_dir / "proxy_requests.jsonl"),
        ]
    )
    try:
        wait_for_proxy(args.proxy_port)
        # Readers (report.py) must never hit a missing file when every session fails before any request/span.
        (args.run_dir / "proxy_requests.jsonl").touch()
        (args.run_dir / "tool_spans.jsonl").touch()
        sessions = asyncio.run(run_all(args, config_path, rows))
    finally:
        proxy.terminate()
        proxy.wait()
    merge_predictions(args.run_dir, [r.instance_id for r in rows])
    write_tables(args.run_dir)
    check_any_trajectory(sessions)


def parse_args() -> RunArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-jsonl", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--rate", type=float, required=True, help="sessions per second; huge = back-to-back")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--proxy-port", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--tool-timeout-s", type=int, required=True)
    parser.add_argument("--agent-config", type=Path, required=True)
    parser.add_argument("--sweagent-python", type=Path, required=True)
    parser.add_argument("--agent-entry", type=Path, required=True)
    parser.add_argument("--docker", required=True, help="docker executable used to remove leaked containers")
    return RunArgs.model_validate(vars(parser.parse_args()))


if __name__ == "__main__":
    run(parse_args())
