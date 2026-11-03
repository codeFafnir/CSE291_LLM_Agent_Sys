""" Synthetic coding-agent workload """

import functools
import math
import random
from dataclasses import dataclass

CHARS_PER_TOKEN = 3.3
PREFIX_CAP_TOKENS = 3072
MIN_RESULT_TOKENS = 24
MAX_RESULT_TOKENS = 4000
SAFETY_TOKENS = 96
BURST_SIZE = 8

TOOL_DEFS = """You are a software engineering agent working in a checked-out repository.
Reply with exactly one tool call as JSON: {"tool": <name>, "args": {...}}. Tools: bash(cmd), read_file(path, start, end),
grep(pattern, path), edit(path, old, new), run_tests(target). Inspect before editing, keep edits minimal, re-run the
failing test after each edit, and stop with {"tool": "finish", "args": {"summary": str}} when done.
"""

WORDS = ("package", "version", "advisory", "result", "target", "entry", "scope", "digest", "load", "parse", "merge", "resolve")


# code-shaped text


def code_line(r: random.Random) -> str:
    a, b, c = (r.choice(WORDS) for _ in range(3))
    return r.choice((
        f"def {a}_{b}({c}: str, strict: bool = False) -> list[str]:",
        f"    {a} = {b}_{c}({c}, limit={r.randrange(4, 512)})",
        f"    if {a} is None or len({b}) > {r.randrange(4, 512)}:",
        f'    raise ValueError(f"bad {a}: {{{b}!r}}")',
    ))


def tool_output(r: random.Random) -> str:
    path = f"src/{r.choice(WORDS)}/{r.choice(WORDS)}.py"
    lines = [code_line(r) for _ in range(r.randrange(6, 20))]
    kind = r.randrange(3)
    if kind == 0:
        return f"$ sed -n '1,{len(lines)}p' {path}\n" + "\n".join(f"{i + 1:4d}  {s}" for i, s in enumerate(lines))
    if kind == 1:
        return f"$ grep -rn '{r.choice(WORDS)}' src/\n" + "\n".join(f"{path}:{r.randrange(1, 900)}:{s.strip()}" for s in lines)
    return f"$ pytest -x {path}\n" + "\n".join(lines) + f"\nE   AssertionError\n=== 1 failed, {r.randrange(1, 90)} passed ==="


def est_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)


def fill(r: random.Random, tokens: int) -> str:
    target = int(tokens * CHARS_PER_TOKEN)
    out: list[str] = []
    while sum(map(len, out)) < target:
        out.append(tool_output(r))
    return "\n".join(out)[:target]


# workloads and tasks


@dataclass(frozen=True)
class Spec:
    name: str
    context_tokens: int  # initial prompt size (shared prefix + task statement)
    output_cap: int  # max_tokens per call
    steps: int  # model requests per task
    tool_gap_s: float  # idle time after each reply
    prefix: str = "shared"  # shared | unique (a nonce first, so no prefix block is shared across tasks)
    result_median_tokens: int = 400  # tool-result size: lognormal, so heavy-tailed


SHORT = Spec("coding_short", 512, 32, 1, 0.0)
MID = Spec("coding_mid", 4096, 256, 4, 1.0)
LONG = Spec("coding_long", 16384, 1024, 8, 1.0)
MIXES: dict[str, tuple[tuple[float, Spec], ...]] = {  
    "smoke": ((1.0, Spec("smoke", 512, 32, 2, 0.1)),),
    "coding_short": ((1.0, SHORT),),
    "coding_mid": ((1.0, MID),),
    "coding_long": ((1.0, LONG),),
    "mixed": ((0.5, SHORT), (0.35, MID), (0.15, LONG)),  
    "heldout": ((0.25, SHORT), (0.35, MID), (0.40, LONG)),  
}


@dataclass(frozen=True)
class Step:
    output_cap: int
    tool_result: str  
    gap_s: float


@dataclass(frozen=True)
class Task:
    task_id: int
    arrival_s: float
    workload: str
    system: str
    user: str
    steps: tuple[Step, ...]


def fit_to_capacity(mix: tuple[tuple[float, Spec], ...], max_model_len: int) -> tuple[tuple[float, Spec], ...]:
    return tuple((w, s) for w, s in mix if s.context_tokens + s.output_cap + SAFETY_TOKENS <= max_model_len)


@functools.cache
def shared_prefix(tokens: int) -> str:
    return TOOL_DEFS + fill(random.Random(0), max(tokens - est_tokens(TOOL_DEFS), 0))


def build_task(spec: Spec, task_id: int, arrival_s: float, r: random.Random, max_model_len: int) -> Task:
    system = shared_prefix(min(spec.context_tokens // 2, PREFIX_CAP_TOKENS))
    if spec.prefix == "unique":
        system = f"Session {task_id}-{random.Random(task_id).getrandbits(64):016x}\n" + system
    user = "Fix the failing behaviour described below.\n" + fill(r, max(spec.context_tokens - est_tokens(system), 16))
    total = est_tokens(system) + est_tokens(user)
    steps: list[Step] = []
    for i in range(spec.steps):
        total += spec.output_cap  
        room = spec.output_cap + SAFETY_TOKENS
        size = min(max(int(r.lognormvariate(math.log(spec.result_median_tokens), 0.9)), MIN_RESULT_TOKENS), MAX_RESULT_TOKENS,
                   max_model_len - total - room)
        if i == spec.steps - 1 or size < MIN_RESULT_TOKENS:  # last step, or no room left: end the task, never overflow
            steps.append(Step(spec.output_cap, "", 0.0))
            break
        result = "Tool output:\n" + fill(r, size)
        total += est_tokens(result)
        steps.append(Step(spec.output_cap, result, spec.tool_gap_s))
    return Task(task_id, arrival_s, spec.name, system, user, tuple(steps))


def arrival_times(kind: str, n: int, rate: float, r: random.Random) -> list[float]:
    """steady: Poisson at `rate`/s. bursty: BURST_SIZE tasks at once, same mean rate. backlog: everything at t=0."""
    if kind == "backlog":
        return [0.0] * n
    if kind == "bursty":
        return [(i // BURST_SIZE) * BURST_SIZE / rate for i in range(n)]
    t, times = 0.0, []
    for _ in range(n):
        t += r.expovariate(rate)
        times.append(t)
    return times


def build_tasks(mix: tuple[tuple[float, Spec], ...], n: int, seed: int, arrival: str, rate: float, max_model_len: int) -> list[Task]:
    r = random.Random(seed)
    times = arrival_times(arrival, n, rate, r)
    specs, weights = [s for _, s in mix], [w for w, _ in mix]
    return [build_task(r.choices(specs, weights)[0], i, ts, random.Random(f"{seed}:{i}"), max_model_len) for i, ts in enumerate(times)]
