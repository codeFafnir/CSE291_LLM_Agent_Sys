""" Send a synthetic coding-agent workload (synth_workload.py) to a running vLLM/SGLang server  """

import argparse
import asyncio
import contextlib
import json
import math
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from metrics import tpot_s
from synth_workload import MIXES, Task, build_tasks, fit_to_capacity

DEFAULT_TASKS = 32  


@dataclass(frozen=True)
class Reply:
    text: str = ""
    ttft_s: float | None = None
    e2e_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None  # prompt tokens served from cache, when the server reports it
    error: str = ""  # "" | timeout | http_<status> | error


def stream_chat(base_url: str, model: str, messages: list[dict[str, str]], max_tokens: int, timeout_s: float) -> Reply:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}  # thinking off, as in run_baseline.sh
    req = urllib.request.Request(f"{base_url}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    ttft, parts, usage = None, [], {}
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                payload = json.loads(line[6:])
                usage = payload.get("usage") or usage
                content = ((payload.get("choices") or [{}])[0].get("delta") or {}).get("content")
                if content:
                    ttft = ttft if ttft is not None else time.perf_counter() - start
                    parts.append(content)
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        return Reply("".join(parts), ttft, time.perf_counter() - start, usage.get("prompt_tokens"), usage.get("completion_tokens"), cached)
    except urllib.error.HTTPError as err:
        return Reply(error=f"http_{err.code}")
    except TimeoutError:
        return Reply(error="timeout")
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return Reply(error="error")


@dataclass
class Run:
    base_url: str
    model: str
    gate: contextlib.AbstractAsyncContextManager[object]  
    pool: ThreadPoolExecutor
    timeout_s: float
    deadline_s: float | None
    t0: float
    requests: list[dict[str, object]]
    tasks: list[dict[str, object]]


def request_row(task: Task, i: int, cap: int, enqueued: float, admitted: float, reply: Reply) -> dict[str, object]:
    tpot = tpot_s(reply.ttft_s, reply.e2e_s, reply.completion_tokens) if reply.ttft_s and reply.e2e_s and reply.completion_tokens else None
    return {"task_id": task.task_id, "step": i, "workload": task.workload, "t_enqueue": enqueued, "t_admit": admitted, "t_end": time.time(),
            "queue_wait_s": admitted - enqueued, "ttft_s": reply.ttft_s, "tpot_s": tpot, "e2e_s": reply.e2e_s, "output_cap": cap,
            "prompt_tokens": reply.prompt_tokens, "completion_tokens": reply.completion_tokens, "cached_tokens": reply.cached_tokens, "error": reply.error}


async def run_task(task: Task, run: Run) -> None:
    loop = asyncio.get_running_loop()
    await asyncio.sleep(max(task.arrival_s - (time.time() - run.t0), 0.0))
    arrival = time.time()
    messages = [{"role": "system", "content": task.system}, {"role": "user", "content": task.user}]
    queue_wait, outcome, done = 0.0, "completed", 0
    for i, step in enumerate(task.steps):
        enqueued = time.time()
        async with run.gate:
            admitted = time.time()
            reply = await loop.run_in_executor(run.pool, stream_chat, run.base_url, run.model, messages, step.output_cap, run.timeout_s)
        queue_wait += admitted - enqueued
        run.requests.append(request_row(task, i, step.output_cap, enqueued, admitted, reply))
        if reply.error:
            outcome = reply.error
            break
        done += 1
        messages.append({"role": "assistant", "content": reply.text or "(empty)"})
        if step.tool_result:
            await asyncio.sleep(step.gap_s)  
            messages.append({"role": "user", "content": step.tool_result})
    total = time.time() - arrival
    run.tasks.append({"task_id": task.task_id, "workload": task.workload, "t_arrival": arrival, "t_end": arrival + total, "completion_s": total,
                      "queue_wait_s": queue_wait, "steps_planned": len(task.steps), "steps_done": done, "outcome": outcome,
                      "success": outcome == "completed" and (run.deadline_s is None or total <= run.deadline_s)})


def pct(values: list[float], q: float) -> float | None:
    s = sorted(values)
    if not s:
        return None
    pos = q * (len(s) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def dist(values: list[float]) -> dict[str, float | None]:
    return {"mean": sum(values) / len(values) if values else None, "p50": pct(values, 0.5), "p95": pct(values, 0.95), "p99": pct(values, 0.99)}


def col(rows: list[dict[str, object]], key: str) -> list[float]:
    return [float(r[key]) for r in rows if r.get(key) is not None]  


def summarize(requests: list[dict[str, object]], tasks: list[dict[str, object]], deadline_s: float | None) -> dict[str, object]:
    ok = [r for r in requests if not r["error"]]
    done = [t for t in tasks if t["outcome"] == "completed"]
    window = max(max((float(t["t_end"]) for t in tasks), default=0.0) - min((float(t["t_arrival"]) for t in tasks), default=0.0), 1e-9)
    prompt, cached = sum(col(ok, "prompt_tokens")), sum(col(ok, "cached_tokens"))
    return {
        "window_s": window, "tasks": len(tasks), "deadline_s": deadline_s,
        "task_outcomes": {o: sum(1 for t in tasks if t["outcome"] == o) for o in {str(t["outcome"]) for t in tasks}},
        "task_goodput_per_s": sum(1 for t in tasks if t["success"]) / window,  # successful tasks per second within the deadline
        "task_completion_s": dist(col(done, "completion_s")), "task_queue_wait_s": dist(col(done, "queue_wait_s")),
        "requests": len(requests), "request_errors": len(requests) - len(ok), "requests_per_s": len(ok) / window,
        "output_tokens_per_s": sum(col(ok, "completion_tokens")) / window,
        "request_queue_wait_s": dist(col(requests, "queue_wait_s")), "ttft_s": dist(col(ok, "ttft_s")), "tpot_s": dist(col(ok, "tpot_s")),
        "e2e_s": dist(col(ok, "e2e_s")), "actual_output_tokens": dist(col(ok, "completion_tokens")),
        "cached_fraction": cached / prompt if prompt and any(r["cached_tokens"] is not None for r in ok) else None,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="Qwen/Qwen3-14B-AWQ")
    p.add_argument("--mix", choices=sorted(MIXES), default="mixed")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-model-len", type=int, required=True, help="server context window; components that cannot fit are dropped")
    p.add_argument("--num-tasks", type=int, default=DEFAULT_TASKS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=8, help="admitted-request concurrency; 0 = no gate (engine FCFS only)")
    p.add_argument("--arrival", choices=("steady", "bursty", "backlog"), default="steady")
    p.add_argument("--rate", type=float, default=0.5, help="mean task arrivals per second")
    p.add_argument("--context", type=int, help="override initial context tokens")
    p.add_argument("--output-cap", type=int, help="override max_tokens per call")
    p.add_argument("--steps", type=int, help="override requests per task")
    p.add_argument("--tool-gap", type=float, help="override tool gap, seconds")
    p.add_argument("--prefix", choices=("shared", "unique"))
    p.add_argument("--timeout-s", type=float, default=300.0, help="per-request timeout")
    p.add_argument("--deadline-s", type=float, help="task-latency target for goodput (arrival to last reply)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    over = {"context_tokens": args.context, "output_cap": args.output_cap, "steps": args.steps, "tool_gap_s": args.tool_gap, "prefix": args.prefix}
    mix = tuple((w, replace(s, **{k: v for k, v in over.items() if v is not None})) for w, s in MIXES[args.mix])
    fits = fit_to_capacity(mix, args.max_model_len)
    if not fits:
        print(f"infeasible: no part of {args.mix} fits max_model_len={args.max_model_len}", file=sys.stderr)
        return 3
    if len(fits) < len(mix):
        print(f"adapted to capacity: dropped {sorted({s.name for _, s in mix} - {s.name for _, s in fits})}", file=sys.stderr)
    tasks = build_tasks(fits, args.num_tasks, args.seed, args.arrival, args.rate, args.max_model_len)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "workload.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "components": [{"p": w, **asdict(s)} for w, s in fits]}, indent=2))

    pool = ThreadPoolExecutor(max_workers=max(len(tasks), 8))
    gate = asyncio.Semaphore(args.k) if args.k else contextlib.nullcontext()
    run = Run(args.base_url, args.model, gate, pool, args.timeout_s, args.deadline_s, time.time(), [], [])

    async def go() -> None:
        await asyncio.gather(*(run_task(t, run) for t in tasks))

    asyncio.run(go())
    pool.shutdown(wait=False)
    for name, rows in (("requests", run.requests), ("tasks", run.tasks)):
        (args.out_dir / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = summarize(run.requests, run.tasks, args.deadline_s)
    (args.out_dir / "synth_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("tasks", "task_outcomes", "task_goodput_per_s", "task_completion_s", "task_queue_wait_s", "ttft_s", "cached_fraction")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
