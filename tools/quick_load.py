"""Quick concurrency check against any OpenAI-compatible server (vLLM, vllm-metal, mlx-lm).

Sends the first step (system prompt + user task) of real Hermes tasks with at most C
requests in flight, greedy decoding, and reports time-to-first-token, request latency,
and aggregate output throughput per concurrency level.

This is closed-loop (fixed concurrency): a sanity check of the server, not the open-loop
task-goodput benchmark. Each level uses a different slice of tasks so it can't reuse the
previous level's cached prompts; the few shared system prompts are warmed up first.

Usage (run with -I: the trace file is downloaded, untrusted input):
    uv run python -I tools/quick_load.py data/raw/hermes/data/train.jsonl --port 8100 --concurrency 1 4 8
"""

import argparse
import asyncio
import json
import time

import numpy as np
from openai import AsyncOpenAI

NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


def load_tasks(path: str, n: int, skip: int = 0) -> list[list[dict]]:
    tasks = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            conv = json.loads(line)["conversations"]
            tasks.append([{"role": "system", "content": conv[0]["value"]},
                          {"role": "user", "content": conv[1]["value"]}])
            if len(tasks) == n:
                break
    return tasks


async def one_request(client, model, messages, max_tokens):
    start = time.perf_counter()
    ttft, usage = None, None
    stream = await client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=0, stream=True,
        stream_options={"include_usage": True}, extra_body=NO_THINKING)
    async for chunk in stream:
        if ttft is None and chunk.choices and chunk.choices[0].delta.content:
            ttft = time.perf_counter() - start
        if chunk.usage:
            usage = chunk.usage
    return ttft, time.perf_counter() - start, usage.prompt_tokens, usage.completion_tokens


async def run_level(client, model, tasks, concurrency, max_tokens):
    sem = asyncio.Semaphore(concurrency)

    async def admitted(messages):
        async with sem:
            return await one_request(client, model, messages, max_tokens)

    start = time.perf_counter()
    results = await asyncio.gather(*(admitted(m) for m in tasks))
    wall = time.perf_counter() - start
    ttft, latency, n_in, n_out = (np.array(x, dtype=float) for x in zip(*results))
    print(f"{concurrency:>6}{len(tasks):>6}{wall:>9.1f}{n_out.sum() / wall:>12.1f}"
          f"{np.mean(ttft):>10.2f}{np.percentile(latency, 50):>10.2f}{np.percentile(latency, 95):>10.2f}"
          f"{n_in.mean():>10,.0f}{n_out.mean():>9.0f}")


async def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("hermes_jsonl")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--model", default="target", help="served model name")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--requests", type=int, default=16, help="requests per concurrency level")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--skip", type=int, default=0, help="skip the first N tasks (fresh prompts for a repeat run)")
    args = p.parse_args()

    client = AsyncOpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="unused", timeout=600)
    tasks = load_tasks(args.hermes_jsonl, args.requests * len(args.concurrency), args.skip)

    # Warm-up: compile kernels and cache each distinct system prompt once.
    distinct_systems = {t[0]["content"]: t for t in tasks}.values()
    await asyncio.gather(*(one_request(client, args.model, t, 1) for t in distinct_systems))

    print(f"{'conc':>6}{'reqs':>6}{'wall s':>9}{'out tok/s':>12}{'TTFT s':>10}"
          f"{'p50 s':>10}{'p95 s':>10}{'in tok':>10}{'out tok':>9}")
    for i, c in enumerate(args.concurrency):
        level_tasks = tasks[i * args.requests:(i + 1) * args.requests]
        await run_level(client, args.model, level_tasks, c, args.max_tokens)


if __name__ == "__main__":
    asyncio.run(main())
