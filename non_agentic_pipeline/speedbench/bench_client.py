"""Stream every SPEED-Bench subset prompt through a running vLLM server and record per-request metrics.

  bench_client.py --subset <speed_subset.jsonl> --run-dir <dir> --server-url http://127.0.0.1:8000 \
    --max-tokens 2048 --concurrency 1

Multi-turn prompts run their turns in order, each with the model's real earlier replies as history.
Writes <run-dir>/requests.csv (one row per turn) and <run-dir>/responses.jsonl (row, text, chunk gaps).
TTFT is the time to the first content chunk. Speculative decoding can emit several tokens per chunk, so
TPOT is (last chunk - first chunk) / (completion_tokens - 1) and chunk-to-chunk gaps are kept as ITL.
At concurrency 1 the server's /metrics counters are diffed around each request for queue/prefill/decode
time and speculative acceptance; above 1 those columns stay empty.
"""

import argparse
import asyncio
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, cast

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from gpu import mib, start_gpu_sampler, stop_gpu_sampler
from models import ChatMessage, RequestRow, ResponseRecord, ServerDelta, SubsetRow
from server_metrics import (
    SUCCESS,
    SPEC_POSITIONS,
    accept_length,
    accept_per_position,
    accept_rate,
    scrape_series,
    server_delta,
    wait_recorded,
)
from stats import percentile

MODEL = "Qwen/Qwen3-14B-AWQ"
GPU_SAMPLE_INTERVAL_S = 0.05
REQUEST_TIMEOUT_S = 1800.0
SCRAPE_TIMEOUT_S = 5.0
RECORD_WAIT_S = 15.0
RECORD_POLL_S = 0.1


class StreamResult(BaseModel):
    text: str
    start_ts: float
    t_send: float
    t_done: float
    content_times: list[float]
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None


@dataclass(frozen=True)
class RunContext:
    client: AsyncOpenAI
    max_tokens: int
    metrics_url: str
    per_request_deltas: bool
    requests_writer: "csv.DictWriter[str]"
    requests_handle: TextIO
    responses_handle: TextIO


def load_subset(path: Path) -> list[SubsetRow]:
    return [SubsetRow.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def chunk_gaps(content_times: list[float]) -> list[float]:
    return [b - a for a, b in zip(content_times, content_times[1:], strict=False)]


def tpot(content_times: list[float], completion_tokens: int) -> float | None:
    if completion_tokens <= 1 or len(content_times) < 2:
        return None
    return (content_times[-1] - content_times[0]) / (completion_tokens - 1)


def build_row(
    prompt: SubsetRow,
    turn_index: int,
    result: StreamResult,
    peak_gpu_mib: float,
    server: ServerDelta | None,
    error: str | None,
) -> RequestRow:
    gaps = chunk_gaps(result.content_times)
    spec = server.spec if server is not None else None
    per_position = accept_per_position(spec) if spec is not None else [None] * SPEC_POSITIONS
    return RequestRow(
        question_id=prompt.question_id,
        category=prompt.category,
        tier=prompt.tier,
        turn_index=turn_index,
        n_turns=prompt.n_turns,
        over_context=prompt.over_context,
        start_ts=result.start_ts,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        finish_reason=result.finish_reason,
        ttft_s=result.content_times[0] - result.t_send if result.content_times else None,
        tpot_s=tpot(result.content_times, result.completion_tokens),
        itl_mean_s=sum(gaps) / len(gaps) if gaps else None,
        itl_p99_s=percentile(sorted(gaps), 0.99) if gaps else None,
        itl_max_s=max(gaps) if gaps else None,
        e2e_s=result.t_done - result.t_send,
        n_chunks=len(result.content_times),
        peak_gpu_mib=peak_gpu_mib,
        queue_s=server.queue_s if server is not None else None,
        prefill_s=server.prefill_s if server is not None else None,
        decode_s=server.decode_s if server is not None else None,
        spec_drafts=spec.drafts if spec is not None else None,
        spec_draft_tokens=spec.draft_tokens if spec is not None else None,
        spec_accepted=spec.accepted if spec is not None else None,
        spec_accept_len=accept_length(spec) if spec is not None else None,
        spec_accept_rate=accept_rate(spec) if spec is not None else None,
        spec_accept_pos1=per_position[0],
        spec_accept_pos2=per_position[1],
        spec_accept_pos3=per_position[2],
        error=error,
    )


async def stream_chat(client: AsyncOpenAI, messages: list[ChatMessage], max_tokens: int) -> StreamResult:
    parts: list[str] = []
    content_times: list[float] = []
    prompt_tokens = 0
    completion_tokens = 0
    finish_reason: str | None = None
    start_ts = time.time()
    t_send = time.perf_counter()
    stream = await client.chat.completions.create(
        model=MODEL,
        messages=cast(list[ChatCompletionMessageParam], [m.model_dump() for m in messages]),
        max_tokens=max_tokens,
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    async for chunk in stream:
        now = time.perf_counter()
        if chunk.usage is not None:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens
        if chunk.choices:
            choice = chunk.choices[0]
            if choice.delta.content:
                parts.append(choice.delta.content)
                content_times.append(now)
            if choice.finish_reason:
                finish_reason = choice.finish_reason
    return StreamResult(
        text="".join(parts),
        start_ts=start_ts,
        t_send=t_send,
        t_done=time.perf_counter(),
        content_times=content_times,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
    )


def emit(ctx: RunContext, record: ResponseRecord) -> None:
    ctx.requests_writer.writerow(record.row.model_dump())
    ctx.requests_handle.flush()
    ctx.responses_handle.write(record.model_dump_json() + "\n")
    ctx.responses_handle.flush()


async def run_turn(
    ctx: RunContext, prompt: SubsetRow, turn_index: int, history: list[ChatMessage]
) -> StreamResult | None:
    before = await asyncio.to_thread(scrape_series, ctx.metrics_url, SCRAPE_TIMEOUT_S) if ctx.per_request_deltas else None
    sampler = start_gpu_sampler(GPU_SAMPLE_INTERVAL_S)
    started_ts = time.time()
    started = time.perf_counter()
    try:
        result = await stream_chat(ctx.client, history, ctx.max_tokens)
    except openai.BadRequestError as exc:
        # A request the server rejects (for example over the context window) is recorded, not hidden.
        peak = mib(stop_gpu_sampler(sampler))
        failed = StreamResult(
            text="",
            start_ts=started_ts,
            t_send=started,
            t_done=time.perf_counter(),
            content_times=[],
            prompt_tokens=0,
            completion_tokens=0,
            finish_reason=None,
        )
        emit(ctx, ResponseRecord(row=build_row(prompt, turn_index, failed, peak, None, str(exc)), text="", chunk_gaps_s=[]))
        return None
    peak = mib(stop_gpu_sampler(sampler))
    server: ServerDelta | None = None
    if before is not None:
        after = await asyncio.to_thread(
            wait_recorded, ctx.metrics_url, before.get(SUCCESS, 0.0), RECORD_WAIT_S, RECORD_POLL_S, SCRAPE_TIMEOUT_S
        )
        server = server_delta(before, after)
    row = build_row(prompt, turn_index, result, peak, server, None)
    emit(ctx, ResponseRecord(row=row, text=result.text, chunk_gaps_s=chunk_gaps(result.content_times)))
    return result


async def run_prompt(sem: asyncio.Semaphore, ctx: RunContext, prompt: SubsetRow) -> None:
    async with sem:
        history: list[ChatMessage] = []
        for turn_index, turn in enumerate(prompt.turns):
            history = [*history, ChatMessage(role="user", content=turn)]
            result = await run_turn(ctx, prompt, turn_index, history)
            if result is None:
                return
            history = [*history, ChatMessage(role="assistant", content=result.text)]


async def run_all(ctx: RunContext, prompts: list[SubsetRow], concurrency: int) -> None:
    sem = asyncio.Semaphore(concurrency)
    await asyncio.gather(*(run_prompt(sem, ctx, prompt) for prompt in prompts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--server-url", required=True, help="e.g. http://127.0.0.1:8000")
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = load_subset(args.subset)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(base_url=f"{args.server_url}/v1", api_key="not-needed", timeout=REQUEST_TIMEOUT_S)
    with (args.run_dir / "requests.csv").open("w", newline="") as requests_handle, (
        args.run_dir / "responses.jsonl"
    ).open("w") as responses_handle:
        writer = csv.DictWriter(requests_handle, fieldnames=list(RequestRow.model_fields))
        writer.writeheader()
        ctx = RunContext(
            client=client,
            max_tokens=args.max_tokens,
            metrics_url=f"{args.server_url}/metrics",
            per_request_deltas=args.concurrency == 1,
            requests_writer=writer,
            requests_handle=requests_handle,
            responses_handle=responses_handle,
        )
        asyncio.run(run_all(ctx, prompts, args.concurrency))
    print(f"wrote {args.run_dir / 'requests.csv'} and responses.jsonl for {len(prompts)} prompts")


if __name__ == "__main__":
    main()
