"""Synthetic prompt-length x concurrency load against a running server, logging where memory runs out.

  capacity_sweep.py sweep ...           climb prompt lengths at one concurrency, append rows to capacity_log.jsonl
  capacity_sweep.py startup-failed ...  record that the server died before serving (oom_startup)

`sweep` exits 10 after an oom_runtime cell so the caller restarts the server.
"""

import argparse
import http.client
import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from capacity_types import OOM_MARKERS, CapacityRow, Outcome
from collect_server_metrics import Engine, sample
from long_prompts import long_prompts
from summarize_sla import Sample, read_env_file, sglang_spec, vllm_spec

TOKENIZER = "Qwen/Qwen3-14B-AWQ"
MODEL = "Qwen/Qwen3-14B-AWQ"
SAMPLE_INTERVAL_S = 0.2
CHAT_OVERHEAD_TOKENS = 32
REQUEST_TIMEOUT_S = 900.0
OOM_EXIT_CODE = 10
PromptMode = Literal["none", "index"]

KV_KEY: dict[Engine, str] = {"vllm": "vllm:kv_cache_usage_perc", "sglang": "sglang:token_usage"}
WAITING_KEY: dict[Engine, str] = {"vllm": "vllm:num_requests_waiting", "sglang": "sglang:num_queue_reqs"}
PREEMPT_KEY: dict[Engine, str] = {
    "vllm": "vllm:num_preemptions_total",
    "sglang": "sglang:num_retracted_requests_total",
}


@dataclass(frozen=True)
class RequestResult:
    status: int
    ttft_s: float | None
    e2e_s: float | None
    error: str | None


@dataclass(frozen=True)
class Peaks:
    gpu_mib: float
    kv_usage: float
    waiting: float


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def corpus_ids(tokenizer: PreTrainedTokenizerBase) -> list[int]:
    text = "\n\n".join(prompt.text for prompt in long_prompts())
    return tokenizer(text)["input_ids"]


def build_prompt(tokenizer: PreTrainedTokenizerBase, corpus: list[int], tokens: int, index: int) -> str:
    # A unique nonce and a distinct window keep requests from sharing prefix-cache blocks.
    body = max(tokens - CHAT_OVERHEAD_TOKENS, 1)
    start = (index * 997) % max(len(corpus) - body, 1)
    return f"Request {uuid.uuid4().hex}\n" + tokenizer.decode(corpus[start : start + body])


def first_content(line: str) -> bool:
    payload = line.strip().removeprefix("data: ")
    if not line.startswith("data: ") or payload == "[DONE]":
        return False
    choices = json.loads(payload).get("choices") or [{}]
    return bool(choices[0].get("delta", {}).get("content"))


def post_stream(base_url: str, prompt: str, max_tokens: int, priority: int | None) -> RequestResult:
    body: dict[str, object] = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    if priority is not None:
        body["priority"] = priority
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    ttft: float | None = None
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            for raw in response:
                if ttft is None and first_content(raw.decode()):
                    ttft = time.perf_counter() - started
            return RequestResult(response.status, ttft, time.perf_counter() - started, None)
    except urllib.error.HTTPError as err:
        return RequestResult(err.code, None, None, err.read().decode(errors="replace")[:300])
    except (urllib.error.URLError, http.client.HTTPException, OSError) as err:
        return RequestResult(0, None, None, repr(err))


def sample_peaks(engine: Engine, url: str, stop: threading.Event, out: list[Peaks]) -> None:
    peak = Peaks(0.0, 0.0, 0.0)
    while not stop.is_set():
        try:
            snap = Sample.model_validate(sample(url, engine, 2.0))
            peak = Peaks(
                max(peak.gpu_mib, snap.gpu_used_mib),
                max(peak.kv_usage, snap.series.get(KV_KEY[engine], 0.0)),
                max(peak.waiting, snap.series.get(WAITING_KEY[engine], 0.0)),
            )
        except (urllib.error.URLError, http.client.HTTPException, OSError):
            pass
        stop.wait(SAMPLE_INTERVAL_S)
    out.append(peak)


def log_marker(log_path: Path, offset: int) -> str | None:
    with log_path.open("rb") as handle:
        handle.seek(offset)
        for line in handle.read().decode(errors="replace").splitlines():
            if any(marker in line for marker in OOM_MARKERS):
                return line.strip()
    return None


def classify(
    results: list[RequestResult], marker: str | None, preemptions: float, peaks: Peaks
) -> tuple[Outcome, str]:
    server_down = any(r.status == 0 for r in results)
    if marker is not None or server_down:
        return "oom_runtime", marker or next(r.error or "" for r in results if r.status == 0)
    rejected = [r for r in results if r.status == 400]
    if rejected:
        return "rejected_too_long", rejected[0].error or ""
    if preemptions > 0.0:
        return "preempted", f"preemptions={preemptions:.0f}"
    if peaks.waiting > 0.0:
        return "queued", f"peak_waiting={peaks.waiting:.0f}"
    return "ok", ""


def run_cell(
    engine: Engine,
    base_url: str,
    run_dir: Path,
    knobs: dict[str, str],
    concurrency: int,
    tokens: int,
    max_tokens: int,
    priority_mode: PromptMode,
    tokenizer: PreTrainedTokenizerBase,
    corpus: list[int],
) -> CapacityRow:
    metrics_url = f"{base_url}/metrics"
    log_path = run_dir / "serve.log"
    offset = log_path.stat().st_size
    before = Sample.model_validate(sample(metrics_url, engine, 5.0))
    stop = threading.Event()
    peaks: list[Peaks] = []
    sampler = threading.Thread(target=sample_peaks, args=(engine, metrics_url, stop, peaks), daemon=True)
    sampler.start()
    prompts = [build_prompt(tokenizer, corpus, tokens, i) for i in range(concurrency)]
    priorities = [i if priority_mode == "index" else None for i in range(concurrency)]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(post_stream, base_url, p, max_tokens, pr) for p, pr in zip(prompts, priorities)]
        results = [f.result() for f in futures]
    stop.set()
    sampler.join()
    peak = peaks[0]
    try:
        after = Sample.model_validate(sample(metrics_url, engine, 5.0))
        preemptions = after.series.get(PREEMPT_KEY[engine], 0.0) - before.series.get(PREEMPT_KEY[engine], 0.0)
        spec = vllm_spec(before, after) if engine == "vllm" else sglang_spec([after])
        accept = spec.mean_acceptance_length
    except (urllib.error.URLError, http.client.HTTPException, OSError):
        preemptions, accept = 0.0, None
    marker = log_marker(log_path, offset)
    outcome, detail = classify(results, marker, preemptions, peak)
    done = [r for r in results if r.status == 200]
    ttfts = [r.ttft_s for r in done if r.ttft_s is not None]
    e2es = [r.e2e_s for r in done if r.e2e_s is not None]
    return CapacityRow(
        engine=engine,
        run_id=run_dir.name,
        max_model_len=int(knobs["MAX_MODEL_LEN"]),
        max_num_seqs=int(knobs["MAX_NUM_SEQS"]),
        max_batched_tokens=int(knobs["MAX_BATCHED_TOKENS"]),
        eviction_policy=knobs["EVICTION_POLICY"],
        preemption_policy=knobs["PREEMPTION_POLICY"],
        concurrency=concurrency,
        prompt_tokens=tokens,
        max_tokens=max_tokens,
        requests_sent=len(results),
        requests_completed=len(done),
        peak_gpu_mib=peak.gpu_mib,
        peak_kv_usage=peak.kv_usage,
        preemptions=preemptions,
        ttft_p50_s=percentile(ttfts, 0.50),
        ttft_p99_s=percentile(ttfts, 0.99),
        e2e_p50_s=percentile(e2es, 0.50),
        e2e_p99_s=percentile(e2es, 0.99),
        accept_length=accept,
        outcome=outcome,
        detail=detail,
    )


def append_row(run_dir: Path, row: CapacityRow) -> None:
    with (run_dir / "capacity_log.jsonl").open("a") as handle:
        handle.write(row.model_dump_json() + "\n")
    print(
        f"{row.engine} len={row.max_model_len} seqs={row.max_num_seqs} conc={row.concurrency} "
        f"prompt={row.prompt_tokens} outcome={row.outcome} peak_mib={row.peak_gpu_mib} {row.detail[:120]}"
    )


def sweep(args: argparse.Namespace) -> int:
    engine: Engine = args.engine
    run_dir: Path = args.run_dir
    knobs = read_env_file(run_dir / "serve_config.env")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    corpus = corpus_ids(tokenizer)
    limit = int(knobs["MAX_MODEL_LEN"]) - args.max_tokens
    for tokens in sorted(t for t in args.prompt_tokens if t <= limit):
        row = run_cell(
            engine, args.base_url, run_dir, knobs, args.concurrency, tokens, args.max_tokens,
            args.request_priority, tokenizer, corpus,
        )
        append_row(run_dir, row)
        if row.outcome == "oom_runtime":
            return OOM_EXIT_CODE
        if row.outcome == "rejected_too_long":
            return 0
    return 0


def startup_failed(args: argparse.Namespace) -> int:
    run_dir: Path = args.run_dir
    knobs = read_env_file(run_dir / "serve_config.env")
    lines = (run_dir / "serve.log").read_text(errors="replace").splitlines()
    marker = next((l.strip() for l in reversed(lines) if any(m in l for m in OOM_MARKERS)), None)
    row = CapacityRow(
        engine=args.engine,
        run_id=run_dir.name,
        max_model_len=int(knobs["MAX_MODEL_LEN"]),
        max_num_seqs=int(knobs["MAX_NUM_SEQS"]),
        max_batched_tokens=int(knobs["MAX_BATCHED_TOKENS"]),
        eviction_policy=knobs["EVICTION_POLICY"],
        preemption_policy=knobs["PREEMPTION_POLICY"],
        concurrency=0,
        prompt_tokens=0,
        max_tokens=0,
        requests_sent=0,
        requests_completed=0,
        peak_gpu_mib=None,
        peak_kv_usage=None,
        preemptions=None,
        ttft_p50_s=None,
        ttft_p99_s=None,
        e2e_p50_s=None,
        e2e_p99_s=None,
        accept_length=None,
        outcome="oom_startup",
        detail=marker or (lines[-1].strip() if lines else "no log"),
    )
    append_row(run_dir, row)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("sweep")
    run.add_argument("--engine", choices=("vllm", "sglang"), required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8000")
    run.add_argument("--concurrency", type=int, required=True)
    run.add_argument("--prompt-tokens", type=int, nargs="+", required=True)
    run.add_argument("--max-tokens", type=int, required=True)
    run.add_argument("--request-priority", choices=("none", "index"), required=True)
    failed = sub.add_parser("startup-failed")
    failed.add_argument("--engine", choices=("vllm", "sglang"), required=True)
    failed.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(sweep(args) if args.command == "sweep" else startup_failed(args))


if __name__ == "__main__":
    main()
