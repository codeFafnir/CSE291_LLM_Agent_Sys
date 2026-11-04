"""Scrape a vLLM or SGLang /metrics endpoint plus GPU memory into JSONL until stopped.

One line per sample: {"ts", "gpu_used_mib", "series"}. Histograms keep _bucket, _sum and _count.
Counters are cumulative, so summarize_sla.py takes last minus first.
"""

import argparse
import json
import signal
import threading
import time
import urllib.request
from pathlib import Path
from typing import Literal

from prometheus_client.parser import text_string_to_metric_families

from metrics import gpu_used_bytes, mib

Engine = Literal["vllm", "sglang"]

VLLM_SERIES: tuple[str, ...] = (
    "vllm:time_to_first_token_seconds",
    "vllm:request_time_per_output_token_seconds",
    "vllm:e2e_request_latency_seconds",
    "vllm:request_prompt_tokens",
    "vllm:prompt_tokens",
    "vllm:generation_tokens",
    "vllm:request_success",
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens",
    "vllm:spec_decode_num_accepted_tokens_per_pos",
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_hits",
    "vllm:kv_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_preemptions",
    "vllm:request_queue_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
    "vllm:inter_token_latency_seconds",
)

SGLANG_SERIES: tuple[str, ...] = (
    "sglang:time_to_first_token_seconds",
    "sglang:inter_token_latency_seconds",
    "sglang:e2e_request_latency_seconds",
    "sglang:prompt_tokens_total",
    "sglang:generation_tokens_total",
    "sglang:num_requests_total",
    "sglang:spec_accept_length",
    "sglang:spec_accept_rate",
    "sglang:cache_hit_rate",
    "sglang:token_usage",
    "sglang:num_running_reqs",
    "sglang:num_queue_reqs",
    "sglang:num_retracted_requests_total",
    "sglang:queue_time_seconds",
)

SERIES_BY_ENGINE: dict[Engine, tuple[str, ...]] = {"vllm": VLLM_SERIES, "sglang": SGLANG_SERIES}
# Labels that split one metric into distinct series. Everything else is summed away.
KEPT_LABELS: tuple[str, ...] = ("le", "position")


def series_key(name: str, labels: dict[str, str]) -> str:
    kept = ",".join(f"{label}={labels[label]}" for label in KEPT_LABELS if label in labels)
    return f"{name}{{{kept}}}" if kept else name


def parse_series(text: str, wanted: tuple[str, ...]) -> dict[str, float]:
    series: dict[str, float] = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            # Counters expose a _total suffix on samples, histograms _bucket/_sum/_count.
            base = sample.name.removesuffix("_total").removesuffix("_bucket").removesuffix("_sum").removesuffix("_count")
            if sample.name not in wanted and base not in wanted and f"{base}_total" not in wanted:
                continue
            key = series_key(sample.name, sample.labels)
            series[key] = series.get(key, 0.0) + sample.value
    return series


def scrape(url: str, timeout_s: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return response.read().decode("utf-8")


def sample(url: str, engine: Engine, timeout_s: float) -> dict[str, object]:
    return {
        "ts": time.time(),
        "gpu_used_mib": mib(gpu_used_bytes()),
        "series": parse_series(scrape(url, timeout_s), SERIES_BY_ENGINE[engine]),
    }


def collect(url: str, engine: Engine, out: Path, interval_s: float, timeout_s: float, stop: threading.Event) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as handle:
        while not stop.is_set():
            handle.write(json.dumps(sample(url, engine, timeout_s)) + "\n")
            handle.flush()
            stop.wait(interval_s)
        # Final sample so counters cover the whole job.
        handle.write(json.dumps(sample(url, engine, timeout_s)) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("vllm", "sglang"), required=True)
    parser.add_argument("--url", required=True, help="e.g. http://127.0.0.1:8000/metrics")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval-s", type=float, required=True)
    parser.add_argument("--timeout-s", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    collect(args.url, args.engine, args.out, args.interval_s, args.timeout_s, stop)


if __name__ == "__main__":
    main()
