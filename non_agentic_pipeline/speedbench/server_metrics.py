"""vLLM /metrics scraping: parser, background collector, and per-request counter deltas.

  server_metrics.py collect --url http://127.0.0.1:8000/metrics --out <jsonl> --interval-s 1 --timeout-s 5

One JSONL line per sample: {"ts", "gpu_used_mib", "series"}. Counters are cumulative, so consumers take differences.
Histograms keep only _sum and _count (exact percentiles come from client-side timings).
"""

import argparse
import signal
import threading
import time
import urllib.request
from pathlib import Path

from prometheus_client.parser import text_string_to_metric_families

from gpu import gpu_used_bytes, mib
from models import Sample, ServerDelta, SpecDelta

SPEC_POSITIONS = 3

RUNNING = "vllm:num_requests_running"
WAITING = "vllm:num_requests_waiting"
KV_USAGE = "vllm:kv_cache_usage_perc"
PREEMPTIONS = "vllm:num_preemptions_total"
PREFIX_QUERIES = "vllm:prefix_cache_queries_total"
PREFIX_HITS = "vllm:prefix_cache_hits_total"
PROMPT_TOKENS = "vllm:prompt_tokens_total"
GENERATION_TOKENS = "vllm:generation_tokens_total"
SUCCESS = "vllm:request_success_total"
SPEC_DRAFTS = "vllm:spec_decode_num_drafts_total"
SPEC_DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens_total"
SPEC_ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
SPEC_PER_POS_PREFIX = "vllm:spec_decode_num_accepted_tokens_per_pos_total{position="
QUEUE_HIST = "vllm:request_queue_time_seconds"
PREFILL_HIST = "vllm:request_prefill_time_seconds"
DECODE_HIST = "vllm:request_decode_time_seconds"

WANTED: frozenset[str] = frozenset(
    {
        "vllm:time_to_first_token_seconds",
        "vllm:request_time_per_output_token_seconds",
        "vllm:e2e_request_latency_seconds",
        "vllm:inter_token_latency_seconds",
        QUEUE_HIST,
        PREFILL_HIST,
        DECODE_HIST,
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
    }
)
SUFFIXES: tuple[str, ...] = ("_total", "_bucket", "_sum", "_count")
# Labels that split one metric into distinct series. Everything else is summed away.
KEPT_LABELS: tuple[str, ...] = ("position",)


def base_name(sample_name: str) -> str:
    for suffix in SUFFIXES:
        if sample_name.endswith(suffix):
            return sample_name[: -len(suffix)]
    return sample_name


def series_key(name: str, labels: dict[str, str]) -> str:
    kept = ",".join(f"{label}={labels[label]}" for label in KEPT_LABELS if label in labels)
    return f"{name}{{{kept}}}" if kept else name


def parse_series(text: str) -> dict[str, float]:
    series: dict[str, float] = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name.endswith("_bucket") or base_name(sample.name) not in WANTED:
                continue
            key = series_key(sample.name, sample.labels)
            series[key] = series.get(key, 0.0) + sample.value
    return series


def scrape_series(url: str, timeout_s: float) -> dict[str, float]:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return parse_series(response.read().decode("utf-8"))


def wait_recorded(url: str, success_before: float, timeout_s: float, poll_s: float, scrape_timeout_s: float) -> dict[str, float]:
    # vLLM records a finished request's stats just after the last chunk is sent, so poll until the counter moves.
    deadline = time.monotonic() + timeout_s
    while True:
        series = scrape_series(url, scrape_timeout_s)
        if series.get(SUCCESS, 0.0) > success_before:
            return series
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"{SUCCESS} stayed at {success_before} for {timeout_s}s after the response finished ({url})"
            )
        time.sleep(poll_s)


def delta(before: dict[str, float], after: dict[str, float], key: str) -> float:
    return after.get(key, 0.0) - before.get(key, 0.0)


def mean_delta(before: dict[str, float], after: dict[str, float], hist: str) -> float | None:
    count = delta(before, after, f"{hist}_count")
    return delta(before, after, f"{hist}_sum") / count if count > 0.0 else None


def spec_delta(before: dict[str, float], after: dict[str, float]) -> SpecDelta:
    per_position = [delta(before, after, f"{SPEC_PER_POS_PREFIX}{i}}}") for i in range(SPEC_POSITIONS)]
    return SpecDelta(
        drafts=delta(before, after, SPEC_DRAFTS),
        draft_tokens=delta(before, after, SPEC_DRAFT_TOKENS),
        accepted=delta(before, after, SPEC_ACCEPTED),
        accepted_per_position=per_position,
    )


def server_delta(before: dict[str, float], after: dict[str, float]) -> ServerDelta:
    return ServerDelta(
        queue_s=mean_delta(before, after, QUEUE_HIST),
        prefill_s=mean_delta(before, after, PREFILL_HIST),
        decode_s=mean_delta(before, after, DECODE_HIST),
        spec=spec_delta(before, after),
    )


def accept_length(spec: SpecDelta) -> float | None:
    return 1.0 + spec.accepted / spec.drafts if spec.drafts > 0.0 else None


def accept_rate(spec: SpecDelta) -> float | None:
    return spec.accepted / spec.draft_tokens if spec.draft_tokens > 0.0 else None


def accept_per_position(spec: SpecDelta) -> list[float | None]:
    return [count / spec.drafts if spec.drafts > 0.0 else None for count in spec.accepted_per_position]


def sample(url: str, timeout_s: float) -> Sample:
    return Sample(ts=time.time(), gpu_used_mib=mib(gpu_used_bytes()), series=scrape_series(url, timeout_s))


def collect(url: str, out: Path, interval_s: float, timeout_s: float, stop: threading.Event) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as handle:
        while not stop.is_set():
            handle.write(sample(url, timeout_s).model_dump_json() + "\n")
            handle.flush()
            stop.wait(interval_s)
        # Final sample so counters cover the whole job.
        handle.write(sample(url, timeout_s).model_dump_json() + "\n")


def load_samples(path: Path) -> list[Sample]:
    return [Sample.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collector = sub.add_parser("collect")
    collector.add_argument("--url", required=True)
    collector.add_argument("--out", type=Path, required=True)
    collector.add_argument("--interval-s", type=float, required=True)
    collector.add_argument("--timeout-s", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    collect(args.url, args.out, args.interval_s, args.timeout_s, stop)


if __name__ == "__main__":
    main()
