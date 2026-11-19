"""Poisson session arrivals and the pilot-based rate rule lambda = C / S."""

import argparse
import csv
import heapq
import random
import statistics
from pathlib import Path


def arrivals(rate: float, n: int, seed: int) -> list[float]:
    """Arrival offsets in seconds: cumulative exponential interarrivals with mean 1 / rate."""
    if rate <= 0:
        raise ValueError(f"rate must be > 0, got rate={rate}")
    if n < 0:
        raise ValueError(f"n must be >= 0, got n={n}")
    rng = random.Random(seed)
    times: list[float] = []
    now = 0.0
    for _ in range(n):
        now += rng.expovariate(rate)
        times.append(now)
    return times


def simulate_inflight(arrival_offsets: list[float], service_times: list[float], cap: int) -> list[tuple[float, float]]:
    """(start, end) per session under FIFO admission while fewer than `cap` sessions are in flight."""
    if cap < 1:
        raise ValueError(f"cap must be >= 1, got cap={cap}")
    if len(arrival_offsets) != len(service_times):
        raise ValueError(f"length mismatch: {len(arrival_offsets)} arrivals vs {len(service_times)} service times")
    ends: list[float] = []
    spans: list[tuple[float, float]] = []
    prev_start = 0.0
    for arrival, service in zip(arrival_offsets, service_times):
        start = max(arrival, prev_start)
        if len(ends) >= cap:
            start = max(start, heapq.heappop(ends))
        end = start + service
        heapq.heappush(ends, end)
        spans.append((start, end))
        prev_start = start
    return spans


def rate_for(concurrency: int, mean_session_s: float) -> float:
    return concurrency / mean_session_s


def mean_session_s(sessions_csv: Path) -> float:
    with sessions_csv.open() as handle:
        rows = list(csv.DictReader(handle))
    return statistics.fmean(float(r["t_end"]) - float(r["t_start"]) for r in rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print lambda = C / S from a c1 pilot's sessions.csv")
    parser.add_argument("--sessions-csv", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mean_s = mean_session_s(args.sessions_csv)
    print(f"mean_session_s={mean_s:.2f} rate_per_s={rate_for(args.concurrency, mean_s):.6f}")


if __name__ == "__main__":
    main()
