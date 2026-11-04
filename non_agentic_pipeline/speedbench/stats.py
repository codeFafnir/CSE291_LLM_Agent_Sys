"""Percentile and summary-stat helpers shared by the client and the report."""

import math

STAT_NAMES: tuple[str, ...] = ("mean", "p50", "p90", "p95", "p99", "max")


def percentile(sorted_values: list[float], q: float) -> float:
    # Linear interpolation between the two nearest ranks.
    rank = (len(sorted_values) - 1) * q
    low, high = math.floor(rank), math.ceil(rank)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (rank - low)


def stats(prefix: str, values: list[float]) -> dict[str, float | None]:
    if not values:
        return {f"{prefix}_{name}": None for name in STAT_NAMES}
    ordered = sorted(values)
    return {
        f"{prefix}_mean": sum(ordered) / len(ordered),
        f"{prefix}_p50": percentile(ordered, 0.50),
        f"{prefix}_p90": percentile(ordered, 0.90),
        f"{prefix}_p95": percentile(ordered, 0.95),
        f"{prefix}_p99": percentile(ordered, 0.99),
        f"{prefix}_max": ordered[-1],
    }


def present(values: list[float | None]) -> list[float]:
    return [v for v in values if v is not None]
