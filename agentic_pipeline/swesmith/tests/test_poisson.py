import random
import statistics
from pathlib import Path

import pytest

from poisson import arrivals, mean_session_s, rate_for, simulate_inflight


def interarrivals(offsets: list[float]) -> list[float]:
    return [b - a for a, b in zip([0.0] + offsets[:-1], offsets)]


def max_overlap(spans: list[tuple[float, float]]) -> int:
    return max(sum(1 for s, e in spans if s <= t < e) for t, _ in spans) if spans else 0


def test_arrivals_deterministic_per_seed() -> None:
    assert arrivals(0.5, 50, 7) == arrivals(0.5, 50, 7)
    assert arrivals(0.5, 50, 7) != arrivals(0.5, 50, 8)


def test_arrivals_strictly_increasing() -> None:
    offsets = arrivals(2.0, 1000, 1)
    assert all(b > a for a, b in zip(offsets, offsets[1:]))
    assert offsets[0] > 0


def test_arrivals_mean_interarrival() -> None:
    rate = 0.25
    gaps = interarrivals(arrivals(rate, 20000, 3))
    assert statistics.fmean(gaps) == pytest.approx(1 / rate, rel=0.05)


def test_arrivals_cv_near_one() -> None:
    gaps = interarrivals(arrivals(1.5, 20000, 4))
    assert statistics.pstdev(gaps) / statistics.fmean(gaps) == pytest.approx(1.0, abs=0.05)


def test_arrivals_validation() -> None:
    with pytest.raises(ValueError, match="rate=0.0"):
        arrivals(0.0, 5, 0)
    with pytest.raises(ValueError, match="rate=-1.0"):
        arrivals(-1.0, 5, 0)
    with pytest.raises(ValueError, match="n=-1"):
        arrivals(1.0, -1, 0)
    assert arrivals(1.0, 0, 0) == []


def test_rate_for() -> None:
    assert rate_for(4, 200.0) == pytest.approx(0.02)


def test_mean_session_s(tmp_path: Path) -> None:
    csv_path = tmp_path / "sessions.csv"
    csv_path.write_text("session_id,t_start,t_end\na,10.0,40.0\nb,5.0,25.0\n")
    assert mean_session_s(csv_path) == pytest.approx(25.0)


def test_simulate_inflight_never_exceeds_cap() -> None:
    rng = random.Random(0)
    for _ in range(200):
        n = rng.randint(0, 40)
        cap = rng.randint(1, 6)
        offsets = sorted(rng.uniform(0, 50) for _ in range(n))
        services = [rng.uniform(0.1, 30) for _ in range(n)]
        spans = simulate_inflight(offsets, services, cap)
        assert max_overlap(spans) <= cap
        assert all(s >= a for (s, _), a in zip(spans, offsets))
        assert all(e - s == pytest.approx(d) for (s, e), d in zip(spans, services))
        assert [s for s, _ in spans] == sorted(s for s, _ in spans)


def test_simulate_inflight_cap_ge_n_starts_at_arrival() -> None:
    offsets = [0.5, 1.0, 1.5, 9.0]
    spans = simulate_inflight(offsets, [100.0, 100.0, 100.0, 100.0], 4)
    assert [s for s, _ in spans] == offsets


def test_simulate_inflight_queues_when_full() -> None:
    assert simulate_inflight([0.0, 1.0, 2.0], [10.0, 10.0, 10.0], 1) == [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]


def test_simulate_inflight_validation() -> None:
    with pytest.raises(ValueError, match="cap=0"):
        simulate_inflight([1.0], [1.0], 0)
    with pytest.raises(ValueError, match="2 arrivals vs 1 service"):
        simulate_inflight([1.0, 2.0], [1.0], 1)
