"""Analysis outputs for one SWE-smith run dir: tool gaps, 1 Hz joined timeseries, derived metrics, Perfetto timeline, dashboard.

  report.py --run-dir DIR [--window-start EPOCH] [--kv-pool-gb GB]

Reads proxy_requests.jsonl, tool_spans.jsonl, sessions.jsonl, server_metrics.jsonl and gpu_dmon.log; writes into DIR/report/.
proxy_requests, tool_spans and sessions are optional (missing = empty); zero requests sets zero_requests in derived.json and warns on stderr.
gpu_dmon.log has no timestamps and rows are ~1.05 s apart: row 0 maps to the first server_metrics ts (or --window-start),
the last row to the last server_metrics ts, and rows in between linearly.
Needs agentic_pipeline/bench and agentic_pipeline/swesmith on PYTHONPATH (flat sibling imports).
A session is "active" from its first proxy request send to its last request end; it is "in a tool gap" between a request end
and its next request send. all_in_gap means every active session is in a gap (nothing is decoding).
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Literal, TypeVar

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import BaseModel, ConfigDict

from request_records import GAUGES, US, Gauges, RequestRecord, attach_peaks, perfetto_trace
from run import MAX_TOOL_TIMEOUT_S, Session
from run_csv import dmon_columns, percentile
from session_proxy import RequestLog
from summarize_sla import Sample, delta, load_samples
from tool_span import ToolSpan

ModelT = TypeVar("ModelT", bound=BaseModel)
Interval = tuple[float, float]
GapKind = Literal["quantiles", "histogram", "prefix_hit"]
SegmentKind = Literal["session", "llm", "tool", "gap"]

GAP_EDGES_S: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 15.0, 30.0)
BUCKET_LABELS: tuple[str, ...] = ("<1s", "1-2s", "2-5s", "5-10s", "10-15s", "15-30s", ">=30s")
ALL = "all"
NO_SPAN = "no_span"
SAMPLE_PERIOD_S = 1.0
SESSIONS_PID = 3
PREFIX_QUERIES_KEY = "vllm:prefix_cache_queries_total"
SPEC_ACCEPTED_KEY = "vllm:spec_decode_num_accepted_tokens_total"
SPEC_DRAFTED_KEY = "vllm:spec_decode_num_draft_tokens_total"
SEGMENT_COLORS: dict[SegmentKind, str] = {"session": "#dddddd", "gap": "#f4b183", "tool": "#70ad47", "llm": "#4472c4"}


class ReportArgs(BaseModel):
    model_config = ConfigDict(frozen=True)
    run_dir: Path
    window_start: float | None
    kv_pool_gb: float | None


class FailedRequestRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    request_id: str
    session_id: str
    turn: int
    status: int
    error: str | None
    t_send: float
    t_end: float
    e2e_s: float
    request_bytes: int
    max_tokens: int | None


class GapRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    session_id: str
    prev_turn: int
    next_turn: int
    t_prev_end: float
    t_next_send: float
    gap_s: float
    tool: str | None
    tool_step: int | None
    tool_duration_s: float | None
    tool_timed_out: bool | None
    tool_timeout_violation: bool
    next_prompt_tokens: int | None
    next_cached_tokens: int | None
    prefix_hit_ratio: float | None


class GapStatRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: GapKind
    tool: str
    bucket: str
    n: int
    gap_mean_s: float | None
    gap_p50_s: float | None
    gap_p90_s: float | None
    gap_p99_s: float | None
    gap_max_s: float | None
    prefix_hit_mean: float | None
    prefix_hit_p50: float | None


class DmonSeries(BaseModel):
    model_config = ConfigDict(frozen=True)
    sm: list[float]
    fb: list[float]
    pwr: list[float]


class SessionWindows(BaseModel):
    model_config = ConfigDict(frozen=True)
    requests: list[Interval]
    gaps: list[Interval]
    active: list[Interval]


class TimeseriesRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    ts: float
    gpu_sm_pct: float | None
    gpu_fb_mib: float | None
    gpu_power_w: float | None
    running: float
    waiting: float
    kv_usage: float
    prefix_hit_rate_delta: float | None
    preemptions_delta: float
    gen_tokens_per_s: float
    spec_accepted_delta: float
    spec_drafted_delta: float
    continuum_pins_delta: float
    continuum_unpins_ttl_delta: float
    continuum_unpins_next_turn_delta: float
    continuum_unpins_pressure_delta: float
    continuum_pinned_blocks: float
    n_sessions_active: int
    n_in_tool_gap: int
    all_in_gap: bool
    n_decoding: int


class DerivedMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)
    zero_requests: bool
    n_sessions: int
    n_requests: int
    n_failed_requests: int
    n_gaps: int
    n_samples: int
    n_samples_all_in_gap: int
    n_samples_decoding: int
    mean_sm_pct_all_in_gap: float | None
    mean_sm_pct_any_decoding: float | None
    mean_kv_usage: float | None
    mean_kv_usage_any_gap: float | None
    mean_kv_usage_all_in_gap: float | None
    mean_pinned_blocks_any_gap: float | None
    mean_pinned_blocks_all_in_gap: float | None
    output_tokens_per_s: float
    goodput_tok_s_per_kv_usage: float | None
    kv_pool_gb: float | None
    goodput_tok_s_per_gb_kv: float | None
    n_span_records: int
    n_tool_spans: int
    n_timed_out: int
    timeout_rate: float | None
    timeout_cap_s_observed: float | None
    tool_timeout_violations: int


class TimeoutStats(BaseModel):
    model_config = ConfigDict(frozen=True)
    n_tool_spans: int
    n_timed_out: int
    timeout_rate: float | None
    timeout_cap_s_observed: float | None
    tool_timeout_violations: int


class SessionRow(Session):
    n_requests: int
    prompt_tokens_total: int
    completion_tokens_total: int
    cached_tokens_total: int | None


class GanttSegment(BaseModel):
    model_config = ConfigDict(frozen=True)
    session_id: str
    kind: SegmentKind
    start: float
    end: float
    label: str


class RunInputs(BaseModel):
    model_config = ConfigDict(frozen=True)
    requests: list[RequestLog]
    spans: list[ToolSpan]
    sessions: list[Session]
    samples: list[Sample]
    dmon: DmonSeries


def read_models(path: Path, model: type[ModelT]) -> list[ModelT]:
    return [model.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def load_dmon(path: Path) -> DmonSeries:
    columns = dmon_columns(path, ("sm", "fb", "pwr"))
    lengths = {name: len(values) for name, values in columns.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"gpu_dmon.log columns differ in length (a '-' cell breaks 1 Hz alignment): {path} {lengths}")
    return DmonSeries(sm=columns["sm"], fb=columns["fb"], pwr=columns["pwr"])


def read_optional_models(path: Path, model: type[ModelT]) -> list[ModelT]:
    return read_models(path, model) if path.exists() else []


def load_inputs(run_dir: Path) -> RunInputs:
    samples = load_samples(run_dir / "server_metrics.jsonl")
    if not samples:
        raise ValueError(f"no samples in {run_dir / 'server_metrics.jsonl'}")
    return RunInputs(
        requests=read_optional_models(run_dir / "proxy_requests.jsonl", RequestLog),
        spans=read_optional_models(run_dir / "tool_spans.jsonl", ToolSpan),
        sessions=read_optional_models(run_dir / "sessions.jsonl", Session),
        samples=samples,
        dmon=load_dmon(run_dir / "gpu_dmon.log"),
    )


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def by_session(requests: list[RequestLog]) -> dict[str, list[RequestLog]]:
    grouped: dict[str, list[RequestLog]] = {}
    for request in sorted(requests, key=lambda r: r.t_send):
        grouped.setdefault(request.session_id, []).append(request)
    return grouped


def is_success(request: RequestLog) -> bool:
    return request.status < 400 and request.prompt_tokens is not None and request.completion_tokens is not None


def record_of(r: RequestLog) -> RequestRecord:
    if r.prompt_tokens is None or r.completion_tokens is None:
        raise ValueError(f"successful request without usage: {r.request_id}")
    return RequestRecord(
        request_id=r.request_id,
        trial=r.session_id,
        arrival_ts=r.t_send,
        queue_s=None,
        prefill_s=None,
        ttft_s=None,
        decode_s=None,
        tpot_s=None,
        e2e_s=r.t_end - r.t_send,
        prompt_tokens=r.prompt_tokens,
        completion_tokens=r.completion_tokens,
        cached_tokens=r.cached_tokens,
        peak_gpu_mib_during=None,
        peak_kv_usage_during=None,
    )


def failed_row_of(r: RequestLog) -> FailedRequestRow:
    return FailedRequestRow(
        request_id=r.request_id,
        session_id=r.session_id,
        turn=r.turn,
        status=r.status,
        error=r.error if r.error is not None else "missing usage in response",
        t_send=r.t_send,
        t_end=r.t_end,
        e2e_s=r.t_end - r.t_send,
        request_bytes=r.request_bytes,
        max_tokens=r.max_tokens,
    )


def split_requests(requests: list[RequestLog]) -> tuple[list[RequestRecord], list[FailedRequestRow]]:
    records = [record_of(r) for r in requests if is_success(r)]
    failed = [failed_row_of(r) for r in requests if not is_success(r)]
    return records, failed


def tool_interval(span: ToolSpan) -> Interval | None:
    if not span.executed or span.t_tool_start is None:
        return None
    end = span.t_tool_end if span.t_tool_end is not None else span.t_tool_start + span.execution_time_s
    return span.t_tool_start, end


def tool_duration_s(span: ToolSpan) -> float:
    if span.t_tool_start is not None and span.t_tool_end is not None:
        return span.t_tool_end - span.t_tool_start
    return span.execution_time_s


def is_timeout_violation(span: ToolSpan) -> bool:
    # A timed-out step runs slightly past the cap by design; only a non-timed-out executed tool over the cap violates it.
    return span.executed and not span.timed_out and tool_duration_s(span) > MAX_TOOL_TIMEOUT_S


def timeout_stats(spans: list[ToolSpan]) -> TimeoutStats:
    # Counts every executed span, so it can exceed the tool_gaps violations (a tool with no following request has no gap row).
    executed = [s for s in spans if s.executed]
    timed_out = [s for s in executed if s.timed_out]
    return TimeoutStats(
        n_tool_spans=len(executed),
        n_timed_out=len(timed_out),
        timeout_rate=len(timed_out) / len(executed) if executed else None,
        timeout_cap_s_observed=max((tool_duration_s(s) for s in timed_out), default=None),
        tool_timeout_violations=sum(1 for s in executed if is_timeout_violation(s)),
    )


def overlap_s(a: Interval, b: Interval) -> float:
    return min(a[1], b[1]) - max(a[0], b[0])


def matching_span(spans: list[ToolSpan], session_id: str, gap: Interval) -> ToolSpan | None:
    candidates: list[tuple[float, ToolSpan]] = []
    for span in spans:
        interval = tool_interval(span)
        if span.session_id == session_id and interval is not None and overlap_s(interval, gap) > 0:
            candidates.append((overlap_s(interval, gap), span))
    return max(candidates, key=lambda c: c[0])[1] if candidates else None


def prefix_ratio(request: RequestLog) -> float | None:
    if request.cached_tokens is None or request.prompt_tokens is None or request.prompt_tokens <= 0:
        return None
    return request.cached_tokens / request.prompt_tokens


def tool_gap_rows(requests: list[RequestLog], spans: list[ToolSpan]) -> list[GapRow]:
    rows: list[GapRow] = []
    for session_id, reqs in by_session(requests).items():
        for prev, nxt in zip(reqs, reqs[1:]):
            gap = (prev.t_end, nxt.t_send)
            span = matching_span(spans, session_id, gap)
            rows.append(
                GapRow(
                    session_id=session_id,
                    prev_turn=prev.turn,
                    next_turn=nxt.turn,
                    t_prev_end=prev.t_end,
                    t_next_send=nxt.t_send,
                    gap_s=nxt.t_send - prev.t_end,
                    tool=span.tool if span else None,
                    tool_step=span.step if span else None,
                    tool_duration_s=tool_duration_s(span) if span else None,
                    tool_timed_out=span.timed_out if span else None,
                    tool_timeout_violation=span is not None and is_timeout_violation(span),
                    next_prompt_tokens=nxt.prompt_tokens,
                    next_cached_tokens=nxt.cached_tokens,
                    prefix_hit_ratio=prefix_ratio(nxt),
                )
            )
    return rows


def bucket_of(gap_s: float) -> str:
    for edge, label in zip(GAP_EDGES_S, BUCKET_LABELS):
        if gap_s < edge:
            return label
    return BUCKET_LABELS[-1]


def quantile_stats(values: list[float]) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    if not values:
        return None, None, None, None, None
    ordered = sorted(values)
    return sum(ordered) / len(ordered), percentile(ordered, 0.50), percentile(ordered, 0.90), percentile(ordered, 0.99), ordered[-1]


def stat_row(kind: GapKind, tool: str, bucket: str, gaps: list[float], ratios: list[float]) -> GapStatRow:
    g_mean, g50, g90, g99, g_max = quantile_stats(gaps)
    ordered = sorted(ratios)
    return GapStatRow(
        kind=kind,
        tool=tool,
        bucket=bucket,
        n=len(ratios) if kind == "prefix_hit" else len(gaps),
        gap_mean_s=g_mean if kind == "quantiles" else None,
        gap_p50_s=g50 if kind == "quantiles" else None,
        gap_p90_s=g90 if kind == "quantiles" else None,
        gap_p99_s=g99 if kind == "quantiles" else None,
        gap_max_s=g_max if kind == "quantiles" else None,
        prefix_hit_mean=mean(ordered) if kind == "prefix_hit" else None,
        prefix_hit_p50=percentile(ordered, 0.50) if kind == "prefix_hit" and ordered else None,
    )


def gap_stat_rows(gaps: list[GapRow]) -> list[GapStatRow]:
    tools = [ALL] + sorted({g.tool if g.tool is not None else NO_SPAN for g in gaps})
    rows: list[GapStatRow] = []
    for tool in tools:
        chosen = [g for g in gaps if tool == ALL or (g.tool if g.tool is not None else NO_SPAN) == tool]
        rows.append(stat_row("quantiles", tool, ALL, [g.gap_s for g in chosen], []))
        for label in BUCKET_LABELS:
            rows.append(stat_row("histogram", tool, label, [g.gap_s for g in chosen if bucket_of(g.gap_s) == label], []))
    for label in BUCKET_LABELS:
        in_bucket = [g for g in gaps if bucket_of(g.gap_s) == label]
        ratios = [g.prefix_hit_ratio for g in in_bucket if g.prefix_hit_ratio is not None]
        rows.append(stat_row("prefix_hit", ALL, label, [g.gap_s for g in in_bucket], ratios))
    return rows


def session_windows(requests: list[RequestLog]) -> SessionWindows:
    request_windows: list[Interval] = [(r.t_send, r.t_end) for r in requests]
    gap_windows: list[Interval] = []
    active: list[Interval] = []
    for reqs in by_session(requests).values():
        active.append((reqs[0].t_send, max(r.t_end for r in reqs)))
        gap_windows.extend((prev.t_end, nxt.t_send) for prev, nxt in zip(reqs, reqs[1:]) if nxt.t_send > prev.t_end)
    return SessionWindows(requests=request_windows, gaps=gap_windows, active=active)


def count_at(intervals: list[Interval], ts: float) -> int:
    return sum(1 for start, end in intervals if start <= ts < end)


def gauge_value(sample: Sample, key: str | None) -> float:
    return sample.series.get(key, 0.0) if key is not None else 0.0


def dmon_at(values: list[float], ts: float, first_ts: float, last_ts: float) -> float | None:
    # Row 0 sits at first_ts and the last row at last_ts; rows are evenly spaced between them.
    if not values:
        return None
    # A window under one period cannot be scaled (rows are ~1 s apart): keep row 0 at first_ts at the native period.
    span = last_ts - first_ts
    step = span / (len(values) - 1) if len(values) > 1 and span >= SAMPLE_PERIOD_S else SAMPLE_PERIOD_S
    index = math.floor((ts - first_ts) / step + 0.5)
    return values[index] if 0 <= index < len(values) else None


def counter_delta(prev: Sample, cur: Sample, key: str | None) -> float:
    # Clipped at 0 so a counter reset yields one empty sample instead of a negative one.
    return max(delta(prev, cur, key), 0.0) if key is not None else 0.0


def counter_total(samples: list[Sample], key: str | None) -> float:
    return sum(counter_delta(prev, cur, key) for prev, cur in zip(samples, samples[1:]))


def hit_rate_delta(prev: Sample, cur: Sample, gauges: Gauges) -> float | None:
    queries = counter_delta(prev, cur, PREFIX_QUERIES_KEY)
    if queries <= 0 or gauges.prefix_hits_total is None:
        return None
    return counter_delta(prev, cur, gauges.prefix_hits_total) / queries


def timeseries_rows(
    samples: list[Sample], dmon: DmonSeries, window_start: float, windows: SessionWindows, gauges: Gauges
) -> list[TimeseriesRow]:
    last_ts = samples[-1].ts
    rows: list[TimeseriesRow] = []
    for i, cur in enumerate(samples):
        prev = samples[max(i - 1, 0)]
        dt = cur.ts - prev.ts
        active = count_at(windows.active, cur.ts)
        in_gap = count_at(windows.gaps, cur.ts)
        rows.append(
            TimeseriesRow(
                ts=cur.ts,
                gpu_sm_pct=dmon_at(dmon.sm, cur.ts, window_start, last_ts),
                gpu_fb_mib=dmon_at(dmon.fb, cur.ts, window_start, last_ts),
                gpu_power_w=dmon_at(dmon.pwr, cur.ts, window_start, last_ts),
                running=gauge_value(cur, gauges.running),
                waiting=gauge_value(cur, gauges.waiting),
                kv_usage=gauge_value(cur, gauges.kv_usage),
                prefix_hit_rate_delta=hit_rate_delta(prev, cur, gauges),
                preemptions_delta=counter_delta(prev, cur, gauges.preemptions_total),
                gen_tokens_per_s=counter_delta(prev, cur, gauges.generation_tokens_total) / dt if dt > 0 else 0.0,
                spec_accepted_delta=counter_delta(prev, cur, SPEC_ACCEPTED_KEY),
                spec_drafted_delta=counter_delta(prev, cur, SPEC_DRAFTED_KEY),
                continuum_pins_delta=counter_delta(prev, cur, gauges.continuum_pins_total),
                continuum_unpins_ttl_delta=counter_delta(prev, cur, gauges.continuum_unpins_ttl_total),
                continuum_unpins_next_turn_delta=counter_delta(prev, cur, gauges.continuum_unpins_next_turn_total),
                continuum_unpins_pressure_delta=counter_delta(prev, cur, gauges.continuum_unpins_pressure_total),
                continuum_pinned_blocks=gauge_value(cur, gauges.continuum_pinned_blocks),
                n_sessions_active=active,
                n_in_tool_gap=in_gap,
                all_in_gap=active > 0 and in_gap == active,
                n_decoding=count_at(windows.requests, cur.ts),
            )
        )
    return rows


def present(values: list[float | None]) -> list[float]:
    return [v for v in values if v is not None]


def derived_metrics(
    rows: list[TimeseriesRow],
    samples: list[Sample],
    gaps: list[GapRow],
    spans: list[ToolSpan],
    sessions: list[Session],
    requests: list[RequestLog],
    gauges: Gauges,
    kv_pool_gb: float | None,
) -> DerivedMetrics:
    all_gap = [r for r in rows if r.all_in_gap]
    any_gap = [r for r in rows if r.n_in_tool_gap > 0]
    decoding = [r for r in rows if r.n_decoding > 0]
    wall = samples[-1].ts - samples[0].ts
    tok_s = counter_total(samples, gauges.generation_tokens_total) / wall if wall > 0 else 0.0
    mean_kv = mean([r.kv_usage for r in rows])
    timeouts = timeout_stats(spans)
    return DerivedMetrics(
        zero_requests=not requests,
        n_sessions=len(sessions),
        n_requests=len(requests),
        n_failed_requests=sum(1 for r in requests if not is_success(r)),
        n_gaps=len(gaps),
        n_samples=len(rows),
        n_samples_all_in_gap=len(all_gap),
        n_samples_decoding=len(decoding),
        mean_sm_pct_all_in_gap=mean(present([r.gpu_sm_pct for r in all_gap])),
        mean_sm_pct_any_decoding=mean(present([r.gpu_sm_pct for r in decoding])),
        mean_kv_usage=mean_kv,
        mean_kv_usage_any_gap=mean([r.kv_usage for r in any_gap]),
        mean_kv_usage_all_in_gap=mean([r.kv_usage for r in all_gap]),
        mean_pinned_blocks_any_gap=mean([r.continuum_pinned_blocks for r in any_gap]),
        mean_pinned_blocks_all_in_gap=mean([r.continuum_pinned_blocks for r in all_gap]),
        output_tokens_per_s=tok_s,
        goodput_tok_s_per_kv_usage=tok_s / mean_kv if mean_kv else None,
        kv_pool_gb=kv_pool_gb,
        goodput_tok_s_per_gb_kv=tok_s / (mean_kv * kv_pool_gb) if mean_kv and kv_pool_gb else None,
        n_span_records=len(spans),
        n_tool_spans=timeouts.n_tool_spans,
        n_timed_out=timeouts.n_timed_out,
        timeout_rate=timeouts.timeout_rate,
        timeout_cap_s_observed=timeouts.timeout_cap_s_observed,
        tool_timeout_violations=timeouts.tool_timeout_violations,
    )


def cached_total(reqs: list[RequestLog]) -> int | None:
    # None when no request reported cached_tokens, so a server without cache details is not read as 0% hits.
    vals = [r.cached_tokens for r in reqs if r.cached_tokens is not None]
    return sum(vals) if vals else None


def session_rows(sessions: list[Session], requests: list[RequestLog]) -> list[SessionRow]:
    grouped = by_session(requests)
    rows: list[SessionRow] = []
    for s in sessions:
        reqs = grouped.get(s.instance_id, [])
        rows.append(
            SessionRow(
                **s.model_dump(),
                n_requests=len(reqs),
                prompt_tokens_total=sum(r.prompt_tokens or 0 for r in reqs),
                completion_tokens_total=sum(r.completion_tokens or 0 for r in reqs),
                cached_tokens_total=cached_total(reqs),
            )
        )
    return rows


def time_origin(samples: list[Sample], records: list[RequestRecord]) -> float:
    # Same t0 as request_records.perfetto_trace so the appended tracks line up.
    return min([r.arrival_ts for r in records] + [s.ts for s in samples])


def session_order(requests: list[RequestLog]) -> list[str]:
    return list(by_session(requests))


def session_slice(name: str, tid: int, start: float, end: float, t0: float, args: dict[str, str | int | float | bool | None]) -> dict[str, object]:
    return {"ph": "X", "pid": SESSIONS_PID, "tid": tid, "name": name, "ts": (start - t0) * US, "dur": (end - start) * US, "args": args}


def session_events(
    requests: list[RequestLog], spans: list[ToolSpan], sessions: list[Session], gaps: list[GapRow], t0: float
) -> list[dict[str, object]]:
    order = session_order(requests)
    tid_calls = {session_id: 2 * i for i, session_id in enumerate(order)}
    events: list[dict[str, object]] = [{"ph": "M", "pid": SESSIONS_PID, "name": "process_name", "args": {"name": "sessions"}}]
    for session_id, tid in tid_calls.items():
        events.append({"ph": "M", "pid": SESSIONS_PID, "tid": tid, "name": "thread_name", "args": {"name": f"{session_id} llm+tools"}})
        events.append({"ph": "M", "pid": SESSIONS_PID, "tid": tid + 1, "name": "thread_name", "args": {"name": f"{session_id} gaps"}})
    for s in sessions:
        if s.instance_id in tid_calls:
            info: dict[str, str | int | float | bool | None] = {"status": s.status, "exit_status": s.exit_status, "submitted": s.submitted}
            events.append(session_slice(f"session {s.instance_id}", tid_calls[s.instance_id], s.t_start, s.t_end, t0, info))
    for r in requests:
        call_info: dict[str, str | int | float | bool | None] = {
            "request_id": r.request_id,
            "status": r.status,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "cached_tokens": r.cached_tokens,
        }
        events.append(session_slice(f"llm t{r.turn}", tid_calls[r.session_id], r.t_send, r.t_end, t0, call_info))
    for span in spans:
        interval = tool_interval(span)
        if interval is not None and span.session_id in tid_calls:
            tool_info: dict[str, str | int | float | bool | None] = {
                "step": span.step,
                "execution_time_s": span.execution_time_s,
                "timed_out": span.timed_out,
                "tool_timeout_violation": is_timeout_violation(span),
                "action_head": span.action_head,
            }
            events.append(session_slice(span.tool, tid_calls[span.session_id], interval[0], interval[1], t0, tool_info))
    for g in gaps:
        gap_info: dict[str, str | int | float | bool | None] = {"gap_s": g.gap_s, "tool": g.tool, "prefix_hit_ratio": g.prefix_hit_ratio}
        events.append(session_slice("gap", tid_calls[g.session_id] + 1, g.t_prev_end, g.t_next_send, t0, gap_info))
    return events


def timeline_trace(
    records: list[RequestRecord], inputs: RunInputs, gaps: list[GapRow], gauges: Gauges
) -> dict[str, list[dict[str, object]]]:
    base = perfetto_trace(records, inputs.samples, gauges)
    t0 = time_origin(inputs.samples, records)
    extra = session_events(inputs.requests, inputs.spans, inputs.sessions, gaps, t0)
    return {"traceEvents": base["traceEvents"] + extra}


def gantt_segments(
    requests: list[RequestLog], spans: list[ToolSpan], sessions: list[Session], gaps: list[GapRow]
) -> list[GanttSegment]:
    segments = [GanttSegment(session_id=s.instance_id, kind="session", start=s.t_start, end=s.t_end, label=s.status) for s in sessions]
    segments += [GanttSegment(session_id=r.session_id, kind="llm", start=r.t_send, end=r.t_end, label=f"llm t{r.turn} status={r.status}") for r in requests]
    for span in spans:
        interval = tool_interval(span)
        if interval is not None:
            segments.append(GanttSegment(session_id=span.session_id, kind="tool", start=interval[0], end=interval[1], label=f"{span.tool} {span.execution_time_s:.1f}s"))
    segments += [GanttSegment(session_id=g.session_id, kind="gap", start=g.t_prev_end, end=g.t_next_send, label=f"gap {g.gap_s:.1f}s tool={g.tool}") for g in gaps]
    return segments


def gap_runs(rows: list[TimeseriesRow]) -> list[Interval]:
    runs: list[Interval] = []
    start: float | None = None
    for row in rows:
        if row.all_in_gap and start is None:
            start = row.ts
        if start is not None and not row.all_in_gap:
            runs.append((start, row.ts))
            start = None
    if start is not None:
        runs.append((start, rows[-1].ts + SAMPLE_PERIOD_S))
    return runs


def timeseries_figure(rows: list[TimeseriesRow], t0: float) -> go.Figure:
    x = [r.ts - t0 for r in rows]
    panels: list[tuple[str, list[tuple[str, list[float | None]]]]] = [
        ("GPU SM %", [("sm %", [r.gpu_sm_pct for r in rows])]),
        ("GPU memory (MiB)", [("fb MiB", [r.gpu_fb_mib for r in rows])]),
        ("KV cache usage %", [("kv %", [r.kv_usage * 100 for r in rows])]),
        ("Requests", [("running", [r.running for r in rows]), ("waiting", [r.waiting for r in rows])]),
        ("Continuum pinned blocks", [("pinned", [r.continuum_pinned_blocks for r in rows])]),
        ("Generation tokens/s", [("tok/s", [r.gen_tokens_per_s for r in rows])]),
    ]
    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, subplot_titles=[p[0] for p in panels], vertical_spacing=0.04)
    for row_index, (_, traces) in enumerate(panels, start=1):
        for name, values in traces:
            fig.add_trace(go.Scatter(x=x, y=values, name=name, mode="lines"), row=row_index, col=1)
    for start, end in gap_runs(rows):
        fig.add_vrect(x0=start - t0, x1=end - t0, fillcolor="orange", opacity=0.15, line_width=0, layer="below", row="all", col=1)
    fig.update_xaxes(title_text="seconds from start", row=len(panels), col=1)
    fig.update_layout(height=1500, title="Server timeseries (shaded: all sessions in tool gap)")
    return fig


def gantt_figure(segments: list[GanttSegment], order: list[str], t0: float) -> go.Figure:
    fig = go.Figure()
    for kind, color in SEGMENT_COLORS.items():
        chosen = [s for s in segments if s.kind == kind]
        fig.add_trace(
            go.Bar(
                orientation="h",
                name=kind,
                y=[s.session_id for s in chosen],
                x=[s.end - s.start for s in chosen],
                base=[s.start - t0 for s in chosen],
                marker_color=color,
                hovertext=[s.label for s in chosen],
            )
        )
    fig.update_layout(
        barmode="overlay",
        title="Per-session timeline (llm / tool / gap)",
        xaxis_title="seconds from start",
        yaxis={"categoryorder": "array", "categoryarray": order, "autorange": "reversed"},
        height=max(300, 60 * len(order) + 150),
    )
    return fig


def dashboard_html(rows: list[TimeseriesRow], segments: list[GanttSegment], order: list[str], t0: float) -> str:
    top = timeseries_figure(rows, t0).to_html(full_html=False, include_plotlyjs="cdn")
    empty_state = "<p><b>No sessions, requests or tool spans recorded: per-session timeline is empty.</b></p>"
    bottom = gantt_figure(segments, order, t0).to_html(full_html=False, include_plotlyjs=False) if segments else empty_state
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>SWE-smith run report</title></head><body>{top}{bottom}</body></html>"


def write_models_csv(path: Path, model: type[ModelT], rows: list[ModelT]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(model.model_fields))
        writer.writeheader()
        writer.writerows(r.model_dump() for r in rows)


def write_report(run_dir: Path, window_start: float | None, kv_pool_gb: float | None) -> None:
    inputs = load_inputs(run_dir)
    gauges = GAUGES["vllm"]
    start = window_start if window_start is not None else inputs.samples[0].ts
    out = run_dir / "report"
    out.mkdir(exist_ok=True)

    records, failed = split_requests(inputs.requests)
    records = attach_peaks(records, inputs.samples, gauges)
    gaps = tool_gap_rows(inputs.requests, inputs.spans)
    rows = timeseries_rows(inputs.samples, inputs.dmon, start, session_windows(inputs.requests), gauges)
    derived = derived_metrics(rows, inputs.samples, gaps, inputs.spans, inputs.sessions, inputs.requests, gauges, kv_pool_gb)

    write_models_csv(out / "requests.csv", RequestRecord, records)
    write_models_csv(out / "failed_requests.csv", FailedRequestRow, failed)
    write_models_csv(out / "sessions.csv", SessionRow, session_rows(inputs.sessions, inputs.requests))
    write_models_csv(out / "tool_gaps.csv", GapRow, gaps)
    write_models_csv(out / "gap_stats.csv", GapStatRow, gap_stat_rows(gaps))
    write_models_csv(out / "timeseries.csv", TimeseriesRow, rows)
    write_models_csv(out / "derived.csv", DerivedMetrics, [derived])
    (out / "derived.json").write_text(derived.model_dump_json(indent=2))
    (out / "timeline.perfetto.json").write_text(json.dumps(timeline_trace(records, inputs, gaps, gauges)))
    t0 = time_origin(inputs.samples, records)
    segments = gantt_segments(inputs.requests, inputs.spans, inputs.sessions, gaps)
    (out / "dashboard.html").write_text(dashboard_html(rows, segments, session_order(inputs.requests), t0))
    if derived.zero_requests:
        print(
            f"WARNING: zero proxy requests in {run_dir} (sessions={derived.n_sessions}, tool_spans={derived.n_span_records}): "
            "no request reached the proxy",
            file=sys.stderr,
        )


def parse_args(argv: list[str]) -> ReportArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--window-start", type=float, default=None, help="epoch of the first dmon row; default first server_metrics ts")
    parser.add_argument("--kv-pool-gb", type=float, default=None, help="KV pool size in GB, enables goodput per GB of KV")
    return ReportArgs.model_validate(vars(parser.parse_args(argv)))


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    write_report(args.run_dir, args.window_start, args.kv_pool_gb)


if __name__ == "__main__":
    main(sys.argv[1:])
