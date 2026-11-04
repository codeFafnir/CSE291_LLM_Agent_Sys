"""Per-request records from vLLM OTLP spans or SGLang per-request metric files, plus a Perfetto timeline.

vLLM: spans.jsonl lines are OTLP JSON exports (resourceSpans). One `llm_request` span per request.
SGLang: one JSON record per request (meta_info fields) in sglang-request-metrics-*.log.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from summarize_sla import Engine, Sample

NS = 1e9
US = 1e6


class Gauges(BaseModel):
    model_config = ConfigDict(frozen=True)
    running: str
    waiting: str
    kv_usage: str
    prompt_tokens_total: str
    generation_tokens_total: str
    preemptions_total: str
    prefix_hits_total: str | None


GAUGES: dict[Engine, Gauges] = {
    "vllm": Gauges(
        running="vllm:num_requests_running",
        waiting="vllm:num_requests_waiting",
        kv_usage="vllm:kv_cache_usage_perc",
        prompt_tokens_total="vllm:prompt_tokens_total",
        generation_tokens_total="vllm:generation_tokens_total",
        preemptions_total="vllm:num_preemptions_total",
        prefix_hits_total="vllm:prefix_cache_hits_total",
    ),
    "sglang": Gauges(
        running="sglang:num_running_reqs",
        waiting="sglang:num_queue_reqs",
        kv_usage="sglang:token_usage",
        prompt_tokens_total="sglang:prompt_tokens_total",
        generation_tokens_total="sglang:generation_tokens_total",
        preemptions_total="sglang:num_retracted_requests_total",
        prefix_hits_total=None,
    ),
}


class RequestRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    request_id: str
    trial: str | None
    arrival_ts: float
    queue_s: float | None
    prefill_s: float | None
    ttft_s: float | None
    decode_s: float | None
    tpot_s: float | None
    e2e_s: float
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int | None
    peak_gpu_mib_during: float | None
    peak_kv_usage_during: float | None


class ParsedRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    record: RequestRecord
    raw: str


class TrialWindow(BaseModel):
    model_config = ConfigDict(frozen=True)
    task: str
    start_ts: float
    end_ts: float


class _AnyValue(BaseModel):
    stringValue: str | None = None
    intValue: str | int | None = None
    doubleValue: float | None = None


class _KeyValue(BaseModel):
    key: str
    value: _AnyValue


class _Span(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    startTimeUnixNano: str | int
    attributes: list[_KeyValue]


class _ScopeSpans(BaseModel):
    spans: list[_Span]


class _ResourceSpans(BaseModel):
    scopeSpans: list[_ScopeSpans]


class _Export(BaseModel):
    resourceSpans: list[_ResourceSpans]


class _VllmAttrs(BaseModel):
    request_id: str = Field(alias="gen_ai.request.id")
    ttft: float = Field(alias="gen_ai.latency.time_to_first_token")
    e2e: float = Field(alias="gen_ai.latency.e2e")
    queue: float = Field(alias="gen_ai.latency.time_in_queue")
    prefill: float = Field(alias="gen_ai.latency.time_in_model_prefill")
    decode: float = Field(alias="gen_ai.latency.time_in_model_decode")
    prompt_tokens: int = Field(alias="gen_ai.usage.prompt_tokens")
    completion_tokens: int = Field(alias="gen_ai.usage.completion_tokens")


class _SglangRecord(BaseModel):
    id: str
    request_received_ts: float
    queue_time: float | None = None
    forward_entry_time: float | None = None
    prefill_finished_time: float | None = None
    first_token_latency: float | None = None
    e2e_latency: float
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int | None = None


def tpot(decode_s: float | None, completion_tokens: int) -> float | None:
    if decode_s is None or completion_tokens < 2:
        return None
    return decode_s / (completion_tokens - 1)


def _attr_value(value: _AnyValue) -> str | int | float:
    if value.stringValue is not None:
        return value.stringValue
    if value.intValue is not None:
        return int(value.intValue)
    if value.doubleValue is not None:
        return value.doubleValue
    raise ValueError(f"unsupported OTLP attribute value: {value}")


def _in_window(ts: float, window: tuple[float, float]) -> bool:
    return window[0] <= ts <= window[1]


def parse_vllm_spans(lines: list[str], window: tuple[float, float]) -> list[ParsedRequest]:
    parsed: list[ParsedRequest] = []
    for line in lines:
        if not line.strip():
            continue
        export = _Export.model_validate_json(line)
        for resource in export.resourceSpans:
            for scope in resource.scopeSpans:
                for span in scope.spans:
                    if span.name != "llm_request":
                        continue
                    arrival = int(span.startTimeUnixNano) / NS
                    if not _in_window(arrival, window):
                        continue
                    attrs = _VllmAttrs.model_validate({a.key: _attr_value(a.value) for a in span.attributes})
                    record = RequestRecord(
                        request_id=attrs.request_id,
                        trial=None,
                        arrival_ts=arrival,
                        queue_s=attrs.queue,
                        prefill_s=attrs.prefill,
                        ttft_s=attrs.ttft,
                        decode_s=attrs.decode,
                        tpot_s=tpot(attrs.decode, attrs.completion_tokens),
                        e2e_s=attrs.e2e,
                        prompt_tokens=attrs.prompt_tokens,
                        completion_tokens=attrs.completion_tokens,
                        cached_tokens=None,
                        peak_gpu_mib_during=None,
                        peak_kv_usage_during=None,
                    )
                    parsed.append(ParsedRequest(record=record, raw=span.model_dump_json()))
    return parsed


def parse_sglang_records(lines: list[str], window: tuple[float, float]) -> list[ParsedRequest]:
    parsed: list[ParsedRequest] = []
    for line in lines:
        if not line.strip():
            continue
        row = _SglangRecord.model_validate_json(line)
        if not _in_window(row.request_received_ts, window):
            continue
        prefill = (
            row.prefill_finished_time - row.forward_entry_time
            if row.prefill_finished_time is not None and row.forward_entry_time is not None
            else None
        )
        decode = row.e2e_latency - row.first_token_latency if row.first_token_latency is not None else None
        record = RequestRecord(
            request_id=row.id,
            trial=None,
            arrival_ts=row.request_received_ts,
            queue_s=row.queue_time,
            prefill_s=prefill,
            ttft_s=row.first_token_latency,
            decode_s=decode,
            tpot_s=tpot(decode, row.completion_tokens),
            e2e_s=row.e2e_latency,
            prompt_tokens=row.prompt_tokens,
            completion_tokens=row.completion_tokens,
            cached_tokens=row.cached_tokens,
            peak_gpu_mib_during=None,
            peak_kv_usage_during=None,
        )
        parsed.append(ParsedRequest(record=record, raw=line))
    return parsed


def read_lines(source: Path) -> list[str]:
    files = sorted(source.glob("*.log")) if source.is_dir() else [source]
    return [line for path in files for line in path.read_text().splitlines()]


def attach_trials(records: list[RequestRecord], trials: list[TrialWindow]) -> list[RequestRecord]:
    # A request that falls in zero or several trial windows gets no trial instead of a guess.
    def trial_of(record: RequestRecord) -> str | None:
        hits = [t.task for t in trials if t.start_ts <= record.arrival_ts <= t.end_ts]
        return hits[0] if len(hits) == 1 else None

    return [r.model_copy(update={"trial": trial_of(r)}) for r in records]


def attach_peaks(records: list[RequestRecord], samples: list[Sample], gauges: Gauges) -> list[RequestRecord]:
    def peaks(record: RequestRecord) -> tuple[float, float]:
        end = record.arrival_ts + record.e2e_s
        inside = [s for s in samples if record.arrival_ts <= s.ts <= end]
        mid = record.arrival_ts + record.e2e_s / 2
        chosen = inside or [min(samples, key=lambda s: abs(s.ts - mid))]
        return max(s.gpu_used_mib for s in chosen), max(s.series.get(gauges.kv_usage, 0.0) for s in chosen)

    updated: list[RequestRecord] = []
    for record in records:
        gpu, kv = peaks(record)
        updated.append(record.model_copy(update={"peak_gpu_mib_during": gpu, "peak_kv_usage_during": kv}))
    return updated


def assign_lanes(records: list[RequestRecord]) -> list[int]:
    # Greedy interval packing: the lowest lane that is free at arrival.
    order = sorted(range(len(records)), key=lambda i: records[i].arrival_ts)
    lane_end: list[float] = []
    lanes = [0] * len(records)
    for i in order:
        record = records[i]
        for lane, end in enumerate(lane_end):
            if end <= record.arrival_ts:
                lanes[i] = lane
                lane_end[lane] = record.arrival_ts + record.e2e_s
                break
        else:
            lanes[i] = len(lane_end)
            lane_end.append(record.arrival_ts + record.e2e_s)
    return lanes


def _slice(name: str, tid: int, start_s: float, dur_s: float, t0: float, args: dict[str, str | int | float | None]) -> dict[str, object]:
    return {"ph": "X", "pid": 1, "tid": tid, "name": name, "ts": (start_s - t0) * US, "dur": dur_s * US, "args": args}


def perfetto_trace(
    records: list[RequestRecord], samples: list[Sample], gauges: Gauges
) -> dict[str, list[dict[str, object]]]:
    t0 = min([r.arrival_ts for r in records] + [s.ts for s in samples])
    lanes = assign_lanes(records)
    events: list[dict[str, object]] = [
        {"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "requests"}},
        {"ph": "M", "pid": 2, "name": "process_name", "args": {"name": "server"}},
    ]
    for lane in range(max(lanes, default=-1) + 1):
        events.append({"ph": "M", "pid": 1, "tid": lane, "name": "thread_name", "args": {"name": f"lane {lane}"}})
    for record, lane in zip(records, lanes):
        info: dict[str, str | int | float | None] = {
            "request_id": record.request_id,
            "trial": record.trial,
            "prompt_tokens": record.prompt_tokens,
            "completion_tokens": record.completion_tokens,
            "ttft_s": record.ttft_s,
            "tpot_s": record.tpot_s,
        }
        events.append(_slice(f"req {record.request_id[:12]}", lane, record.arrival_ts, record.e2e_s, t0, info))
        if record.queue_s is not None:
            events.append(_slice("queue", lane, record.arrival_ts, record.queue_s, t0, {}))
        if record.prefill_s is not None and record.queue_s is not None:
            events.append(_slice("prefill", lane, record.arrival_ts + record.queue_s, record.prefill_s, t0, {}))
        if record.decode_s is not None:
            events.append(_slice("decode", lane, record.arrival_ts + record.e2e_s - record.decode_s, record.decode_s, t0, {}))
    for sample in samples:
        counters = {
            "running": sample.series.get(gauges.running, 0.0),
            "waiting": sample.series.get(gauges.waiting, 0.0),
            "kv_usage": sample.series.get(gauges.kv_usage, 0.0),
            "gpu_used_mib": sample.gpu_used_mib,
        }
        for name, value in counters.items():
            events.append({"ph": "C", "pid": 2, "name": name, "ts": (sample.ts - t0) * US, "args": {name: value}})
    return {"traceEvents": events}
