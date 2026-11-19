import csv
import json
from pathlib import Path

import pytest
from pydantic import BaseModel

import report
from run import Session
from session_proxy import RequestLog
from summarize_sla import Sample
from tool_span import ToolSpan

T0 = 1_700_000_000.0
N_SAMPLES = 35
FIXTURE_GLOB = "baseline-vllm-nemotron-1-c1-*"
DMON_HEADER = (
    "# gpu    pwr  gtemp  mtemp     sm    mem    enc    dec    jpg    ofa   mclk   pclk  pviol  tviol     fb   bar1   ccpm  sbecc  dbecc    pci  rxpci  txpci \n"
    "# Idx      W      C      C      %      %      %      %      %      %    MHz    MHz      %   bool     MB     MB     MB   errs   errs   errs   MB/s   MB/s \n"
)
GAUGES = report.GAUGES["vllm"]


def req(base: float, session: str, turn: int, send: float, end: float, status: int, tokens: tuple[int, int, int] | None) -> RequestLog:
    return RequestLog(
        session_id=session,
        turn=turn,
        request_id=f"{session}-t{turn:03d}",
        path="v1/chat/completions",
        t_send=base + send,
        t_first_byte=base + end if status < 400 else None,
        t_end=base + end,
        status=status,
        error=None if status < 400 else "maximum context length exceeded",
        max_tokens=2048,
        request_bytes=1000,
        prompt_tokens=tokens[0] if tokens else None,
        completion_tokens=tokens[1] if tokens else None,
        cached_tokens=tokens[2] if tokens else None,
    )


def span(base: float, session: str, step: int, tool: str, start: float, end: float, timed_out: bool) -> ToolSpan:
    return ToolSpan(
        session_id=session,
        step=step,
        tool=tool,
        action_head=tool,
        executed=True,
        t_step_start=base + start - 0.2,
        t_tool_start=base + start,
        t_tool_end=base + end,
        execution_time_s=end - start,
        timed_out=timed_out,
        step_exit_status=None,
        observation_chars=10,
    )


def session(base: float, instance_id: str, start: float, end: float) -> Session:
    return Session(
        instance_id=instance_id,
        image_key="srt",
        arrival_offset_s=0.0,
        t_arrival=base + start,
        t_start=base + start,
        t_end=base + end,
        returncode=0,
        status="completed",
        exit_status="submitted",
        api_calls=3,
        submitted=True,
        failed_requests=0,
        error_excerpt=None,
    )


def scenario(base: float) -> tuple[list[RequestLog], list[ToolSpan], list[Session]]:
    requests = [
        req(base, "s1", 1, 0, 4, 200, (1000, 50, 0)),
        req(base, "s1", 2, 20, 24, 200, (1200, 60, 1000)),
        req(base, "s1", 3, 30, 33, 400, None),
        req(base, "s2", 1, 2, 8, 200, (900, 40, 0)),
        req(base, "s2", 2, 20, 26, 200, (1100, 30, 450)),
    ]
    spans = [
        span(base, "s1", 1, "bash", 4.2, 20.2, False),
        span(base, "s1", 2, "str_replace_editor:view", 24.5, 29.5, False),
        span(base, "s2", 1, "bash", 8.0, 20.0, True),
    ]
    return requests, spans, [session(base, "s1", -1, 34), session(base, "s2", 1, 27)]


REQUESTS, SPANS, SESSIONS = scenario(T0)


def all_in_gap_offsets() -> set[int]:
    return set(range(8, 20)) | set(range(26, 30))


def write_jsonl(path: Path, rows: list[BaseModel]) -> None:
    path.write_text("".join(r.model_dump_json() + "\n" for r in rows))


def dmon_line(sm: int, fb: int, pwr: int) -> str:
    return f"{0:5d}{pwr:7d}{55:7d}{'-':>7}{sm:7d}{50:7d}{0:7d}{0:7d}{0:7d}{0:7d}{6251:7d}{1500:7d}{0:7d}{0:7d}{fb:7d}{fb:7d}{0:7d}{0:7d}{0:7d}{0:7d}{0:7d}{0:7d} \n"


def write_dmon(path: Path) -> None:
    quiet = all_in_gap_offsets()
    lines = [DMON_HEADER]
    for k in range(N_SAMPLES):
        if k == 20:
            lines.append(DMON_HEADER)
        lines.append(dmon_line(0 if k in quiet else 80, 19064, 30 + k))
    path.write_text("".join(lines))


def metrics_line(k: int, with_continuum: bool) -> str:
    series = {
        "vllm:num_requests_running": 1.0,
        "vllm:num_requests_waiting": 0.0,
        "vllm:kv_cache_usage_perc": 0.01 * k,
        "vllm:prefix_cache_queries_total": 100.0 * k,
        "vllm:prefix_cache_hits_total": 50.0 * k,
        "vllm:num_preemptions_total": float(k // 10),
        "vllm:prompt_tokens_total": 1000.0 * k,
        "vllm:generation_tokens_total": 10.0 * k,
        "vllm:spec_decode_num_draft_tokens_total": 6.0 * k,
        "vllm:spec_decode_num_accepted_tokens_total": 3.0 * k,
    }
    if with_continuum:
        series |= {
            "vllm:continuum_pins_total": float(k),
            "vllm:continuum_unpins_total{reason=ttl}": float(k // 2),
            "vllm:continuum_unpins_total{reason=next_turn}": float(k // 3),
            "vllm:continuum_unpins_total{reason=pressure}": 0.0,
            "vllm:continuum_pinned_blocks": 100.0 if k in all_in_gap_offsets() else 20.0,
        }
    return json.dumps({"ts": T0 + k, "gpu_used_mib": 19064.0, "series": series}) + "\n"


def make_run_dir(root: Path, with_continuum: bool) -> Path:
    run_dir = root / "run"
    run_dir.mkdir()
    write_jsonl(run_dir / "proxy_requests.jsonl", list(REQUESTS))
    write_jsonl(run_dir / "tool_spans.jsonl", list(SPANS))
    write_jsonl(run_dir / "sessions.jsonl", list(SESSIONS))
    (run_dir / "server_metrics.jsonl").write_text("".join(metrics_line(k, with_continuum) for k in range(N_SAMPLES)))
    write_dmon(run_dir / "gpu_dmon.log")
    (run_dir / "serve_config.env").write_text("MAX_MODEL_LEN=30000\nMAX_NUM_SEQS=8\n")
    return run_dir


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_tool_gaps_and_timeout_violation() -> None:
    gaps = {(g.session_id, g.next_turn): g for g in report.tool_gap_rows(REQUESTS, SPANS)}
    assert len(gaps) == 3
    long_gap = gaps[("s1", 2)]
    assert long_gap.gap_s == pytest.approx(16.0)
    assert long_gap.tool == "bash" and long_gap.tool_duration_s == pytest.approx(16.0)
    assert long_gap.tool_timeout_violation is True
    assert long_gap.prefix_hit_ratio == pytest.approx(1000 / 1200)
    edit_gap = gaps[("s1", 3)]
    assert edit_gap.gap_s == pytest.approx(6.0)
    assert edit_gap.tool == "str_replace_editor:view" and edit_gap.tool_timeout_violation is False
    assert edit_gap.prefix_hit_ratio is None
    ok_gap = gaps[("s2", 2)]
    assert ok_gap.gap_s == pytest.approx(12.0)
    assert ok_gap.tool_duration_s == pytest.approx(12.0)
    assert ok_gap.tool_timed_out is True and ok_gap.tool_timeout_violation is False


def test_gap_without_span_has_no_tool() -> None:
    gaps = report.tool_gap_rows(REQUESTS, [])
    assert all(g.tool is None and not g.tool_timeout_violation for g in gaps)


def test_failed_requests_split() -> None:
    records, failed = report.split_requests(REQUESTS)
    assert sorted(r.request_id for r in records) == ["s1-t001", "s1-t002", "s2-t001", "s2-t002"]
    assert [f.request_id for f in failed] == ["s1-t003"]
    assert failed[0].error == "maximum context length exceeded"
    first = next(r for r in records if r.request_id == "s1-t002")
    assert first.trial == "s1" and first.arrival_ts == T0 + 20 and first.e2e_s == pytest.approx(4.0)
    assert first.cached_tokens == 1000


def test_gap_stats_buckets() -> None:
    stats = report.gap_stat_rows(report.tool_gap_rows(REQUESTS, SPANS))
    overall = next(s for s in stats if s.kind == "quantiles" and s.tool == "all")
    assert overall.n == 3 and overall.gap_max_s == pytest.approx(16.0)
    hist = {s.bucket: s.n for s in stats if s.kind == "histogram" and s.tool == "all"}
    assert hist["5-10s"] == 1 and hist["10-15s"] == 1 and hist["15-30s"] == 1 and hist["<1s"] == 0
    prefix = {s.bucket: s for s in stats if s.kind == "prefix_hit"}
    assert prefix["15-30s"].prefix_hit_mean == pytest.approx(1000 / 1200)
    assert prefix["5-10s"].n == 0 and prefix["5-10s"].prefix_hit_mean is None
    per_tool = {s.tool for s in stats if s.kind == "quantiles"}
    assert per_tool == {"all", "bash", "str_replace_editor:view"}


def test_all_in_gap_logic() -> None:
    samples = [Sample.model_validate_json(metrics_line(k, False)) for k in range(N_SAMPLES)]
    dmon = report.DmonSeries(sm=[0.0] * N_SAMPLES, fb=[1.0] * N_SAMPLES, pwr=[2.0] * N_SAMPLES)
    rows = report.timeseries_rows(samples, dmon, T0, report.session_windows(REQUESTS), GAUGES)
    flagged = {round(r.ts - T0) for r in rows if r.all_in_gap}
    assert flagged == all_in_gap_offsets()
    at = {round(r.ts - T0): r for r in rows}
    assert at[3].n_decoding == 2 and at[3].n_in_tool_gap == 0 and at[3].n_sessions_active == 2
    assert at[5].n_in_tool_gap == 1 and at[5].n_decoding == 1
    assert at[27].n_sessions_active == 1 and at[27].n_in_tool_gap == 1
    assert at[31].n_decoding == 1 and not at[31].all_in_gap
    assert at[33].n_sessions_active == 0 and not at[33].all_in_gap
    assert at[10].gen_tokens_per_s == pytest.approx(10.0)
    assert at[10].prefix_hit_rate_delta == pytest.approx(0.5)
    assert at[0].gen_tokens_per_s == 0.0


def test_missing_continuum_keys_are_zero(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path, with_continuum=False)
    report.main(["--run-dir", str(run_dir)])
    rows = read_csv(run_dir / "report" / "timeseries.csv")
    assert len(rows) == N_SAMPLES
    for key in ("continuum_pins_delta", "continuum_unpins_ttl_delta", "continuum_unpins_next_turn_delta", "continuum_unpins_pressure_delta", "continuum_pinned_blocks"):
        assert {float(r[key]) for r in rows} == {0.0}


def test_full_report_outputs(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path, with_continuum=True)
    report.main(["--run-dir", str(run_dir), "--window-start", str(T0), "--kv-pool-gb", "4.0"])
    out = run_dir / "report"
    for name in ("requests.csv", "failed_requests.csv", "tool_gaps.csv", "gap_stats.csv", "timeseries.csv", "derived.csv", "derived.json", "timeline.perfetto.json", "dashboard.html"):
        assert (out / name).exists(), name

    assert sorted(r["request_id"] for r in read_csv(out / "requests.csv")) == ["s1-t001", "s1-t002", "s2-t001", "s2-t002"]
    failed = read_csv(out / "failed_requests.csv")
    assert [f["request_id"] for f in failed] == ["s1-t003"] and "context length" in failed[0]["error"]
    violations = {(g["session_id"], g["next_turn"]): g["tool_timeout_violation"] for g in read_csv(out / "tool_gaps.csv")}
    assert violations[("s1", "2")] == "True" and violations[("s2", "2")] == "False"

    series = {round(float(r["ts"]) - T0): r for r in read_csv(out / "timeseries.csv")}
    assert float(series[10]["continuum_pinned_blocks"]) == 100.0 and float(series[2]["continuum_pinned_blocks"]) == 20.0
    assert float(series[10]["gpu_sm_pct"]) == 0.0 and float(series[2]["gpu_sm_pct"]) == 80.0
    assert float(series[10]["gpu_power_w"]) == 40.0 and float(series[10]["gpu_fb_mib"]) == 19064.0
    assert float(series[22]["gpu_sm_pct"]) == 80.0 and float(series[22]["continuum_pins_delta"]) == 1.0

    derived = json.loads((out / "derived.json").read_text())
    assert derived["mean_sm_pct_all_in_gap"] == pytest.approx(0.0)
    assert derived["mean_sm_pct_any_decoding"] == pytest.approx(80.0)
    assert derived["tool_timeout_violations"] == 1
    assert derived["n_tool_spans"] == 3 and derived["n_timed_out"] == 1 and derived["timeout_rate"] == pytest.approx(1 / 3)
    assert derived["timeout_cap_s_observed"] == pytest.approx(12.0)
    assert derived["n_failed_requests"] == 1 and derived["n_gaps"] == 3
    assert derived["mean_pinned_blocks_all_in_gap"] == pytest.approx(100.0)
    assert derived["output_tokens_per_s"] == pytest.approx(10.0)
    assert derived["goodput_tok_s_per_gb_kv"] == pytest.approx(10.0 / (derived["mean_kv_usage"] * 4.0))
    assert read_csv(out / "derived.csv")[0]["tool_timeout_violations"] == "1"


def test_timeline_has_session_tracks(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path, with_continuum=True)
    report.main(["--run-dir", str(run_dir)])
    events = json.loads((run_dir / "report" / "timeline.perfetto.json").read_text())["traceEvents"]
    process_names = {e["args"]["name"] for e in events if e["ph"] == "M" and e["name"] == "process_name"}
    assert {"requests", "server", "sessions"} <= process_names
    slices = [e for e in events if e["ph"] == "X" and e["pid"] == report.SESSIONS_PID]
    tools = [e for e in slices if e["name"] in ("bash", "str_replace_editor:view")]
    assert len(tools) == 3
    long_tool = next(e for e in tools if e["args"]["tool_timeout_violation"])
    assert long_tool["dur"] == pytest.approx(16.0 * report.US)
    assert long_tool["ts"] == pytest.approx(4.2 * report.US)
    assert sum(1 for e in slices if e["name"] == "gap") == 3
    assert sum(1 for e in slices if str(e["name"]).startswith("llm ")) == 5
    assert any(e["pid"] == 1 for e in events)


def test_dashboard_has_plotly_div(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path, with_continuum=True)
    report.main(["--run-dir", str(run_dir)])
    html = (run_dir / "report" / "dashboard.html").read_text()
    assert html.count('class="plotly-graph-div"') == 2
    assert "plotly" in html and "Per-session timeline" in html


def test_smoke_on_real_fixture(tmp_path: Path) -> None:
    fixtures = sorted((Path(__file__).resolve().parents[3] / "results" / "agentic_results" / "c1").glob(FIXTURE_GLOB))
    if not fixtures:
        pytest.skip("real fixture run dir not present")
    src = fixtures[0]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metrics_lines = src.joinpath("server_metrics.jsonl").read_text().splitlines()[:300]
    (run_dir / "server_metrics.jsonl").write_text("\n".join(metrics_lines) + "\n")
    dmon_lines = src.joinpath("gpu_dmon.log").read_text().splitlines()[:300]
    (run_dir / "gpu_dmon.log").write_text("\n".join(dmon_lines) + "\n")
    (run_dir / "serve_config.env").write_text(src.joinpath("serve_config.env").read_text())
    requests, spans, sessions = scenario(json.loads(metrics_lines[0])["ts"])
    write_jsonl(run_dir / "proxy_requests.jsonl", list(requests))
    write_jsonl(run_dir / "tool_spans.jsonl", list(spans))
    write_jsonl(run_dir / "sessions.jsonl", list(sessions))
    report.main(["--run-dir", str(run_dir)])
    out = run_dir / "report"
    assert len(read_csv(out / "timeseries.csv")) == 300
    assert json.loads((out / "derived.json").read_text())["n_samples"] == 300
    assert json.loads((out / "timeline.perfetto.json").read_text())["traceEvents"]
    assert "plotly-graph-div" in (out / "dashboard.html").read_text()


def test_sessions_csv_has_new_fields_and_token_totals(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path, with_continuum=True)
    failing = SESSIONS[0].model_copy(update={"failed_requests": 1, "error_excerpt": "maximum context length exceeded"})
    write_jsonl(run_dir / "sessions.jsonl", [failing, SESSIONS[1]])
    report.main(["--run-dir", str(run_dir)])
    rows = {r["instance_id"]: r for r in read_csv(run_dir / "report" / "sessions.csv")}
    assert set(rows) == {"s1", "s2"}
    assert rows["s1"]["failed_requests"] == "1" and rows["s1"]["error_excerpt"] == "maximum context length exceeded"
    assert rows["s2"]["failed_requests"] == "0" and rows["s2"]["error_excerpt"] == ""
    assert rows["s1"]["n_requests"] == "3" and rows["s2"]["n_requests"] == "2"
    assert rows["s1"]["prompt_tokens_total"] == "2200" and rows["s1"]["completion_tokens_total"] == "110"
    assert rows["s1"]["cached_tokens_total"] == "1000"
    assert rows["s2"]["prompt_tokens_total"] == "2000" and rows["s2"]["cached_tokens_total"] == "450"
    assert list(rows["s1"])[: len(Session.model_fields)] == list(Session.model_fields)


def test_cached_tokens_total_is_none_when_no_request_reports_it(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path, with_continuum=True)
    no_cache = [r.model_copy(update={"cached_tokens": None}) for r in REQUESTS]
    write_jsonl(run_dir / "proxy_requests.jsonl", no_cache)
    report.main(["--run-dir", str(run_dir)])
    rows = {r["instance_id"]: r for r in read_csv(run_dir / "report" / "sessions.csv")}
    assert rows["s1"]["cached_tokens_total"] == "" and rows["s2"]["cached_tokens_total"] == ""
    assert rows["s1"]["prompt_tokens_total"] == "2200"


def plain_span(duration: float, timed_out: bool, executed: bool, timestamps: bool) -> ToolSpan:
    return ToolSpan(
        session_id="s1",
        step=1,
        tool="bash",
        action_head="x",
        executed=executed,
        t_step_start=T0,
        t_tool_start=T0 if timestamps and executed else None,
        t_tool_end=T0 + duration if timestamps and executed else None,
        execution_time_s=duration,
        timed_out=timed_out,
        step_exit_status=None,
        observation_chars=0,
    )


def test_timeout_violation_semantics() -> None:
    assert report.is_timeout_violation(plain_span(16.0, False, True, True))
    assert not report.is_timeout_violation(plain_span(15.4, True, True, True))
    assert not report.is_timeout_violation(plain_span(12.0, False, True, True))
    assert not report.is_timeout_violation(plain_span(20.0, False, False, False))
    assert report.is_timeout_violation(plain_span(16.0, False, True, False))
    assert not report.is_timeout_violation(plain_span(12.0, False, True, False))


def test_timeout_stats_use_executed_spans_only() -> None:
    spans = [
        plain_span(16.0, False, True, True),
        plain_span(15.4, True, True, True),
        plain_span(12.0, False, True, True),
        plain_span(10.2, True, True, True),
        plain_span(99.0, False, False, False),
    ]
    stats = report.timeout_stats(spans)
    assert stats.n_tool_spans == 4 and stats.n_timed_out == 2 and stats.timeout_rate == pytest.approx(0.5)
    assert stats.timeout_cap_s_observed == pytest.approx(15.4)
    assert stats.tool_timeout_violations == 1
    empty = report.timeout_stats([])
    assert empty.timeout_rate is None and empty.timeout_cap_s_observed is None and empty.tool_timeout_violations == 0


def test_dmon_rows_scale_to_collector_span() -> None:
    n_dmon, step_row = 950, 475
    samples = [Sample(ts=T0 + k, gpu_used_mib=0.0, series={}) for k in range(1001)]
    dmon = report.DmonSeries(
        sm=[0.0 if i < step_row else 100.0 for i in range(n_dmon)], fb=[1.0] * n_dmon, pwr=[2.0] * n_dmon
    )
    empty = report.SessionWindows(requests=[], gaps=[], active=[])
    rows = report.timeseries_rows(samples, dmon, T0, empty, GAUGES)
    step_time = step_row * 1000 / (n_dmon - 1)
    assert step_time == pytest.approx(500.5, abs=0.1)
    for row in rows:
        offset = row.ts - T0
        if offset < step_time - 2:
            assert row.gpu_sm_pct == 0.0, offset
        elif offset > step_time + 2:
            assert row.gpu_sm_pct == 100.0, offset
    assert rows[0].gpu_sm_pct == 0.0 and rows[-1].gpu_sm_pct == 100.0


def test_counter_reset_is_clipped_without_spike() -> None:
    gen = [100.0, 115.0, 5.0, 20.0, 35.0]
    samples = [
        Sample(ts=T0 + k, gpu_used_mib=0.0, series={"vllm:generation_tokens_total": g, "vllm:num_preemptions_total": g})
        for k, g in enumerate(gen)
    ]
    dmon = report.DmonSeries(sm=[0.0] * 5, fb=[0.0] * 5, pwr=[0.0] * 5)
    empty = report.SessionWindows(requests=[], gaps=[], active=[])
    rows = report.timeseries_rows(samples, dmon, T0, empty, GAUGES)
    assert [r.gen_tokens_per_s for r in rows] == [0.0, 15.0, 0.0, 15.0, 15.0]
    assert [r.preemptions_delta for r in rows] == [0.0, 15.0, 0.0, 15.0, 15.0]
    assert report.counter_total(samples, "vllm:generation_tokens_total") == pytest.approx(45.0)


def run_report(run_dir: Path) -> dict[str, object]:
    report.main(["--run-dir", str(run_dir)])
    return json.loads((run_dir / "report" / "derived.json").read_text())


def assert_empty_state_outputs(out: Path) -> None:
    for name in ("requests.csv", "failed_requests.csv", "tool_gaps.csv", "gap_stats.csv", "timeseries.csv", "derived.csv", "derived.json", "timeline.perfetto.json", "dashboard.html"):
        assert (out / name).exists(), name
    assert read_csv(out / "requests.csv") == [] and read_csv(out / "tool_gaps.csv") == []
    assert len(read_csv(out / "timeseries.csv")) == N_SAMPLES
    assert json.loads((out / "timeline.perfetto.json").read_text())["traceEvents"]
    assert "plotly-graph-div" in (out / "dashboard.html").read_text()


@pytest.mark.parametrize("empty_requests, empty_spans", [(True, False), (False, True), (True, True)])
def test_empty_optional_inputs(tmp_path: Path, capsys: pytest.CaptureFixture[str], empty_requests: bool, empty_spans: bool) -> None:
    run_dir = make_run_dir(tmp_path, with_continuum=True)
    if empty_requests:
        (run_dir / "proxy_requests.jsonl").write_text("")
    if empty_spans:
        (run_dir / "tool_spans.jsonl").write_text("")
    derived = run_report(run_dir)
    assert derived["zero_requests"] is empty_requests
    assert derived["n_requests"] == (0 if empty_requests else 5)
    assert derived["n_span_records"] == (0 if empty_spans else 3)
    assert derived["n_sessions"] == 2
    assert ("WARNING" in capsys.readouterr().err) is empty_requests
    if empty_requests:
        assert_empty_state_outputs(run_dir / "report")


def test_missing_optional_files_are_empty(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = make_run_dir(tmp_path, with_continuum=True)
    for name in ("proxy_requests.jsonl", "tool_spans.jsonl", "sessions.jsonl"):
        (run_dir / name).unlink()
    derived = run_report(run_dir)
    assert derived["zero_requests"] is True
    assert derived["n_requests"] == 0 and derived["n_sessions"] == 0 and derived["n_span_records"] == 0
    err = capsys.readouterr().err
    assert err.count("WARNING") == 1 and "requests" in err
    assert_empty_state_outputs(run_dir / "report")
    assert "No sessions, requests or tool spans" in (run_dir / "report" / "dashboard.html").read_text()


def test_missing_required_inputs_stay_strict(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path, with_continuum=True)
    (run_dir / "gpu_dmon.log").unlink()
    with pytest.raises(FileNotFoundError):
        report.main(["--run-dir", str(run_dir)])


def test_single_dmon_row(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path, with_continuum=True)
    (run_dir / "gpu_dmon.log").write_text(DMON_HEADER + dmon_line(80, 19064, 30))
    run_report(run_dir)
    series = {round(float(r["ts"]) - T0): r for r in read_csv(run_dir / "report" / "timeseries.csv")}
    assert float(series[0]["gpu_sm_pct"]) == 80.0
    assert series[5]["gpu_sm_pct"] == ""


def test_zero_length_window(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path, with_continuum=True)
    (run_dir / "server_metrics.jsonl").write_text(metrics_line(0, True))
    derived = run_report(run_dir)
    rows = read_csv(run_dir / "report" / "timeseries.csv")
    assert len(rows) == 1 and float(rows[0]["gpu_sm_pct"]) == 80.0
    assert derived["n_samples"] == 1 and derived["output_tokens_per_s"] == 0.0


def test_dmon_anchor_under_one_second() -> None:
    values = [10.0, 20.0, 30.0]
    assert report.dmon_at(values, T0, T0, T0 + 0.5) == 10.0
    assert report.dmon_at(values, T0 + 0.4, T0, T0 + 0.5) == 10.0
    assert report.dmon_at(values, T0, T0, T0) == 10.0
    assert report.dmon_at(values, T0 + 1.0, T0, T0) == 20.0
    assert report.dmon_at(values, T0, T0 + 1.0, T0) is None
    assert report.dmon_at([], T0, T0, T0) is None
