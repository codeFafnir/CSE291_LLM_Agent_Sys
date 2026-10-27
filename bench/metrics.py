"""Shared bench rows. Peak memory is sampled while the engine runs.
NVIDIA: device-wide used memory. Apple silicon: phys_footprint of this process tree.
"""

import ctypes
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass

SAMPLE_INTERVAL_S = 0.05
# rusage_info_v0: 16-byte uuid, then seven uint64 fields before ri_phys_footprint.
_RUSAGE_INFO_V0 = 0
_PHYS_FOOTPRINT_OFFSET = 72


@dataclass(frozen=True)
class BenchRow:
    prompt_id: str
    prompt_tokens: int
    output_tokens: int
    e2e_s: float
    ttft_s: float
    tpot_s: float | None
    prefill_tok_s: float | None
    decode_tok_s: float | None
    peak_gpu_mib: float
    accept_length: float | None
    accept_rate: float | None


def tpot_s(ttft_s: float, e2e_s: float, output_tokens: int) -> float | None:
    # First token is TTFT. TPOT is the mean gap over the remaining tokens.
    if output_tokens <= 1 or e2e_s <= ttft_s:
        return None
    return (e2e_s - ttft_s) / (output_tokens - 1)


def prefill_tok_s(prompt_tokens: int, ttft_s: float) -> float | None:
    if ttft_s <= 0.0:
        return None
    return prompt_tokens / ttft_s


def decode_tok_s(tpot: float | None) -> float | None:
    if tpot is None or tpot <= 0.0:
        return None
    return 1.0 / tpot


def gpu_used_bytes() -> int:
    # Imported here so Apple silicon runs do not need NVML.
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    return int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)


def unified_used_bytes() -> int:
    # Unified memory has no device counter. phys_footprint is Activity Monitor's Memory
    # column and includes Metal buffers. vLLM runs EngineCore as a child process.
    import psutil

    me = psutil.Process()
    pids = [me.pid, *(child.pid for child in me.children(recursive=True))]
    return sum(_phys_footprint(pid) for pid in pids)


def _phys_footprint(pid: int) -> int:
    buffer = ctypes.create_string_buffer(256)
    if ctypes.CDLL("/usr/lib/libproc.dylib").proc_pid_rusage(pid, _RUSAGE_INFO_V0, buffer) != 0:
        return 0  # Child exited between listing and reading.
    return int.from_bytes(buffer.raw[_PHYS_FOOTPRINT_OFFSET : _PHYS_FOOTPRINT_OFFSET + 8], "little")


def _sample_loop(
    stop: threading.Event, peak: list[int], interval_s: float, read_bytes: Callable[[], int]
) -> None:
    while not stop.is_set():
        peak[0] = max(peak[0], read_bytes())
        stop.wait(interval_s)


def start_gpu_sampler(
    interval_s: float, read_bytes: Callable[[], int] = gpu_used_bytes
) -> tuple[threading.Event, threading.Thread, list[int]]:
    stop = threading.Event()
    peak = [read_bytes()]
    thread = threading.Thread(target=_sample_loop, args=(stop, peak, interval_s, read_bytes), daemon=True)
    thread.start()
    return stop, thread, peak


def stop_gpu_sampler(stop: threading.Event, thread: threading.Thread, peak: list[int]) -> int:
    stop.set()
    thread.join()
    return peak[0]


def mib(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def row_dict(row: BenchRow) -> dict[str, object]:
    return asdict(row)


def summarize(rows: tuple[BenchRow, ...]) -> dict[str, float]:
    ttft = [row.ttft_s for row in rows]
    tpot = [row.tpot_s for row in rows if row.tpot_s is not None]
    prefill = [row.prefill_tok_s for row in rows if row.prefill_tok_s is not None]
    decode = [row.decode_tok_s for row in rows if row.decode_tok_s is not None]
    peak = [row.peak_gpu_mib for row in rows]
    summary = {
        "requests": float(len(rows)),
        "ttft_s_mean": _mean(ttft),
        "ttft_s_p50": _p50(ttft),
        "peak_gpu_mib_max": max(peak) if peak else 0.0,
    }
    if tpot:
        summary["tpot_s_mean"] = _mean(tpot)
        summary["tpot_s_p50"] = _p50(tpot)
    if prefill:
        summary["prefill_tok_s_mean"] = _mean(prefill)
    if decode:
        summary["decode_tok_s_mean"] = _mean(decode)
    return summary


def format_row(row: BenchRow) -> str:
    tpot_ms = row.tpot_s * 1000.0 if row.tpot_s is not None else float("nan")
    return (
        f"{row.prompt_id} prompt_tokens={row.prompt_tokens} "
        f"ttft_s={row.ttft_s:.3f} tpot_ms={tpot_ms:.2f} "
        f"prefill_tok_s={row.prefill_tok_s or 0.0:.1f} "
        f"decode_tok_s={row.decode_tok_s or 0.0:.1f} "
        f"peak_mib={row.peak_gpu_mib:.0f} accept_length={row.accept_length}"
    )


def format_summary(summary: dict[str, float]) -> str:
    parts = [f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}" for key, value in summary.items()]
    return "summary " + " ".join(parts)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _p50(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]

