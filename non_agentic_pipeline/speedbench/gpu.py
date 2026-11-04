"""Device-wide GPU memory, sampled from a background thread."""

import threading
from dataclasses import dataclass

import pynvml


@dataclass(frozen=True)
class GpuSampler:
    stop: threading.Event
    thread: threading.Thread
    peak: list[int | None]


def gpu_used_bytes() -> int | None:
    # JupyterHub denies NVML memory queries. A denied read is an empty sample, not a failed run.
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
    except pynvml.NVMLError:
        return None


def _sample_loop(stop: threading.Event, peak: list[int | None], interval_s: float) -> None:
    while not stop.is_set():
        used = gpu_used_bytes()
        if used is not None:
            peak[0] = used if peak[0] is None else max(peak[0], used)
        stop.wait(interval_s)


def start_gpu_sampler(interval_s: float) -> GpuSampler:
    stop = threading.Event()
    peak: list[int | None] = [gpu_used_bytes()]
    thread = threading.Thread(target=_sample_loop, args=(stop, peak, interval_s), daemon=True)
    thread.start()
    return GpuSampler(stop=stop, thread=thread, peak=peak)


def stop_gpu_sampler(sampler: GpuSampler) -> int | None:
    sampler.stop.set()
    sampler.thread.join()
    return sampler.peak[0]


def mib(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)
