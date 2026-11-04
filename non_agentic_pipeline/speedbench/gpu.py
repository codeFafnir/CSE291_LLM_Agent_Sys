"""Device-wide GPU memory, sampled from a background thread."""

import threading
from dataclasses import dataclass

import pynvml


@dataclass(frozen=True)
class GpuSampler:
    stop: threading.Event
    thread: threading.Thread
    peak: list[int]


def gpu_used_bytes() -> int:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    return int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)


def _sample_loop(stop: threading.Event, peak: list[int], interval_s: float) -> None:
    while not stop.is_set():
        peak[0] = max(peak[0], gpu_used_bytes())
        stop.wait(interval_s)


def start_gpu_sampler(interval_s: float) -> GpuSampler:
    stop = threading.Event()
    peak = [gpu_used_bytes()]
    thread = threading.Thread(target=_sample_loop, args=(stop, peak, interval_s), daemon=True)
    thread.start()
    return GpuSampler(stop=stop, thread=thread, peak=peak)


def stop_gpu_sampler(sampler: GpuSampler) -> int:
    sampler.stop.set()
    sampler.thread.join()
    return sampler.peak[0]


def mib(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)
