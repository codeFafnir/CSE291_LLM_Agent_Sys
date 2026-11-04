"""Row schema shared by capacity_sweep.py (writer) and summarize_sla.py (reader)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

Outcome = Literal["ok", "queued", "preempted", "rejected_too_long", "oom_runtime", "oom_startup"]

# OOM and KV-pool failure strings that vLLM and SGLang write to serve.log.
OOM_MARKERS: tuple[str, ...] = (
    "No available memory for the cache blocks",
    "Not enough memory. Please try to increase --mem-fraction-static",
    "torch.OutOfMemoryError",
    "CUDA out of memory",
    "Prefill out of memory",
    "Decode out of memory",
)


class CapacityRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    engine: Literal["vllm", "sglang"]
    run_id: str
    max_model_len: int
    max_num_seqs: int
    max_batched_tokens: int
    eviction_policy: str
    preemption_policy: str
    concurrency: int
    prompt_tokens: int
    max_tokens: int
    requests_sent: int
    requests_completed: int
    peak_gpu_mib: float | None
    peak_kv_usage: float | None
    preemptions: float | None
    ttft_p50_s: float | None
    ttft_p99_s: float | None
    e2e_p50_s: float | None
    e2e_p99_s: float | None
    accept_length: float | None
    outcome: Outcome
    detail: str
