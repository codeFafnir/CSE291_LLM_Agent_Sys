"""Typed records shared by the SPEED-Bench scripts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

Tier = Literal["easy", "medium"]


class SpeedRow(BaseModel):
    # One row of the prepared SPEED-Bench qualitative parquet.
    question_id: str
    category: str
    sub_category: str | None
    turns: list[str]
    source: str
    src_id: str
    difficulty: str | None
    multiturn: bool


class SubsetRow(BaseModel):
    question_id: str
    category: str
    sub_category: str | None
    tier: Tier
    turns: list[str]
    n_turns: int
    first_turn_tokens: int
    worst_case_context: int
    over_context: bool
    source: str
    src_id: str


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class RequestRow(BaseModel):
    # One chat request (one turn of one prompt). Server-side columns are None above concurrency 1.
    question_id: str
    category: str
    tier: Tier
    turn_index: int
    n_turns: int
    over_context: bool
    start_ts: float
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None
    ttft_s: float | None
    tpot_s: float | None
    itl_mean_s: float | None
    itl_p99_s: float | None
    itl_max_s: float | None
    e2e_s: float
    n_chunks: int
    peak_gpu_mib: float | None
    queue_s: float | None
    prefill_s: float | None
    decode_s: float | None
    spec_drafts: float | None
    spec_draft_tokens: float | None
    spec_accepted: float | None
    spec_accept_len: float | None
    spec_accept_rate: float | None
    spec_accept_pos1: float | None
    spec_accept_pos2: float | None
    spec_accept_pos3: float | None
    error: str | None


class ResponseRecord(BaseModel):
    row: RequestRow
    text: str
    chunk_gaps_s: list[float]


class Sample(BaseModel):
    model_config = ConfigDict(frozen=True)
    ts: float
    gpu_used_mib: float | None
    series: dict[str, float]


class SpecDelta(BaseModel):
    # Raw spec-decode counter deltas between two scrapes.
    drafts: float
    draft_tokens: float
    accepted: float
    accepted_per_position: list[float]


class ServerDelta(BaseModel):
    queue_s: float | None
    prefill_s: float | None
    decode_s: float | None
    spec: SpecDelta
