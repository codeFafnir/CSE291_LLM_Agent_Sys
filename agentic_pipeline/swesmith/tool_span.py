"""ToolSpan: one agent step's tool call timing, written by agent_entry.py and read by run.py and report.py."""

from pydantic import BaseModel, ConfigDict


class ToolSpan(BaseModel):
    model_config = ConfigDict(frozen=True)
    session_id: str
    step: int
    tool: str
    action_head: str
    executed: bool
    t_step_start: float
    t_tool_start: float | None
    t_tool_end: float | None
    execution_time_s: float
    timed_out: bool
    step_exit_status: str | None
    observation_chars: int
