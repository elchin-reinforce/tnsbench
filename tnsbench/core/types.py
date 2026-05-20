"""Shared base types used across TnSBench."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class TnsBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ToolError(TnsBaseModel):
    """Returned by tools when something fails."""

    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(TnsBaseModel):
    """Successful tool result."""

    ok: bool = True
    data: Any = None


ToolOutput = Union[ToolResult, ToolError]


class AgentAction(TnsBaseModel):
    """Base class for agent actions."""

    type: str


class MessageAction(AgentAction):
    type: str = "message"
    content: str


class ToolCallAction(AgentAction):
    type: str = "tool_call"
    tool_name: str
    args: Dict[str, Any] = Field(default_factory=dict)


class TranscriptTurn(TnsBaseModel):
    role: str  # "user", "assistant", "tool"
    content: Optional[str] = None
    tool_name: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    tool_result: Optional[Any] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SimulatorTurn(TnsBaseModel):
    user_message: str
    strategy: str = "default"
    should_continue: bool = True
    simulator_goal_progress: str = ""
    stop_reason: Optional[str] = None


class CostEstimate(TnsBaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_usd: float = 0.0
    provider: Optional[str] = None
    model: Optional[str] = None


__all__ = [
    "TnsBaseModel",
    "ToolError",
    "ToolResult",
    "ToolOutput",
    "AgentAction",
    "MessageAction",
    "ToolCallAction",
    "TranscriptTurn",
    "SimulatorTurn",
    "CostEstimate",
]
