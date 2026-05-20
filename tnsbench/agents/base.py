"""Agent base interface."""
from __future__ import annotations

from typing import List, Optional

from ..core.transcript import Transcript
from ..core.types import AgentAction, CostEstimate, MessageAction, ToolCallAction
from ..tasks.schema import Task
from ..tools.registry import ToolRegistry


class BaseAgent:
    name: str = "base"

    def reset(self, task: Task, policy_text: str, tools: ToolRegistry) -> None:
        self._task = task
        self._policy = policy_text
        self._tools = tools
        self._step = 0
        self._cost = CostEstimate()

    def step(self, transcript: Transcript) -> AgentAction:
        raise NotImplementedError

    def cost(self) -> CostEstimate:
        return self._cost


__all__ = ["BaseAgent", "MessageAction", "ToolCallAction", "AgentAction"]
