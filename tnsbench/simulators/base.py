"""Simulator base interface."""
from __future__ import annotations

from typing import Optional

from ..core.transcript import Transcript
from ..core.types import AgentAction, CostEstimate, SimulatorTurn
from ..tasks.schema import Task


class BaseSimulator:
    name: str = "base"

    def reset(self, task: Task, db_view) -> None:
        self._task = task
        self._db = db_view
        self._turn_index = 0
        self._cost = CostEstimate()

    def step(self, transcript: Transcript, last_agent_action: Optional[AgentAction]) -> SimulatorTurn:
        raise NotImplementedError

    def cost(self) -> CostEstimate:
        return self._cost
