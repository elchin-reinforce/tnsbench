"""Scripted (deterministic) simulator."""
from __future__ import annotations

from typing import Optional

from ..core.transcript import Transcript
from ..core.types import AgentAction, SimulatorTurn
from .base import BaseSimulator
from .placeholders import resolve as resolve_placeholders


class ScriptedSimulator(BaseSimulator):
    name = "scripted"

    def step(self, transcript: Transcript, last_agent_action: Optional[AgentAction]) -> SimulatorTurn:
        if self._turn_index >= len(self._task.scripted_turns):
            return SimulatorTurn(
                user_message="(no further user turns)",
                strategy="end",
                should_continue=False,
                stop_reason="script_exhausted",
            )
        turn = self._task.scripted_turns[self._turn_index]
        self._turn_index += 1
        # Resolve auth-credential placeholders ({user.email}, {wrong_zip},
        # etc.) against the DB the agent is interacting with. This is what
        # makes a "here's my email + zip" turn emit a concrete value the
        # agent has to actually validate against tool output.
        message = resolve_placeholders(turn.user_message, self._task, self._db)
        # Note: simulator never decides safety pass/fail.
        return SimulatorTurn(
            user_message=message,
            strategy=turn.strategy,
            should_continue=not turn.stop_after,
            simulator_goal_progress=f"turn {self._turn_index}/{len(self._task.scripted_turns)}",
        )
