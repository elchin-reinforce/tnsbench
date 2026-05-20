"""LLM-backed adversarial user simulator with hybrid (scripted-first) mode."""
from __future__ import annotations

import json
from typing import Optional

from ..agents.providers import LLMProvider, ProviderConfig
from ..core.cost import add_costs
from ..core.transcript import Transcript
from ..core.types import AgentAction, CostEstimate, SimulatorTurn
from .base import BaseSimulator
from .placeholders import resolve as resolve_placeholders
from .prompts import render_simulator_system
from .scripted import ScriptedSimulator


def _task_context(task) -> str:
    """Build the prompt context for the LLM-driven user simulator.

    τ-bench-style: the simulator gets a persona, a long-form instruction
    describing what the user wants, escalation notes, and a list of seed
    turns. The simulator improvises naturally — it does NOT echo seed turns
    verbatim except possibly as the first message.
    """
    # The new schema (v0.5) drops simulator_persona / simulator_instruction /
    # escalation_notes — the scripted_turns + adversarial_goal +
    # attack_strategies carry the role-play instructions implicitly.
    return json.dumps(
        {
            "task_id": task.id,
            "title": task.title,
            "primary_category": task.primary_category,
            "categories": task.categories,
            "adversarial_goal": task.adversarial_goal,
            "legitimate_user_need": task.legitimate_user_need,
            "allowed_strategies": task.attack_strategies,
            "forbidden_behaviors_for_agent": task.forbidden_behaviors,
            "seed_turns": [t.user_message for t in task.scripted_turns],
            "max_turns": task.max_turns,
        },
        indent=2,
    )


class LLMSimulator(BaseSimulator):
    name = "llm_simulator"

    def __init__(self, provider_config: ProviderConfig, seed_first_turn: bool = True) -> None:
        """
        seed_first_turn:
            If True (default), turn 1 is the task's first scripted turn — a
            safety net for models that return "" on an empty transcript.
            The cascading simulator's *primary* sets this to False so we can
            see Grok's actual turn-1 output and detect refusal at turn 1.
        """
        self.provider = LLMProvider(provider_config)
        self.seed_first_turn = seed_first_turn

    def reset(self, task, db_view):
        super().reset(task, db_view)
        self._system = render_simulator_system(_task_context(task))

    def step(self, transcript: Transcript, last_agent_action: Optional[AgentAction]) -> SimulatorTurn:
        self._turn_index += 1
        # Seed turn 1 from the task's first scripted turn (opt-in).
        if self.seed_first_turn and self._turn_index == 1 and self._task.scripted_turns:
            first = self._task.scripted_turns[0]
            return SimulatorTurn(
                user_message=resolve_placeholders(first.user_message, self._task, self._db),
                strategy=first.strategy or "default",
                should_continue=True,
                simulator_goal_progress="turn 1/seed",
            )
        messages = []
        for t in transcript:
            if t.role == "user":
                messages.append({"role": "assistant", "content": t.content or ""})
            elif t.role == "assistant" and t.content:
                messages.append({"role": "user", "content": t.content})
            elif t.role == "assistant" and t.tool_name:
                messages.append({"role": "user", "content": f"[agent tool_call: {t.tool_name}]"})
            elif t.role == "tool":
                messages.append({"role": "user", "content": f"[tool result: {t.tool_name}]"})

        text, cost = self.provider.complete(self._system, messages or [{"role": "assistant", "content": "(turn 1)"}])
        self._cost = add_costs(self._cost, cost)

        try:
            obj = json.loads(_extract_json(text))
        except Exception:
            obj = {"user_message": text or "Can you help with my request?", "strategy": "default", "should_continue": True}

        # Preserve provider-error markers so the report's provider_errors
        # counter can see them; otherwise they get silently dropped here.
        if not obj.get("user_message") and "[provider_error" in (text or ""):
            obj["user_message"] = text
            obj["strategy"] = "provider_error"
        if self._turn_index >= self._task.max_turns:
            obj["should_continue"] = False
            obj["stop_reason"] = obj.get("stop_reason") or "max_turns"
        # Resolve any leftover credential placeholders from the LLM's output.
        message = resolve_placeholders(obj.get("user_message", "") or "",
                                       self._task, self._db)
        return SimulatorTurn(**{
            "user_message": message,
            "strategy": obj.get("strategy", "default"),
            "should_continue": bool(obj.get("should_continue", True)),
            "simulator_goal_progress": obj.get("simulator_goal_progress", ""),
            "stop_reason": obj.get("stop_reason"),
        })


class HybridSimulator(BaseSimulator):
    """Uses scripted for first N turns, then LLM for adaptive pressure."""

    name = "hybrid"

    def __init__(self, provider_config: ProviderConfig, scripted_turns_first: int = 3) -> None:
        self._scripted = ScriptedSimulator()
        self._llm = LLMSimulator(provider_config)
        self._scripted_first = scripted_turns_first

    def reset(self, task, db_view):
        super().reset(task, db_view)
        self._scripted.reset(task, db_view)
        self._llm.reset(task, db_view)

    def step(self, transcript: Transcript, last_agent_action: Optional[AgentAction]) -> SimulatorTurn:
        if self._scripted._turn_index < self._scripted_first and self._scripted._turn_index < len(self._task.scripted_turns):
            return self._scripted.step(transcript, last_agent_action)
        return self._llm.step(transcript, last_agent_action)


def _extract_json(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return text[start:end]
    except ValueError:
        return "{}"
