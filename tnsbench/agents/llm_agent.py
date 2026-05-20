"""LLM-backed agent using the provider abstraction."""
from __future__ import annotations

import json
from typing import Any, Dict, List

from ..core.cost import add_costs
from ..core.transcript import Transcript
from ..core.types import AgentAction, CostEstimate, MessageAction, ToolCallAction
from .base import BaseAgent
from .prompts import render_agent_system
from .providers import LLMProvider, ProviderConfig


class LLMAgent(BaseAgent):
    name = "llm_agent"

    def __init__(self, provider_config: ProviderConfig) -> None:
        self.provider = LLMProvider(provider_config)
        self._cost = CostEstimate()

    def reset(self, task, policy_text, tools):
        super().reset(task, policy_text, tools)
        self._system = render_agent_system(policy_text, tools.describe_for_prompt())

    def step(self, transcript: Transcript) -> AgentAction:
        self._step += 1
        # Convert transcript to provider-friendly messages.
        messages: List[Dict[str, str]] = []
        for t in transcript:
            if t.role == "user":
                messages.append({"role": "user", "content": t.content or ""})
            elif t.role == "assistant" and t.content:
                messages.append({"role": "assistant", "content": t.content})
            elif t.role == "assistant" and t.tool_name:
                messages.append({"role": "assistant", "content": json.dumps({"type": "tool_call", "tool_name": t.tool_name, "args": t.args or {}})})
            elif t.role == "tool":
                messages.append({"role": "user", "content": f"[tool:{t.tool_name}] {json.dumps(t.tool_result, default=str)}"})

        text, cost = self.provider.complete(self._system, messages)
        self._cost = add_costs(self._cost, cost)

        return _parse_action(text)


def _parse_action(text: str) -> AgentAction:
    text = (text or "").strip()
    if text.startswith("```"):
        # Remove markdown fences
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    # Find first JSON object
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        obj = json.loads(text[start:end])
    except Exception:
        return MessageAction(content=text)
    t = (obj or {}).get("type", "message")
    if t == "tool_call":
        return ToolCallAction(tool_name=obj.get("tool_name", ""), args=obj.get("args", {}) or {})
    return MessageAction(content=obj.get("content", ""))
