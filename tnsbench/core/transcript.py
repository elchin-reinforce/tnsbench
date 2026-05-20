"""Transcript helpers."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .types import TranscriptTurn


class Transcript:
    def __init__(self) -> None:
        self.turns: List[TranscriptTurn] = []

    def add_user(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.turns.append(TranscriptTurn(role="user", content=content, metadata=metadata or {}))

    def add_assistant_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.turns.append(
            TranscriptTurn(role="assistant", content=content, metadata=metadata or {})
        )

    def add_tool_call(self, tool_name: str, args: Dict[str, Any]) -> None:
        self.turns.append(
            TranscriptTurn(role="assistant", tool_name=tool_name, args=args, content=None)
        )

    def add_tool_result(self, tool_name: str, result: Any) -> None:
        self.turns.append(
            TranscriptTurn(role="tool", tool_name=tool_name, tool_result=result)
        )

    def as_list(self) -> List[Dict[str, Any]]:
        return [t.model_dump() for t in self.turns]

    def assistant_texts(self) -> List[str]:
        return [t.content or "" for t in self.turns if t.role == "assistant" and t.content]

    def tool_calls(self) -> List[TranscriptTurn]:
        return [t for t in self.turns if t.role == "assistant" and t.tool_name]

    def user_messages(self) -> List[str]:
        return [t.content or "" for t in self.turns if t.role == "user" and t.content]

    def __len__(self) -> int:
        return len(self.turns)

    def __iter__(self) -> Iterable[TranscriptTurn]:
        return iter(self.turns)

    def render_for_prompt(self) -> str:
        lines: List[str] = []
        for t in self.turns:
            if t.role == "user":
                lines.append(f"User: {t.content}")
            elif t.role == "assistant" and t.content:
                lines.append(f"Assistant: {t.content}")
            elif t.role == "assistant" and t.tool_name:
                lines.append(f"Assistant tool_call: {t.tool_name}({t.args})")
            elif t.role == "tool":
                lines.append(f"Tool[{t.tool_name}] result: {t.tool_result}")
        return "\n".join(lines)
