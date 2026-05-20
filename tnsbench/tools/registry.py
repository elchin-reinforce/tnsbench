"""Tool registry — direct Python, no MCP."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..core.types import ToolError, ToolOutput
from ..env.database import RetailDB
from ..env.events import EventLogger
from .schemas import TOOL_DESCRIPTIONS, TOOL_SCHEMAS


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: Dict[str, Any]
    is_write: bool
    func: Callable[..., ToolOutput]


class ToolRegistry:
    def __init__(self, db: RetailDB, events: EventLogger) -> None:
        self.db = db
        self.events = events
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, name: str, func: Callable[..., ToolOutput], is_write: bool) -> None:
        self._tools[name] = ToolSpec(
            name=name,
            description=TOOL_DESCRIPTIONS.get(name, ""),
            schema=TOOL_SCHEMAS.get(name, {"type": "object", "properties": {}}),
            is_write=is_write,
            func=func,
        )

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def specs(self) -> List[ToolSpec]:
        return list(self._tools.values())

    def call(self, name: str, args: Dict[str, Any], actor: str = "agent") -> ToolOutput:
        spec = self._tools.get(name)
        self.events.log(actor=actor, event_type="tool_call", tool_name=name, args=args)
        if spec is None:
            err = ToolError(code="NOT_FOUND", message=f"Tool '{name}' not registered.")
            self.events.log(actor="system", event_type="tool_result", tool_name=name, result=err.model_dump())
            return err
        try:
            result: ToolOutput = spec.func(**(args or {}))
        except TypeError as e:
            result = ToolError(code="INVALID_ARGUMENT", message=str(e))
        except Exception as e:  # pragma: no cover — defensive
            result = ToolError(code="TOOL_EXCEPTION", message=f"{type(e).__name__}: {e}")
        try:
            payload = result.model_dump() if hasattr(result, "model_dump") else result
        except Exception:  # pragma: no cover
            payload = str(result)
        self.events.log(
            actor="system", event_type="tool_result", tool_name=name, result=payload,
            metadata={"is_write": spec.is_write},
        )
        return result

    def describe_for_prompt(self) -> str:
        out: List[str] = []
        for spec in self._tools.values():
            kind = "WRITE" if spec.is_write else "READ"
            out.append(f"- {spec.name} [{kind}]: {spec.description}")
        return "\n".join(out)
