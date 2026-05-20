"""Event log for an episode."""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from ..core.ids import make_event_id
from .models import EventLog


class EventLogger:
    def __init__(self, episode_id: str) -> None:
        self.episode_id = episode_id
        self._events: List[EventLog] = []

    def _ts(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S")

    def log(
        self,
        actor: str,
        event_type: str,
        tool_name: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
        result: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EventLog:
        ev = EventLog(
            event_id=make_event_id(self.episode_id, len(self._events)),
            episode_id=self.episode_id,
            actor=actor,
            event_type=event_type,
            tool_name=tool_name,
            args=args,
            result=result,
            timestamp=self._ts(),
            metadata=metadata or {},
        )
        self._events.append(ev)
        return ev

    @property
    def events(self) -> List[EventLog]:
        return list(self._events)

    def find(self, predicate: Callable[[EventLog], bool]) -> List[EventLog]:
        return [e for e in self._events if predicate(e)]

    def to_list(self) -> List[Dict[str, Any]]:
        return [e.model_dump() for e in self._events]
