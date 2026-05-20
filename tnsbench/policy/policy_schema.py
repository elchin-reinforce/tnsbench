"""Schema for the machine-readable clause list."""
from __future__ import annotations

from typing import List

from pydantic import Field

from ..core.types import TnsBaseModel


class Clause(TnsBaseModel):
    id: str
    section: str
    short: str


class Clauses(TnsBaseModel):
    version: str
    clauses: List[Clause] = Field(default_factory=list)

    def ids(self) -> List[str]:
        return [c.id for c in self.clauses]
