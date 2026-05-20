"""Load policy text and clauses."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..core.config import CLAUSES_PATH, POLICY_PATH
from .policy_schema import Clauses


def load_policy_text() -> str:
    return Path(POLICY_PATH).read_text(encoding="utf-8")


def load_clauses() -> Clauses:
    data = json.loads(Path(CLAUSES_PATH).read_text(encoding="utf-8"))
    return Clauses(**data)


def policy_hash() -> str:
    h = hashlib.sha256()
    h.update(load_policy_text().encode())
    h.update(Path(CLAUSES_PATH).read_bytes())
    return h.hexdigest()[:16]
