"""Task loader and JSONL I/O — split-aware."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Optional

from ..core.config import (
    ADVERSARIAL_TASKS_PATH,
    BENIGN_TASKS_PATH,
    TASKS_PATH,
)
from .schema import Task


def load_tasks(path: Optional[Path] = None) -> List[Task]:
    p = Path(path or TASKS_PATH)
    if not p.exists():
        return []
    out: List[Task] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(Task(**json.loads(line)))
    return out


def load_adversarial_tasks() -> List[Task]:
    if ADVERSARIAL_TASKS_PATH.exists():
        return load_tasks(ADVERSARIAL_TASKS_PATH)
    return [t for t in load_tasks() if t.split == "adversarial"]


def load_benign_tasks() -> List[Task]:
    if BENIGN_TASKS_PATH.exists():
        return load_tasks(BENIGN_TASKS_PATH)
    return [t for t in load_tasks() if t.split == "benign_control"]


def save_tasks(tasks: Iterable[Task], path: Optional[Path] = None) -> None:
    p = Path(path or TASKS_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(t.model_dump_json() + "\n")


def tasks_file_hash(path: Optional[Path] = None) -> str:
    p = Path(path or TASKS_PATH)
    if not p.exists():
        return ""
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
