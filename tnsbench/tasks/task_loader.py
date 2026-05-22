"""Task loader and JSONL I/O — TnSBench-Hard is adversarial-only.

Benign control / over-refusal calibration tasks have been removed from
TnSBench-Hard. Any caller asking for a benign mode hits a hard error so
no part of the pipeline silently mixes benign tasks into the main
benchmark.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Optional

from ..core.config import (
    ADVERSARIAL_TASKS_PATH,
    TASKS_PATH,
)
from .schema import Task


_BENIGN_REMOVAL_MSG = (
    "Benign/control tasks have been removed from TnSBench-Hard. "
    "The benchmark is adversarial-only. Valid task modes are "
    "'adversarial' and 'all' (both load the same 100 adversarial tasks)."
)


def load_tasks(path: Optional[Path] = None) -> List[Task]:
    p = Path(path or TASKS_PATH)
    if not p.exists():
        return []
    out: List[Task] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        t = Task(**json.loads(line))
        if t.split != "adversarial":
            raise RuntimeError(
                f"Task {t.id} in {p.name} has split={t.split!r}. "
                + _BENIGN_REMOVAL_MSG
            )
        out.append(t)
    return out


def load_adversarial_tasks() -> List[Task]:
    if ADVERSARIAL_TASKS_PATH.exists():
        return load_tasks(ADVERSARIAL_TASKS_PATH)
    return load_tasks()


def load_benign_tasks() -> List[Task]:
    raise RuntimeError(_BENIGN_REMOVAL_MSG)


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
