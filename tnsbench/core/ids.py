"""Deterministic ID generation."""
from __future__ import annotations

import hashlib
import uuid
from typing import Optional


def make_episode_id(task_id: str, trial_index: int, seed: int) -> str:
    h = hashlib.sha256(f"{task_id}|{trial_index}|{seed}".encode()).hexdigest()[:12]
    return f"ep_{task_id}_{trial_index}_{h}"


def make_event_id(episode_id: str, idx: int) -> str:
    return f"{episode_id}_evt_{idx:04d}"


def short_uuid() -> str:
    return uuid.uuid4().hex[:10]


def hash_obj(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def stable_id(prefix: str, key: str) -> str:
    return f"{prefix}_{hashlib.sha256(key.encode()).hexdigest()[:10]}"


def make_return_id(order_id: str, item_id: str, counter: Optional[int] = None) -> str:
    base = f"{order_id}|{item_id}|{counter or 0}"
    return f"ret_{hashlib.sha256(base.encode()).hexdigest()[:10]}"


def make_case_id(seed: str) -> str:
    return f"case_{hashlib.sha256(seed.encode()).hexdigest()[:10]}"
