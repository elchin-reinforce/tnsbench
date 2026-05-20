"""Deterministic random helpers."""
from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Iterator


def seeded_random(seed: int) -> random.Random:
    return random.Random(seed)


@contextmanager
def temp_seed(seed: int) -> Iterator[random.Random]:
    """Context manager that yields a fresh seeded Random instance."""
    rng = random.Random(seed)
    try:
        yield rng
    finally:
        pass
