"""Smoke checks on the public generator_hard API surface.

The generator is the source of truth for the TnSBench-Hard dataset. Tests
here cover what is intended to be public via ``__all__`` plus the wildcard
import pathway (so a stale or invalid name in ``__all__`` fails fast).
"""
from __future__ import annotations

import importlib


def test_wildcard_import_resolves_all_names():
    """`from tnsbench.tasks.generator_hard import *` must not raise.

    Pydantic / typing tooling sometimes leaves stale names in ``__all__``;
    this test catches that early instead of letting the failure surface in a
    benchmark run.
    """
    mod = importlib.import_module("tnsbench.tasks.generator_hard")
    exported = getattr(mod, "__all__", None)
    assert exported, "generator_hard.__all__ must be defined"
    for name in exported:
        assert hasattr(mod, name), (
            f"generator_hard.__all__ lists '{name}' but the module does not "
            "define it"
        )


def test_public_builders_callable():
    from tnsbench.tasks.generator_hard import (
        build_adversarial_tasks_hard,
        build_all_tasks_hard,
        write_split_files_hard,
    )
    assert callable(build_adversarial_tasks_hard)
    assert callable(build_all_tasks_hard)
    assert callable(write_split_files_hard)
