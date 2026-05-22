"""Deprecated. Use `python -m tnsbench.cli generate-tasks` instead.

TnSBench-Hard is generated only from tnsbench.tasks.generator_hard. This
shim previously called the legacy benign-inclusive generator and is now
intentionally a hard error so accidental invocation cannot rewrite the
adversarial dataset.
"""
from __future__ import annotations

import sys

_DEPRECATION = (
    "tnsbench.scripts.generate_tasks is removed. TnSBench-Hard is "
    "adversarial-only and is generated only from "
    "tnsbench.tasks.generator_hard via "
    "`python -m tnsbench.cli generate-tasks`."
)


def main() -> int:
    raise SystemExit(_DEPRECATION)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
