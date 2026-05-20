"""Validate tasks.jsonl against the linter."""
from __future__ import annotations

import sys

from ..tasks.lint import lint_tasks, summarize_distribution
from ..tasks.task_loader import load_tasks


def main() -> int:
    tasks = load_tasks()
    print(f"Loaded {len(tasks)} tasks. Distribution: {summarize_distribution(tasks)}")
    ok, errors = lint_tasks(tasks)
    if not ok:
        print("LINT FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("Lint OK.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
