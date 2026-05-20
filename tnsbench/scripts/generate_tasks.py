"""Generate the 100-task tasks.jsonl and fail if linting fails."""
from __future__ import annotations

import sys

from ..tasks.generator import build_all_tasks
from ..tasks.lint import lint_tasks, summarize_distribution
from ..tasks.task_loader import save_tasks


def main() -> int:
    tasks = build_all_tasks()
    save_tasks(tasks)
    ok, errors = lint_tasks(tasks)
    dist = summarize_distribution(tasks)
    print(f"Wrote {len(tasks)} tasks. Distribution: {dist}")
    if not ok:
        print("LINT FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("Lint OK.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
