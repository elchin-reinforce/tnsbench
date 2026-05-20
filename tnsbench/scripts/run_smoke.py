"""End-to-end smoke script: generate tasks, validate, run, report."""
from __future__ import annotations

import subprocess
import sys


def run(cmd: list) -> int:
    print("$", " ".join(cmd))
    r = subprocess.run(cmd)
    return r.returncode


def main() -> int:
    steps = [
        ["python3", "-m", "tnsbench.cli", "generate-tasks"],
        ["python3", "-m", "tnsbench.cli", "validate-tasks"],
        ["python3", "-m", "tnsbench.cli", "run",
         "--agent", "mock_safe", "--simulator", "scripted",
         "--tasks", "all", "--trials", "1",
         "--out", "results/mock_safe.jsonl"],
        ["python3", "-m", "tnsbench.cli", "run",
         "--agent", "mock_unsafe", "--simulator", "scripted",
         "--limit", "20", "--trials", "1",
         "--out", "results/mock_unsafe.jsonl"],
        ["python3", "-m", "tnsbench.cli", "report", "results/mock_safe.jsonl",
         "--out", "results/mock_safe_report.md"],
        ["python3", "-m", "tnsbench.cli", "report", "results/mock_unsafe.jsonl",
         "--out", "results/mock_unsafe_report.md"],
    ]
    for s in steps:
        rc = run(s)
        if rc != 0:
            return rc
    print("Smoke run complete.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
