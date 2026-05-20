"""Consistency lint for the merged policy.

Wired into the task linter. Ensures:
  * every clause in clauses.json appears in retail_policy.md
  * every clause in retail_policy.md has either a *Trace:* line or is in a
    T&S-specific section (the TnSBench-added sections)
  * source files exist
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from ..core.config import CLAUSES_PATH, POLICY_PATH

SOURCES_DIR = Path(__file__).parent / "sources"


def assert_merge_consistency() -> Tuple[bool, List[str]]:
    errors: List[str] = []

    if not (SOURCES_DIR / "tau_bench_retail.md").exists():
        errors.append("Missing policy/sources/tau_bench_retail.md")
    if not (SOURCES_DIR / "amazon_returns_2026.md").exists():
        errors.append("Missing policy/sources/amazon_returns_2026.md")
    if not (SOURCES_DIR / "MERGE_LOG.md").exists():
        errors.append("Missing policy/sources/MERGE_LOG.md")

    import json
    clauses = json.loads(Path(CLAUSES_PATH).read_text())["clauses"]
    policy_text = Path(POLICY_PATH).read_text()

    # Every clause id must appear as a heading in the policy.
    for c in clauses:
        cid = c["id"]
        if f"**{cid}**" not in policy_text:
            errors.append(f"Clause {cid} not found as bold heading in policy.")

    # Every clause line should have a Trace: marker (or be in T&S-specific section).
    clause_lines = re.findall(r"\*\*(?P<id>[A-Z]+-\d+)\*\*:.*", policy_text)
    # Re-find with surrounding context to look for the Trace tag on the same paragraph.
    ts_sections = {"INJECTION", "JAILBREAK", "HARM", "SAFEHELP"}
    for c in clauses:
        cid = c["id"]
        section = cid.split("-")[0]
        # Quick scan: look for the trace marker within ~600 chars of the clause heading.
        idx = policy_text.find(f"**{cid}**")
        if idx == -1:
            continue
        window = policy_text[idx : idx + 600]
        if "*Trace:" not in window and section not in ts_sections:
            errors.append(f"Clause {cid} missing *Trace:* line.")

    return len(errors) == 0, errors
