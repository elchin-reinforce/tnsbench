"""Report-format tests — v0.8 Trust & Safety-realigned structure.

The headline leaderboard must contain ONLY the three T&S metrics:
    * Safety Pass ↑
    * Unsafe Tool Action ↓
    * PII/Auth Failure ↓

Strict success, helpfulness, over-refusal, cost, turns must NOT appear
in the main leaderboard (sections 1-2). They are allowed in the
diagnostics appendix (section 6).
"""
from __future__ import annotations

from pathlib import Path

from tnsbench.runner.aggregate import aggregate
from tnsbench.runner.report import render_markdown
from tnsbench.runner.run import run_benchmark


def _run_small():
    out = Path("results/_test_smoke.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    results = run_benchmark("mock_safe", "scripted", "adversarial",
                            trials=1, limit=8, out_path=out)
    assert results
    return aggregate(results), results


def test_main_leaderboard_has_only_three_headline_metrics():
    """Sections 1-2 are the leaderboard. They MUST mention only the three
    T&S headline metrics, NOT strict_success / helpfulness / overrefusal /
    cost / turns / tool calls."""
    agg, _ = _run_small()
    md = render_markdown(agg)
    leaderboard = md.split("## 3.")[0]
    for label in ["Safety Pass", "Unsafe Tool Action", "PII/Auth Failure"]:
        assert label in leaderboard, f"missing headline metric: {label}"
    for label in ["Strict Success", "Helpful Safe Completion",
                  "Over-refusal", "total_cost", "total_tokens",
                  "Avg scripted turns", "Avg tool calls"]:
        assert label not in leaderboard, (
            f"diagnostic '{label}' must not appear in headline sections."
        )


def test_report_section_structure():
    agg, _ = _run_small()
    md = render_markdown(agg)
    # Final-version section structure. New sections were inserted after
    # the per-category block (per-domain, tool-use stats, robustness,
    # judge summary), so the diagnostics appendix moved from section 6
    # to section 9. All required sections are still present and the
    # main leaderboard is still section 2.
    for section in [
        "## 1. Summary",
        "## 2. Main leaderboard",
        "## 3. Safety breakdown",
        "## 4. Per-category Safety Pass",
        "## 5. Most common safety failures",
        "## 9. Diagnostics appendix",
    ]:
        assert section in md, f"missing section: {section}"


def test_diagnostics_appendix_contains_strict_success():
    """Strict success is still computed and reported — but only in the
    section 6 diagnostics appendix, never in the headline."""
    agg, _ = _run_small()
    md = render_markdown(agg)
    appendix = md.split("## 9.")[1] if "## 9." in md else ""
    assert "Strict Success Rate" in appendix
    assert "Helpful Safe Completion Rate" in appendix
    assert "Over-refusal Rate" in appendix


def test_three_headline_metric_definitions():
    """safety_pass_rate / unsafe_tool_action_rate / pii_auth_failure_rate
    must be derivable from the underlying booleans."""
    agg, results = _run_small()
    o = agg["overall"]
    expected_safety = sum(
        1 for r in results
        if (not r.unsafe_action) and (not r.unsafe_content)
    ) / len(results)
    assert abs(o["safety_pass_rate"] - round(expected_safety, 4)) < 1e-9
    expected_action = sum(1 for r in results if r.unsafe_action) / len(results)
    assert abs(o["unsafe_tool_action_rate"] - round(expected_action, 4)) < 1e-9
    expected_pii_auth = sum(
        1 for r in results if r.pii_leak or r.auth_failure
    ) / len(results)
    assert abs(o["pii_auth_failure_rate"] - round(expected_pii_auth, 4)) < 1e-9


def test_safety_breakdown_present():
    """The aggregate must expose a `safety_breakdown` dict that the
    report renders in section 3."""
    agg, _ = _run_small()
    sb = agg.get("safety_breakdown") or {}
    for k in ("prompt_injection_failure_rate", "refund_fraud_failure_rate",
              "account_takeover_failure_rate", "pii_failure_rate",
              "bias_failure_rate", "unsafe_content_rate"):
        assert k in sb, f"safety_breakdown missing key '{k}'"


def test_common_safety_failures_excludes_helpfulness():
    """Section 5 reports only safety-kind failures, never helpfulness or
    overrefusal failures (those live in the diagnostics appendix)."""
    agg, _ = _run_small()
    for entry in agg.get("common_safety_failures") or []:
        assert entry["name"].startswith(("safety_state:", "transcript_safety:")), (
            f"common_safety_failures includes non-safety failure: {entry['name']}"
        )
