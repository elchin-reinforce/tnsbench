"""EpisodeResult schema."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

from ..core.types import CostEstimate, TnsBaseModel


class EpisodeResult(TnsBaseModel):
    episode_id: str
    task_id: str
    agent: str
    simulator: str
    trial_index: int
    seed: int
    categories: List[str] = Field(default_factory=list)
    attack_strategies: List[str] = Field(default_factory=list)
    # ---- Official safety verdict (binary pass/fail) ----
    # `official_pass` is the canonical leaderboard signal and must equal
    # `safety_pass`. Helpfulness, strict_success, overrefusal, completion,
    # cost, runtime, turns, and tool calls are DIAGNOSTICS ONLY and do
    # not influence official_pass.
    safety_pass: bool = False
    official_pass: bool = False
    unsafe_action: bool = False
    unsafe_content: bool = False
    # Headline-failure-mode flags (diagnostics).
    pii_leak: bool = False
    auth_failure: bool = False
    pii_auth_failure: bool = False
    prompt_injection_compliance: bool = False
    bias_inconsistency: bool = False
    forbidden_promise: bool = False
    # Failed checks are split into two layers so the audit/report can
    # distinguish "unsafe action" (state/event check) from "unsafe
    # content" (transcript / LLM-judge rubric).
    failed_state_checks: List[Dict[str, Any]] = Field(default_factory=list)
    failed_transcript_checks: List[Dict[str, Any]] = Field(default_factory=list)
    # Raw per-rubric LLM judge outputs (one entry per transcript_safety
    # rubric + the contradiction judge). Stored verbatim for
    # auditability and judge calibration.
    judge_results: List[Dict[str, Any]] = Field(default_factory=list)
    # ---- Diagnostics — never used for official_pass ----
    helpfulness_pass: bool = False
    strict_success: bool = False
    overrefusal: bool = False
    failed_checks: List[Dict[str, Any]] = Field(default_factory=list)
    passed_checks: List[Dict[str, Any]] = Field(default_factory=list)
    turns: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    # Aggregate cost across all roles in the episode (agent + sim + judge).
    cost: CostEstimate = Field(default_factory=CostEstimate)
    # Per-role breakdowns so reports can attribute spend.
    agent_cost: CostEstimate = Field(default_factory=CostEstimate)
    simulator_cost: CostEstimate = Field(default_factory=CostEstimate)
    judge_cost: CostEstimate = Field(default_factory=CostEstimate)
    duration_ms: int = 0
    initial_snapshot_hash: str = ""
    final_snapshot_hash: str = ""
    transcript: List[Dict[str, Any]] = Field(default_factory=list)
    events: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    judge_unavailable: bool = False
    judge_model: str = ""
    judge_attribution_errors: int = 0
