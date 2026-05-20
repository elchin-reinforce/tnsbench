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
    safety_pass: bool = False
    helpfulness_pass: bool = False
    strict_success: bool = False
    overrefusal: bool = False
    unsafe_action: bool = False
    unsafe_content: bool = False
    pii_leak: bool = False
    auth_failure: bool = False
    prompt_injection_compliance: bool = False
    bias_inconsistency: bool = False
    failed_checks: List[Dict[str, Any]] = Field(default_factory=list)
    passed_checks: List[Dict[str, Any]] = Field(default_factory=list)
    turns: int = 0
    tool_calls: int = 0
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
