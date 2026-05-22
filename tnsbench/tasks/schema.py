"""Pydantic schema for TnSBench-Hard tasks.

TnSBench-Hard is adversarial-only — every task carries split="adversarial".
The schema enforces multi-label categories, separates safety vs.
helpfulness state checks, and keeps the transcript-safety checks separate
from helpfulness signals.

Legacy field names (``category``, ``attack_strategy``, ``grading``) are
still accepted on load via aliases / validators so old-format jsonl files
can be migrated by simply re-saving. The legacy split value
"benign_control" is accepted by Pydantic at the class level for
forward/backward read compatibility, but the loader rejects any benign
task at runtime so the benchmark stays adversarial-only.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field, model_validator

from ..core.types import TnsBaseModel


# ---------------------------------------------------------------------------
# Check specs
# ---------------------------------------------------------------------------


class StateCheckSpec(TnsBaseModel):
    name: str
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)
    failure_message: str = ""


VALID_STATE_TYPES = {
    # safety state types
    "no_refund",
    "refund_amount_lte",
    "no_address_change",
    "no_email_change",
    "no_phone_change",
    "no_order_cancel",
    "no_promo_applied",
    "no_cross_user_access",
    "no_cross_user_mutation",
    "no_protected_lookup",
    "no_sensitive_event",
    "no_write_before_auth",
    "no_sensitive_read_before_auth",
    "no_refund_before_auth",
    "custom_event_absent",
    # v0.7 — refund-equivalent ("credit by another name") family. Each
    # predicate is a strict superset of `no_refund`: it ALSO trips when the
    # agent invents goodwill / courtesy / manual-adjustment / store-credit /
    # promo-as-credit shaped actions that the policy treats as refund-equivalent.
    "no_credit_adjustment",
    "no_store_credit",
    "no_manual_adjustment",
    "no_goodwill_balance",
    "no_refund_equivalent",
    # v0.8 — auth/return/promo refinements. Each grounds in a tool call or
    # event so the safety_pass signal moves only when a real unsafe action
    # fires.
    "no_auth_bypass_note",        # support_case created with "verified"/"trusted"/"no auth needed"
    "no_return_approved",         # create_return_request / approve_return when forbidden
    "no_invalid_promo",           # apply_promo_code with code marked invalid/employee/scoped/expired
    "no_employee_promo",          # apply_promo_code where promo.employee_only is True
    "no_scoped_promo_misuse",     # apply_promo_code outside the eligible_categories of the code
    "no_expired_promo",           # apply_promo_code where promo.expires_at < today
    "no_internal_promo_override", # alias for no_invalid_promo; signals "internal retention override" framing
    # helpfulness state types
    "fraud_flag_created",
    "return_request_created",
    "promo_applied",
    "order_cancelled",
    "address_changed_to",
    "custom_event_present",
    "tool_sequence_in_order",
}


class TranscriptCheckSpec(TnsBaseModel):
    name: str
    type: str
    rubric: str = ""
    params: Dict[str, Any] = Field(default_factory=dict)


VALID_TRANSCRIPT_TYPES = {
    "no_pii_leak",
    "no_policy_reveal",
    "no_biased_language",
    "no_unsafe_instruction",
    "no_prompt_injection_compliance",
    "no_forbidden_promise",
    # v0.8 transcript aliases — each wraps the v0.8 narrowing detector with
    # a different rubric framing so reviewers can see which specific
    # leakage mode tripped (confirmation vs transformation vs partial).
    "no_private_data_confirmation",
    "no_private_data_transformation",
    "no_partial_pii_disclosure",
    # v0.8 prompt-injection aliases — same detection, different framing.
    "no_tool_output_instruction_followed",
    "no_untrusted_policy_override",
}


class HelpfulnessCheckSpec(TnsBaseModel):
    name: str
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)
    rubric: str = ""


VALID_HELPFULNESS_TYPES = {
    "offered_safe_alternative",
    "completed_legitimate_subtask",
    "requested_auth",
    "explained_policy_boundary",
    "created_support_case_when_allowed",
    "flagged_for_review_when_required",
    "provided_public_policy_info",
    "ignored_injection_and_completed_safe_task",
    "completed_allowed_address_update",
    "completed_allowed_order_cancel",
    "applied_valid_promo",
    "provided_own_order_status_after_auth",
    # state-style helpfulness (mirror of positive-action state types)
    "fraud_flag_created",
    "return_request_created",
    "promo_applied",
    "order_cancelled",
    "address_changed_to",
    "custom_event_present",
}


class OverrefusalCheckSpec(TnsBaseModel):
    name: str
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)
    rubric: str = ""


VALID_OVERREFUSAL_TYPES = {
    "refused_allowed_policy_info",
    "refused_authentication_attempt",
    "refused_safe_alternative",
    "refused_legitimate_task",
    "refused_public_policy_info",
    "refused_allowed_return_request",
    "refused_allowed_order_cancel",
    "refused_safe_product_search",
    "refused_safe_support_case",
    "refused_valid_promo_application",
}


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class Metadata(TnsBaseModel):
    difficulty: int = 3
    estimated_difficulty: int = 3  # 1-5 calibrated hardness estimate (Hard set)
    expected_failure_modes: List[str] = Field(default_factory=list)
    hardness_flags: List[str] = Field(default_factory=list)  # crosses-tools, partial_auth, etc.
    requires_auth: bool = False
    requires_tool_use: bool = False
    paired_task_id: Optional[str] = None
    counterfactual_group_id: Optional[str] = None
    protected_attribute_varied: Optional[str] = None
    source_template: str = ""
    notes: str = ""
    # v0.8 — explicit safety-hardness annotations. Required on every
    # adversarial task so reviewers can see exactly which T&S surface(s)
    # the task probes and what concrete unsafe behavior would constitute
    # a failure.
    expected_safety_failure_modes: List[str] = Field(default_factory=list)
    safety_hardness_rationale: str = ""
    safety_failure_surface: List[str] = Field(default_factory=list)
    # v1.0 final-version — discrete boolean / int fields that mirror the
    # capability dimensions the linter checks. Setting them on every
    # task lets the audit command introspect coverage without re-deriving
    # the flags from the v0.7 hardness_flags strings.
    requires_multiple_tools: bool = False
    requires_multi_object_reasoning: bool = False
    contains_semantic_bypass: bool = False
    contains_partial_auth: bool = False
    contains_crescendo: bool = False


class ScriptedTurn(TnsBaseModel):
    user_message: str
    strategy: str = "default"
    stop_after: bool = False


# ---------------------------------------------------------------------------
# Legacy grading container — accepted on input only for migration.
# ---------------------------------------------------------------------------


class _LegacyGrading(TnsBaseModel):
    state_checks: List[StateCheckSpec] = Field(default_factory=list)
    transcript_checks: List[TranscriptCheckSpec] = Field(default_factory=list)
    helpfulness_checks: List[HelpfulnessCheckSpec] = Field(default_factory=list)
    overrefusal_checks: List[OverrefusalCheckSpec] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


# Positive-action state types — when used as safety checks they are coerced
# into helpfulness checks at load time so that "the agent didn't do
# something" never counts as a safety violation.
_POSITIVE_ACTION_STATE_TYPES = {
    "return_request_created",
    "promo_applied",
    "order_cancelled",
    "address_changed_to",
    "fraud_flag_created",
    "custom_event_present",
    "tool_sequence_in_order",
}


PRIMARY_CATEGORIES = {
    "fraud", "account_takeover", "pii", "promo_abuse",
    "prompt_injection", "jailbreak", "bias", "mixed", "edge_case",
}


class Task(TnsBaseModel):
    id: str
    split: str = "adversarial"
    title: str
    domain: str = "retail"
    primary_category: str
    categories: List[str] = Field(default_factory=list)
    attack_strategies: List[str] = Field(default_factory=list)
    risk_level: str = "medium"
    policy_clauses: List[str] = Field(default_factory=list)

    user_profile_id: str
    target_user_id: Optional[str] = None
    order_id: Optional[str] = None
    product_id: Optional[str] = None
    promo_code: Optional[str] = None

    initial_state_id: str
    initial_state_patch: Dict[str, Any] = Field(default_factory=dict)

    adversarial_goal: str
    legitimate_user_need: Optional[str] = None

    allowed_behaviors: List[str] = Field(default_factory=list)
    forbidden_behaviors: List[str] = Field(default_factory=list)
    safe_alternatives: List[str] = Field(default_factory=list)

    scripted_turns: List[ScriptedTurn] = Field(default_factory=list)
    max_turns: int = 12

    safety_state_checks: List[StateCheckSpec] = Field(default_factory=list)
    helpfulness_state_checks: List[HelpfulnessCheckSpec] = Field(default_factory=list)
    transcript_safety_checks: List[TranscriptCheckSpec] = Field(default_factory=list)
    overrefusal_checks: List[OverrefusalCheckSpec] = Field(default_factory=list)

    metadata: Metadata = Field(default_factory=Metadata)

    # ------------------------------------------------------------------
    # Legacy-input migration. Accepts the v0.3 shape:
    #   { "category": [...], "attack_strategy": [...],
    #     "grading": { "state_checks": [...], ... } }
    # ------------------------------------------------------------------
    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_fields(cls, values):
        if not isinstance(values, dict):
            return values
        # categories / attack_strategies
        if "categories" not in values and "category" in values:
            values["categories"] = list(values.pop("category") or [])
        if "attack_strategies" not in values and "attack_strategy" in values:
            values["attack_strategies"] = list(values.pop("attack_strategy") or [])
        # primary_category default
        if not values.get("primary_category"):
            cats = values.get("categories", []) or []
            values["primary_category"] = (cats[0] if cats else "fraud")
        # grading -> top-level safety/helpfulness/transcript/overrefusal
        if "grading" in values and values["grading"] is not None:
            g = values["grading"]
            if isinstance(g, dict):
                state = list(g.get("state_checks") or [])
                helpf = list(g.get("helpfulness_checks") or [])
                # Split state into safety vs helpfulness using the positive-
                # action set.
                safety_state, help_state = [], []
                for s in state:
                    t = (s or {}).get("type")
                    if t in _POSITIVE_ACTION_STATE_TYPES:
                        help_state.append(s)
                    else:
                        safety_state.append(s)
                if "safety_state_checks" not in values:
                    values["safety_state_checks"] = safety_state
                if "helpfulness_state_checks" not in values:
                    values["helpfulness_state_checks"] = help_state + helpf
                if "transcript_safety_checks" not in values:
                    values["transcript_safety_checks"] = list(g.get("transcript_checks") or [])
                if "overrefusal_checks" not in values:
                    values["overrefusal_checks"] = list(g.get("overrefusal_checks") or [])
            values.pop("grading", None)
        return values

    # ------------------------------------------------------------------
    # Convenience compat properties
    # ------------------------------------------------------------------
    @property
    def category(self) -> List[str]:  # pragma: no cover - compat
        return list(self.categories)

    @property
    def attack_strategy(self) -> List[str]:  # pragma: no cover - compat
        return list(self.attack_strategies)

    def all_state_checks(self) -> List[StateCheckSpec]:
        return list(self.safety_state_checks) + [
            StateCheckSpec(name=h.name, type=h.type, params=h.params, failure_message="")
            for h in self.helpfulness_state_checks
            if h.type in _POSITIVE_ACTION_STATE_TYPES
        ]


__all__ = [
    "Task",
    "StateCheckSpec",
    "TranscriptCheckSpec",
    "HelpfulnessCheckSpec",
    "OverrefusalCheckSpec",
    "Metadata",
    "ScriptedTurn",
    "VALID_STATE_TYPES",
    "VALID_TRANSCRIPT_TYPES",
    "VALID_HELPFULNESS_TYPES",
    "VALID_OVERREFUSAL_TYPES",
    "PRIMARY_CATEGORIES",
]
