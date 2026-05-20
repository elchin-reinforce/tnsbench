"""Cost accounting helpers."""
from __future__ import annotations

from typing import Dict

from .types import CostEstimate

# Approximate USD per million tokens — wide ranges, used only for reporting.
# These are illustrative; users should override via provider config.
# USD per 1M tokens. Rates as of 2026-05; verify against provider console.
# Keys are normalized to lowercase "provider:model".
PRICE_TABLE: Dict[str, Dict[str, float]] = {
    # OpenAI
    "openai:gpt-5.5":     {"input":  5.00, "output": 40.00},
    "openai:gpt-4o":      {"input":  2.50, "output": 10.00},
    "openai:gpt-4o-mini": {"input":  0.15, "output":  0.60},
    # DeepInfra
    "deepinfra:deepseek-ai/deepseek-v4-pro":                       {"input": 0.70, "output": 1.20},
    "deepinfra:qwen/qwen3.5-35b-a3b":                              {"input": 0.07, "output": 0.30},
    "deepinfra:uni-ai/qwen3.5-35b-a3b-abliterated-no-reasoning":   {"input": 0.07, "output": 0.30},
    # xAI
    "xai:grok-4-fast-non-reasoning": {"input": 0.20, "output": 0.50},
    # Anthropic (kept for completeness)
    "anthropic:claude-haiku":  {"input":  1.00, "output":  5.00},
    "anthropic:claude-sonnet": {"input":  3.00, "output": 15.00},
    "anthropic:claude-opus":   {"input": 15.00, "output": 75.00},
    # Mock / default
    "mock": {"input": 0.0, "output": 0.0},
}


def estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> CostEstimate:
    key = f"{provider}:{model}".lower()
    rates = PRICE_TABLE.get(key, {"input": 0.0, "output": 0.0})
    usd = (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates[
        "output"
    ]
    return CostEstimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_usd=round(usd, 6),
        provider=provider,
        model=model,
    )


def zero_cost() -> CostEstimate:
    return CostEstimate()


def add_costs(a: CostEstimate, b: CostEstimate) -> CostEstimate:
    return CostEstimate(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        estimated_usd=round(a.estimated_usd + b.estimated_usd, 6),
        provider=a.provider or b.provider,
        model=a.model or b.model,
    )
