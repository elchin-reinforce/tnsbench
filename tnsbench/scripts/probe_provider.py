"""Quickly verify a provider+model is reachable (1-shot ping)."""
from __future__ import annotations

import argparse
import json
import sys

from tnsbench.agents.providers import LLMProvider, ProviderConfig


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--provider", required=True)
    p.add_argument("--model", required=True)
    args = p.parse_args()

    prov = LLMProvider(ProviderConfig(provider=args.provider, model=args.model, max_tokens=64))
    print(f"resolved base_url={prov.config.base_url} api_key_env={prov.config.api_key_env}")
    if not prov.available():
        print("NOT_AVAILABLE: no API key found in environment")
        return 2
    text, cost = prov.complete(
        "You are a JSON echo. Reply with exactly {\"type\":\"message\",\"content\":\"ok\"}.",
        [{"role": "user", "content": "ping"}],
    )
    print("OK_RESPONSE:", text[:400])
    print("COST:", json.dumps(cost.model_dump()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
