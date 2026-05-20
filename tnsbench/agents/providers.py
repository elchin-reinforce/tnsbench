"""Provider abstraction for LLM calls.

Supports openai-style chat completions if OPENAI_API_KEY is set; otherwise
gracefully degrades to a deterministic no-op response so tests pass with no
network/key. Real provider calls are kept simple and avoid pulling SDKs we
might not have.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()  # pick up .env at import time if present
except Exception:
    pass

from ..core.cost import estimate_cost
from ..core.types import CostEstimate


# Provider presets: short name -> (base_url, api_key_env).
# All entries use an OpenAI-compatible chat-completions interface.
PROVIDER_PRESETS: Dict[str, Tuple[str, str]] = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "deepinfra": ("https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_KEY"),
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    # convenient aliases
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
}


@dataclass
class ProviderConfig:
    provider: str = "mock"
    model: str = "mock-model"
    temperature: float = 0.0
    # 4096 leaves headroom for reasoning models that consume tokens before
    # emitting visible content (Qwen3.5 A3B, DeepSeek R-style, etc.).
    max_tokens: int = 4096
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    # Capability/effort tier for reasoning-capable models. Accepted values:
    # "minimal" / "low" / "medium" / "high" (per OpenAI's reasoning_effort
    # field). Forwarded to OpenAI and OpenRouter (which proxy the same
    # parameter for OpenAI-family models). Ignored for providers that don't
    # support the field — the request still succeeds.
    reasoning_effort: Optional[str] = None

    def resolve(self) -> "ProviderConfig":
        """Fill in base_url / api_key_env from PROVIDER_PRESETS if matched."""
        preset = PROVIDER_PRESETS.get(self.provider.lower())
        if preset:
            base, key_env = preset
            if self.base_url is None:
                self.base_url = base
            # Allow caller-supplied env name to override preset.
            if self.api_key_env == "OPENAI_API_KEY" and self.provider.lower() != "openai":
                self.api_key_env = key_env
        return self


class LLMProvider:
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config.resolve()

    def available(self) -> bool:
        if self.config.provider == "mock":
            return True
        # If a preset is known, prefer its key env; otherwise fall back to OPENAI_API_KEY.
        preset = PROVIDER_PRESETS.get(self.config.provider.lower())
        envs = []
        if preset:
            envs.append(preset[1])
        envs.append(self.config.api_key_env)
        envs.append("OPENAI_API_KEY")
        return any(os.environ.get(e) for e in envs)

    def complete(self, system: str, messages: List[Dict[str, str]]) -> Tuple[str, CostEstimate]:
        """Return (text, cost). For mock providers, returns a refusal-style
        canned reply. For openai-compatible providers, calls the HTTP API."""
        if self.config.provider == "mock":
            return self._mock_reply(messages), CostEstimate()
        api_key = (
            os.environ.get(self.config.api_key_env)
            or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            return self._mock_reply(messages), CostEstimate()
        try:
            return self._call_openai_compat(api_key, system, messages)
        except Exception as e:  # pragma: no cover — network failure
            detail = str(e)
            body = getattr(e, "_tnsbench_body", "")
            if body:
                detail = f"{detail} body={body[:400]}"
            return (
                json.dumps({"type": "message", "content": f"[provider_error: {type(e).__name__}: {detail}]"}),
                CostEstimate(),
            )

    # ---------------- internal ----------------
    def _mock_reply(self, messages: List[Dict[str, str]]) -> str:
        # Always emit a safe message: "Could you please authenticate by email and zip code?"
        return json.dumps(
            {
                "type": "message",
                "content": (
                    "To help you, I need to verify your identity. Please share your email "
                    "and the zip code on file (or name + zip code + last 4 of phone)."
                ),
            }
        )

    def _call_openai_compat(
        self, api_key: str, system: str, messages: List[Dict[str, str]]
    ) -> Tuple[str, CostEstimate]:  # pragma: no cover — network
        import urllib.request

        url = (self.config.base_url or "https://api.openai.com/v1") + "/chat/completions"
        body: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "system", "content": system}] + messages,
        }
        # OpenAI's newer models (gpt-5+, o-series) require max_completion_tokens,
        # do not accept max_tokens, and only accept temperature=1.
        if self.config.provider.lower() == "openai":
            body["max_completion_tokens"] = self.config.max_tokens
            # Skip temperature for OpenAI to stay compatible across model families.
        else:
            body["max_tokens"] = self.config.max_tokens
            body["temperature"] = self.config.temperature
        # Reasoning-effort tier. OpenAI's gpt-5+ / o-series natively accept
        # this. OpenRouter proxies it for OpenAI-family models. For other
        # providers we still send the field — most will ignore unknown
        # parameters, and the small number that reject will return an HTTP
        # 400 which we already capture as a provider_error.
        if self.config.reasoning_effort:
            body["reasoning_effort"] = self.config.reasoning_effort
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                body_text = e.read().decode("utf-8", errors="replace")
            except Exception:
                body_text = ""
            err = urllib.error.HTTPError(e.url, e.code, e.reason, e.headers, None)
            err._tnsbench_body = body_text  # type: ignore[attr-defined]
            raise err
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        cost = estimate_cost(
            self.config.provider,
            self.config.model,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )
        return text, cost
