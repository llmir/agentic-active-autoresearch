"""Deterministic and OpenAI-compatible strategy policies with hard guardrails."""

from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.request
from dataclasses import replace
from typing import Protocol
from urllib.parse import urlparse

from .artifacts import safe_error
from .config import AgentConfig, GuardrailConfig
from .registry import ACQUISITION_FUNCTIONS
from .types import GuardrailResult, PolicyContext, StrategyProposal


class Policy(Protocol):
    def propose(self, context: PolicyContext) -> StrategyProposal: ...


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        request: object,
        file_pointer: object,
        code: int,
        message: str,
        headers: object,
        new_url: str,
    ) -> None:
        del request, file_pointer, message, headers, new_url
        raise RuntimeError(f"policy endpoint redirects are forbidden (HTTP {code})")


class RuleBasedPolicy:
    def __init__(self, anchor_weights: dict[str, float]) -> None:
        self.anchor_weights = dict(anchor_weights)

    def propose(self, context: PolicyContext) -> StrategyProposal:
        weights = dict(self.anchor_weights)
        correlation = float(context.metrics.get("uncertainty_error_spearman", 0.0))
        if correlation < 0.05 and "uncertainty" in weights:
            shift = min(0.10, weights["uncertainty"])
            weights["uncertainty"] -= shift
            weights["diversity"] = weights.get("diversity", 0.0) + 0.7 * shift
            weights["random"] = weights.get("random", 0.0) + 0.3 * shift
            rationale = "Weak uncertainty-error alignment; shifted weight toward coverage."
        else:
            rationale = "Kept the reproducible anchor mixture."
        return StrategyProposal(weights=weights, rationale=rationale, source="rule_based")


class OpenAICompatiblePolicy:
    def __init__(self, config: AgentConfig, fallback: Policy) -> None:
        self.config = config
        self.fallback = fallback

    def propose(self, context: PolicyContext) -> StrategyProposal:
        try:
            return self._request(context)
        except Exception as error:
            if not self.config.fallback_to_rule_based:
                raise
            fallback = self.fallback.propose(context)
            return replace(
                fallback,
                fallback_used=True,
                fallback_reason=safe_error(error),
                rationale=f"Provider unavailable or invalid; {fallback.rationale}",
            )

    def _request(self, context: PolicyContext) -> StrategyProposal:
        allowed = sorted(ACQUISITION_FUNCTIONS)
        system_prompt = (
            "You propose active-learning acquisition weights. Return one JSON object only with "
            "keys 'weights' and 'rationale'. Weights must be non-negative and use only these "
            f"components: {allowed}. Do not request tools, code execution, more data, or a budget change."
        )
        parsed = request_policy_json(self.config, system_prompt, context.to_public_dict())
        weights = parsed.get("weights")
        if not isinstance(weights, dict):
            raise ValueError("policy response 'weights' must be an object")
        rationale = str(parsed.get("rationale", "No rationale supplied."))[:500]
        return StrategyProposal(
            weights={str(name): float(value) for name, value in weights.items()},
            rationale=rationale,
            source="openai_compatible",
        )


def request_policy_json(
    config: AgentConfig, system_prompt: str, public_context: dict[str, object]
) -> dict[str, object]:
    """Send aggregate-only context to one OpenAI-compatible JSON policy endpoint."""

    base_url = os.environ.get(config.base_url_env, "").rstrip("/")
    api_key = os.environ.get(config.api_key_env, "")
    model = config.model or os.environ.get(config.model_env, "")
    if not base_url or not api_key or not model:
        raise RuntimeError(
            "OpenAI-compatible policy requires API base, key, and model environment settings"
        )
    _validate_base_url(base_url, allow_local_http=config.allow_local_http)
    endpoint = (
        base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
    )
    payload = {
        "model": model,
        "temperature": config.temperature,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(public_context, sort_keys=True)},
        ],
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        opener = urllib.request.build_opener(_NoRedirectHandler())
        with opener.open(request, timeout=config.timeout_seconds) as response:
            body = response.read(config.max_response_bytes + 1)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"policy endpoint returned HTTP {error.code}") from error
    if len(body) > config.max_response_bytes:
        raise ValueError("policy response exceeded max_response_bytes")
    response_payload = json.loads(body.decode("utf-8"))
    content = response_payload["choices"][0]["message"]["content"]
    return _parse_json_object(content)


def apply_guardrails(
    proposal: StrategyProposal,
    anchor_weights: dict[str, float],
    config: GuardrailConfig,
) -> GuardrailResult:
    original = dict(proposal.weights)
    if not config.enabled:
        return GuardrailResult(
            proposal=replace(proposal, weights=_normalize_weights(original)),
            original_weights=original,
            repairs=[],
        )
    repairs: list[str] = []
    weights: dict[str, float] = {}
    is_external = proposal.source == "openai_compatible"
    for name, raw_value in original.items():
        normalized_name = name.strip().lower().replace("-", "_")
        if normalized_name not in ACQUISITION_FUNCTIONS:
            repairs.append(f"dropped_unknown_component:{normalized_name}")
            continue
        value = float(raw_value)
        if not math.isfinite(value) or value < 0:
            repairs.append(f"repaired_invalid_weight:{normalized_name}")
            value = 0.0
        if is_external:
            anchor = float(anchor_weights.get(normalized_name, 0.0))
            low = max(0.0, anchor - config.max_agent_delta)
            high = min(config.max_component_weight, anchor + config.max_agent_delta)
        else:
            low, high = 0.0, config.max_component_weight
        clipped = min(max(value, low), high)
        if abs(clipped - value) > 1e-12:
            repairs.append(f"clipped_weight:{normalized_name}:{value:.6g}->{clipped:.6g}")
        weights[normalized_name] = clipped
    if not weights or sum(weights.values()) <= 0:
        repairs.append("restored_anchor_weights")
        weights = dict(anchor_weights)
    normalized = _normalize_weights(weights)
    guarded = replace(proposal, weights=normalized)
    return GuardrailResult(
        proposal=guarded,
        original_weights=original,
        repairs=repairs,
    )


def build_policy(config: AgentConfig, anchor_weights: dict[str, float]) -> Policy:
    fallback = RuleBasedPolicy(anchor_weights)
    if config.enabled and config.provider == "openai_compatible":
        return OpenAICompatiblePolicy(config, fallback)
    return fallback


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    positive = {name: max(0.0, float(value)) for name, value in weights.items()}
    total = sum(positive.values())
    if total <= 0:
        raise ValueError("strategy weights must contain a positive value")
    return {name: value / total for name, value in positive.items()}


def _parse_json_object(content: str) -> dict[str, object]:
    cleaned = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.I)
    if fenced:
        cleaned = fenced.group(1)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("policy response root must be an object")
    return parsed


def _validate_base_url(base_url: str, *, allow_local_http: bool) -> None:
    parsed = urlparse(base_url)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("API base must not contain credentials, a query, or a fragment")
    if parsed.scheme == "https" and parsed.netloc:
        return
    local_hosts = {"127.0.0.1", "localhost", "::1"}
    if (
        allow_local_http
        and parsed.scheme == "http"
        and parsed.hostname in local_hosts
        and parsed.netloc
    ):
        return
    raise ValueError("API base must use HTTPS; HTTP is allowed only for localhost")
