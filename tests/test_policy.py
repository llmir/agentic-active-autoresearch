import json
from typing import Any

import pytest

from agentic_al.config import AgentConfig, GuardrailConfig
from agentic_al.policy import (
    OpenAICompatiblePolicy,
    RuleBasedPolicy,
    _NoRedirectHandler,
    _parse_json_object,
    _validate_base_url,
    apply_guardrails,
    build_policy,
)
from agentic_al.types import PolicyContext, StrategyProposal

ANCHOR = {"uncertainty": 0.5, "diversity": 0.3, "random": 0.15, "target": 0.05}


def _context() -> PolicyContext:
    return PolicyContext(
        round_index=0,
        total_rounds=3,
        labeled_size=20,
        pool_size=100,
        batch_size=10,
        remaining_budget=30,
        task="regression",
        metrics={"mae": 1.0, "uncertainty_error_spearman": 0.01},
    )


def test_external_weights_are_clipped_and_unknown_is_removed() -> None:
    result = apply_guardrails(
        StrategyProposal(
            weights={"uncertainty": 1.0, "unknown": 10.0},
            rationale="test",
            source="openai_compatible",
        ),
        ANCHOR,
        GuardrailConfig(max_agent_delta=0.1),
    )
    assert "unknown" not in result.proposal.weights
    assert abs(sum(result.proposal.weights.values()) - 1.0) < 1e-12
    assert result.repairs


def test_rule_policy_shifts_when_uncertainty_is_weak() -> None:
    proposal = RuleBasedPolicy(ANCHOR).propose(_context())
    assert proposal.weights["uncertainty"] < ANCHOR["uncertainty"]
    assert proposal.weights["diversity"] > ANCHOR["diversity"]


def test_missing_provider_environment_uses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("AGENTIC_AL_API_BASE", "AGENTIC_AL_API_KEY", "AGENTIC_AL_MODEL"):
        monkeypatch.delenv(name, raising=False)
    policy = OpenAICompatiblePolicy(
        AgentConfig(enabled=True, provider="openai_compatible"), RuleBasedPolicy(ANCHOR)
    )
    proposal = policy.propose(_context())
    assert proposal.fallback_used is True
    assert proposal.source == "rule_based"
    assert "API" in proposal.fallback_reason


def test_guardrails_repair_invalid_and_disabled_paths() -> None:
    invalid = apply_guardrails(
        StrategyProposal(
            weights={"random": float("nan"), "uncertainty": -2.0},
            rationale="invalid",
            source="rule_based",
        ),
        ANCHOR,
        GuardrailConfig(),
    )
    assert invalid.proposal.weights == ANCHOR
    assert "restored_anchor_weights" in invalid.repairs

    disabled = apply_guardrails(
        StrategyProposal(weights={"random": 3, "diversity": 1}, rationale="ok", source="custom"),
        ANCHOR,
        GuardrailConfig(enabled=False),
    )
    assert disabled.proposal.weights == {"random": 0.75, "diversity": 0.25}


def test_json_and_url_validation_helpers() -> None:
    assert _parse_json_object('```json\n{"weights":{"random":1}}\n```')["weights"] == {"random": 1}
    with pytest.raises(ValueError, match="root"):
        _parse_json_object("[]")
    _validate_base_url("https://api.example.test/v1", allow_local_http=False)
    _validate_base_url("http://localhost:8000/v1", allow_local_http=True)
    with pytest.raises(ValueError, match="HTTPS"):
        _validate_base_url("http://example.test/v1", allow_local_http=True)
    for unsafe in (
        "https://user:password@example.test/v1",  # pragma: allowlist secret
        "https://example.test/v1?key=value",
        "https://example.test/v1#fragment",
    ):
        with pytest.raises(ValueError, match="credentials"):
            _validate_base_url(unsafe, allow_local_http=False)


def test_openai_compatible_success_uses_aggregate_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTIC_AL_API_BASE", "https://api.example.test/v1")
    monkeypatch.setenv("AGENTIC_AL_API_KEY", "unit-test-secret")
    monkeypatch.setenv("AGENTIC_AL_MODEL", "test-model")
    captured: dict[str, Any] = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size: int) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {"message": {"content": '{"weights":{"random":1},"rationale":"test"}'}}
                    ]
                }
            ).encode()

    class Opener:
        def open(self, request, timeout):
            captured["payload"] = json.loads(request.data)
            captured["timeout"] = timeout
            return Response()

    monkeypatch.setattr("urllib.request.build_opener", lambda handler: Opener())
    policy = OpenAICompatiblePolicy(
        AgentConfig(enabled=True, provider="openai_compatible"), RuleBasedPolicy(ANCHOR)
    )
    proposal = policy.propose(_context())
    assert proposal.source == "openai_compatible"
    public_context = json.loads(captured["payload"]["messages"][1]["content"])
    assert public_context["pool_size"] == 100
    assert "rows" not in public_context


def test_redirects_are_forbidden_before_following() -> None:
    with pytest.raises(RuntimeError, match="redirects are forbidden"):
        _NoRedirectHandler().redirect_request(None, None, 302, "Found", {}, "https://other.test")


def test_policy_selection_and_fallback_disable(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(build_policy(AgentConfig(), ANCHOR), RuleBasedPolicy)
    monkeypatch.delenv("AGENTIC_AL_API_BASE", raising=False)
    policy = OpenAICompatiblePolicy(
        AgentConfig(
            enabled=True,
            provider="openai_compatible",
            fallback_to_rule_based=False,
        ),
        RuleBasedPolicy(ANCHOR),
    )
    with pytest.raises(RuntimeError, match="requires API"):
        policy.propose(_context())
