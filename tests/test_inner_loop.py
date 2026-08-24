import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from agentic_al.config import AppConfig, ModelConfig
from agentic_al.inner_loop import _inner_split_positions, run_inner_loop
from agentic_al.registry import TRAINING_CANDIDATE_GENERATORS
from agentic_al.types import (
    CheckpointCatalog,
    CheckpointReference,
    Prediction,
    TrainingCandidate,
    TrainingCandidateContext,
)


class _FakeModel:
    def __init__(self, config: ModelConfig, fit_calls: list[float]) -> None:
        self.learning_rate = float(config.parameters.get("learning_rate", 0.001))
        self.loss = str(config.parameters.get("loss", "mse"))
        self.fit_calls = fit_calls

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> None:
        del frame, target
        self.fit_calls.append(self.learning_rate)

    def predict(self, frame: pd.DataFrame) -> Prediction:
        mean = np.full(len(frame), self.learning_rate * 100.0)
        return Prediction(mean=mean, uncertainty=np.zeros(len(frame)))

    def featurize(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[["x"]].to_numpy(dtype=float)

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "fake",
            "device": "mps",
            "accelerator_used": True,
            "learning_rate": self.learning_rate,
            "loss": self.loss,
        }


class _TrackingGenerator:
    def __init__(self) -> None:
        self.completed_counts: list[int] = []

    def propose(self, context: TrainingCandidateContext, count: int) -> list[TrainingCandidate]:
        self.completed_counts.append(len(context.completed_trials))
        learning_rate = 0.0008 if len(context.completed_trials) == 1 else 0.0004
        return [
            TrainingCandidate(
                name=f"sequential_candidate_{len(context.completed_trials)}",
                parameters={"learning_rate": learning_rate, "epochs": 999},
                rationale="Generated from completed inner-trial evidence.",
                source="tracking_test",
            )
        ][:count]


def _config(generator: str = "tracking_test", trial_budget: int = 3) -> AppConfig:
    return AppConfig.model_validate(
        {
            "model": {
                "name": "torch_mlp",
                "device": "mps",
                "require_accelerator": True,
                "parameters": {
                    "hidden_dims": [8],
                    "dropout": 0.1,
                    "epochs": 4,
                    "batch_size": 8,
                    "learning_rate": 0.001,
                    "weight_decay": 0.00001,
                    "ensemble_size": 1,
                    "mc_dropout_passes": 1,
                },
            },
            "inner_loop": {
                "candidate_generator": generator,
                "trial_budget": trial_budget,
                "candidates_per_step": 1,
                "min_labeled_size": 12,
            },
        }
    )


def _labeled() -> tuple[pd.DataFrame, np.ndarray]:
    return pd.DataFrame({"x": np.linspace(-1.0, 1.0, 24)}), np.zeros(24)


def test_group_aware_inner_split_has_no_group_overlap() -> None:
    target = np.linspace(0.0, 1.0, 24)
    groups = np.repeat(["scaffold-a", "scaffold-b", "scaffold-c", "scaffold-d"], 6)
    train, validation = _inner_split_positions(
        target,
        task="regression",
        fraction=0.25,
        seed=7,
        groups=groups,
    )
    assert set(groups[train]).isdisjoint(groups[validation])


def test_group_aware_multiclass_inner_split_retains_every_class() -> None:
    target = np.asarray([value for _ in range(8) for value in (0, 1, 2)])
    groups = np.asarray([f"scaffold-{index // 3}" for index in range(len(target))])
    train, validation = _inner_split_positions(
        target,
        task="classification",
        fraction=0.25,
        seed=9,
        groups=groups,
    )
    assert set(target[train]) == {0, 1, 2}
    assert set(target[validation]) == {0, 1, 2}
    assert set(groups[train]).isdisjoint(groups[validation])


def test_sequential_candidate_generation_selection_and_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generator = _TrackingGenerator()
    monkeypatch.setitem(TRAINING_CANDIDATE_GENERATORS, "tracking_test", lambda config: generator)
    fit_calls: list[float] = []
    monkeypatch.setattr(
        "agentic_al.inner_loop.build_model",
        lambda config, **kwargs: _FakeModel(config, fit_calls),
    )
    labeled, target = _labeled()
    result = run_inner_loop(
        _config(),
        labeled=labeled,
        target=target,
        task="regression",
        feature_columns=["x"],
        seed=7,
        round_index=2,
        phase="active_round",
        artifact_dir=tmp_path,
    )

    assert generator.completed_counts == [1, 2]
    assert result.summary["attempted_trials"] == 3
    assert result.summary["candidate_generation_steps"] == 2
    assert result.best_candidate.parameters == {"learning_rate": 0.0004}
    assert fit_calls == [0.001, 0.0008, 0.0004, 0.0004]
    assert (tmp_path / "step_002" / "agent_context.json").is_file()
    plan = json.loads((tmp_path / "step_001" / "trial_001" / "training_plan.json").read_text())
    candidate = plan["candidate"]
    assert candidate["parameters"] == {"learning_rate": 0.0008}
    assert "dropped_non_tunable_parameter:epochs" in candidate["repairs"]
    assert plan["locked_parameters"]["epochs"] == 4
    assert plan["locked_model_contract"]["device"] == "mps"


def test_completed_candidate_trials_are_reused_by_plan_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generator = _TrackingGenerator()
    monkeypatch.setitem(TRAINING_CANDIDATE_GENERATORS, "tracking_test", lambda config: generator)
    fit_calls: list[float] = []
    monkeypatch.setattr(
        "agentic_al.inner_loop.build_model",
        lambda config, **kwargs: _FakeModel(config, fit_calls),
    )
    labeled, target = _labeled()
    arguments = {
        "labeled": labeled,
        "target": target,
        "task": "regression",
        "feature_columns": ["x"],
        "seed": 11,
        "round_index": 0,
        "phase": "active_round",
        "artifact_dir": tmp_path,
    }

    first = run_inner_loop(_config(), **arguments)
    second = run_inner_loop(_config(), **arguments)

    assert not any(item["reused"] for item in first.trial_results)
    assert all(item["reused"] for item in second.trial_results)
    assert len(fit_calls) == 5  # three trials + first refit + second refit


def test_changed_software_fingerprint_invalidates_completed_trial_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generator = _TrackingGenerator()
    monkeypatch.setitem(TRAINING_CANDIDATE_GENERATORS, "tracking_test", lambda config: generator)
    fit_calls: list[float] = []
    monkeypatch.setattr(
        "agentic_al.inner_loop.build_model",
        lambda config, **kwargs: _FakeModel(config, fit_calls),
    )
    labeled, target = _labeled()
    arguments = {
        "labeled": labeled,
        "target": target,
        "task": "regression",
        "feature_columns": ["x"],
        "seed": 11,
        "round_index": 0,
        "phase": "active_round",
        "artifact_dir": tmp_path,
    }
    run_inner_loop(_config(), **arguments)
    monkeypatch.setattr("agentic_al.inner_loop.software_fingerprint", lambda: "changed")
    second = run_inner_loop(_config(), **arguments)
    assert not any(item["reused"] for item in second.trial_results)
    assert len(fit_calls) == 8  # two sets of three trials plus two selected-plan refits


def test_rule_generator_produces_a_real_robust_loss_training_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fit_calls: list[float] = []
    monkeypatch.setattr(
        "agentic_al.inner_loop.build_model",
        lambda config, **kwargs: _FakeModel(config, fit_calls),
    )
    labeled, target = _labeled()
    result = run_inner_loop(
        _config(generator="rule_based", trial_budget=2),
        labeled=labeled,
        target=target,
        task="regression",
        feature_columns=["x"],
        seed=13,
        round_index=0,
        phase="active_round",
        artifact_dir=tmp_path,
    )
    candidate = result.trial_results[1]["candidate"]
    assert candidate["name"] == "robust_huber_loss"
    assert candidate["paradigm"] == "robust_loss_training"
    assert result.trial_results[1]["resolved_parameters"]["loss"] == "huber"
    assert result.trial_results[1]["model"]["loss"] == "huber"
    plan = json.loads((tmp_path / "step_001" / "trial_001" / "training_plan.json").read_text())
    assert plan["paradigm_parameters"] == {"loss": "huber"}
    assert "loss" not in plan["locked_parameters"]


def test_rule_generator_exposes_all_available_training_paradigms_before_tuning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fit_calls: list[float] = []
    monkeypatch.setattr(
        "agentic_al.inner_loop.build_model",
        lambda config, **kwargs: _FakeModel(config, fit_calls),
    )
    reference = CheckpointReference(
        source="previous_round",
        weights_path=tmp_path / "private" / "weights.npz",
        manifest_path=tmp_path / "private" / "manifest.json",
        origin_round=0,
        origin_phase="active_round",
        selection_metric="mae",
        selection_mode="min",
        selection_value=0.25,
        weights_sha256="a" * 64,
        manifest_sha256="b" * 64,
        lineage_training_fits=1,
    )
    catalog = CheckpointCatalog(
        sources={
            "global_best": replace(reference, source="global_best"),
            "previous_round": reference,
        }
    )
    labeled, target = _labeled()
    result = run_inner_loop(
        _config(generator="rule_based", trial_budget=5),
        labeled=labeled,
        target=target,
        task="regression",
        feature_columns=["x"],
        seed=17,
        round_index=1,
        phase="active_round",
        artifact_dir=tmp_path / "inner",
        checkpoint_catalog=catalog,
    )
    paradigms = [item["candidate"]["paradigm"] for item in result.trial_results]
    assert paradigms == [
        "baseline_equivalent_control",
        "finetune_global_best",
        "finetune_previous_round",
        "robust_loss_training",
        "curriculum_training",
    ]
    public_context = json.loads(
        (tmp_path / "inner" / "step_001" / "agent_context.json").read_text()
    )
    assert public_context["checkpoint_sources"]["global_best"]["origin_round"] == 0
    assert "weights_path" not in json.dumps(public_context)
    assert str(tmp_path) not in json.dumps(public_context)


def test_external_candidate_context_is_aggregate_only_and_repairs_are_audited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[dict[str, object]] = []

    def fake_request(config, system_prompt, public_context):
        del config, system_prompt
        captured.append(public_context)
        return {
            "candidates": [
                {
                    "name": "unsafe proposal",
                    "parameters": {"learning_rate": 99, "epochs": 999},
                    "paradigm": "warm_start",
                    "rationale": "bounded by deterministic validation",
                    "path": "/Users/example/private.csv",
                }
            ]
        }

    fit_calls: list[float] = []
    monkeypatch.setattr("agentic_al.inner_loop.request_policy_json", fake_request)
    monkeypatch.setattr(
        "agentic_al.inner_loop.build_model",
        lambda config, **kwargs: _FakeModel(config, fit_calls),
    )
    labeled, target = _labeled()
    payload = _config(generator="auto", trial_budget=2).model_dump()
    payload["agent"] = {"enabled": True, "provider": "openai_compatible"}
    result = run_inner_loop(
        AppConfig.model_validate(payload),
        labeled=labeled,
        target=target,
        task="regression",
        feature_columns=["x"],
        seed=5,
        round_index=0,
        phase="active_round",
        artifact_dir=tmp_path,
    )

    assert len(captured) == 1
    serialized = json.dumps(captured[0], sort_keys=True)
    for forbidden in ("sample_id", "smiles", "private.csv", "/Users/", '"rows"', '"labels"'):
        assert forbidden not in serialized
    candidate = result.trial_results[1]["candidate"]
    assert candidate["parameters"] == {"learning_rate": 0.005}
    assert "dropped_non_tunable_parameter:epochs" in candidate["repairs"]
    assert "unsupported_paradigm:warm_start->retrain_from_scratch" in candidate["repairs"]
    assert any(item.startswith("clipped_parameter:learning_rate") for item in candidate["repairs"])


def test_skipped_search_still_writes_control_plan(tmp_path: Path, monkeypatch) -> None:
    fit_calls: list[float] = []
    monkeypatch.setattr(
        "agentic_al.inner_loop.build_model",
        lambda config, **kwargs: _FakeModel(config, fit_calls),
    )
    labeled, target = _labeled()
    payload = _config().model_dump()
    payload["inner_loop"]["enabled"] = False
    result = run_inner_loop(
        AppConfig.model_validate(payload),
        labeled=labeled,
        target=target,
        task="regression",
        feature_columns=["x"],
        seed=5,
        round_index=0,
        phase="active_round",
        artifact_dir=tmp_path,
    )

    assert result.summary["skip_reason"] == "inner_loop_disabled"
    assert result.summary["candidate_generation_steps"] == 0
    assert (tmp_path / "step_000" / "trial_000" / "training_plan.json").is_file()
    assert (tmp_path / "step_000" / "trial_000" / "trial_result.json").is_file()


def test_external_candidate_provider_fallback_is_explicit_and_configurable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("AGENTIC_AL_API_BASE", "AGENTIC_AL_API_KEY", "AGENTIC_AL_MODEL"):
        monkeypatch.delenv(name, raising=False)
    fit_calls: list[float] = []
    monkeypatch.setattr(
        "agentic_al.inner_loop.build_model",
        lambda config, **kwargs: _FakeModel(config, fit_calls),
    )
    labeled, target = _labeled()
    payload = _config(generator="auto", trial_budget=2).model_dump()
    payload["agent"] = {
        "enabled": True,
        "provider": "openai_compatible",
        "fallback_to_rule_based": True,
    }
    result = run_inner_loop(
        AppConfig.model_validate(payload),
        labeled=labeled,
        target=target,
        task="regression",
        feature_columns=["x"],
        seed=5,
        round_index=0,
        phase="active_round",
        artifact_dir=tmp_path / "fallback",
    )
    assert result.trial_results[1]["candidate"]["fallback_used"] is True
    assert "requires API" in result.trial_results[1]["candidate"]["fallback_reason"]

    payload["agent"]["fallback_to_rule_based"] = False
    with pytest.raises(RuntimeError, match="requires API"):
        run_inner_loop(
            AppConfig.model_validate(payload),
            labeled=labeled,
            target=target,
            task="regression",
            feature_columns=["x"],
            seed=5,
            round_index=0,
            phase="active_round",
            artifact_dir=tmp_path / "no-fallback",
        )
