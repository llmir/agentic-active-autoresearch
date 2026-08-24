import json
from pathlib import Path

import pandas as pd
import pytest

from agentic_al import engine
from agentic_al.config import AppConfig
from agentic_al.engine import run_experiment
from agentic_al.oracle import CallableOracle
from agentic_al.types import StrategyProposal


def _config(output_dir: Path, rounds: int = 2) -> AppConfig:
    return AppConfig.model_validate(
        {
            "project_name": "unit-e2e",
            "dataset": {"kind": "synthetic", "synthetic_samples": 100, "synthetic_features": 5},
            "run": {
                "seed": 7,
                "rounds": rounds,
                "initial_size": 16,
                "batch_size": 8,
                "output_dir": output_dir,
            },
            "model": {
                "name": "random_forest",
                "device": "cpu",
                "require_accelerator": False,
                "parameters": {"n_estimators": 16, "min_samples_leaf": 1, "n_jobs": 1},
            },
            "acquisition": {
                "weights": {"uncertainty": 0.5, "diversity": 0.3, "random": 0.2},
                "goal": "minimize",
            },
        }
    )


def _torch_checkpoint_config(output_dir: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "project_name": "checkpoint-lineage-e2e",
            "dataset": {
                "kind": "synthetic",
                "synthetic_samples": 60,
                "synthetic_features": 4,
                "synthetic_informative": 3,
            },
            "run": {
                "seed": 9,
                "rounds": 2,
                "initial_size": 16,
                "batch_size": 4,
                "output_dir": output_dir,
            },
            "model": {
                "name": "torch_mlp",
                "device": "cpu",
                "require_accelerator": False,
                "parameters": {
                    "hidden_dims": [8],
                    "dropout": 0.0,
                    "epochs": 1,
                    "batch_size": 8,
                    "learning_rate": 0.01,
                    "weight_decay": 0.0,
                    "ensemble_size": 1,
                    "mc_dropout_passes": 1,
                },
            },
            "inner_loop": {
                "trial_budget": 5,
                "candidates_per_step": 1,
                "min_labeled_size": 12,
            },
        }
    )


def test_end_to_end_artifact_contract(tmp_path: Path) -> None:
    output = tmp_path / "run"
    result = run_experiment(_config(output))
    assert result.completed_rounds == 2
    assert result.report_path.is_file()
    assert (output / "round_000" / "commit.json").is_file()
    assert (output / "round_000" / "inner_loop" / "inner_loop_summary.json").is_file()
    assert (output / "round_000" / "inner_loop" / "summary.csv").is_file()
    assert (output / "final_inner_loop" / "best_trial.json").is_file()
    assert len(pd.read_csv(output / "summary.csv")) == 2
    strategy = json.loads((output / "round_000" / "strategy.json").read_text())
    assert strategy["source"] == "rule_based"
    assert strategy["policy_metric_source"] == "labeled_only_inner_validation"
    assert abs(sum(strategy["weights"].values()) - 1.0) < 1e-12
    metrics = json.loads((output / "final_metrics.json").read_text())
    assert metrics["device"] == "cpu"
    assert metrics["accelerator_used"] is False
    assert metrics["best_training_candidate"] == "baseline_equivalent_control"
    summary = pd.read_csv(output / "summary.csv")
    assert "best_training_candidate" in summary
    assert summary["inner_training_trials"].tolist() == [1, 1]


def test_torch_rounds_execute_checkpoint_curriculum_and_robust_paradigms(
    tmp_path: Path,
) -> None:
    output = tmp_path / "torch-lineage"
    result = run_experiment(_torch_checkpoint_config(output))
    assert result.completed_rounds == 2
    assert (output / "round_000" / "inner_loop" / "selected_checkpoint.npz").is_file()
    assert (output / "round_001" / "inner_loop" / "selected_checkpoint.json").is_file()
    trials = []
    for path in sorted(
        (output / "round_001" / "inner_loop").glob("step_*/trial_*/trial_result.json")
    ):
        trials.append(json.loads(path.read_text()))
    paradigms = [item["candidate"]["paradigm"] for item in trials]
    assert paradigms == [
        "baseline_equivalent_control",
        "finetune_global_best",
        "finetune_previous_round",
        "robust_loss_training",
        "curriculum_training",
    ]
    steps = [item["model"]["optimizer_steps"] for item in trials]
    assert len(set(steps)) == 1
    continuation = [
        item for item in trials if item["candidate"]["paradigm"].startswith("finetune_")
    ]
    assert all(item["model"]["lineage_training_fits"] == 2 for item in continuation)
    assert all(item["checkpoint_initialization"] is not None for item in continuation)


def test_torch_resume_requires_checkpoint_evidence(tmp_path: Path) -> None:
    output = tmp_path / "torch-resume"
    config = _torch_checkpoint_config(output)
    run_experiment(config)
    (output / "round_001" / "inner_loop" / "selected_checkpoint.npz").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        run_experiment(config, resume=True)


def test_acquisition_policy_never_receives_outer_validation_metrics(tmp_path: Path) -> None:
    contexts = []

    class CapturePolicy:
        def propose(self, context):
            contexts.append(context)
            return StrategyProposal(
                weights={"random": 1.0},
                rationale="capture labeled-only context",
                source="capture_test",
            )

    output = tmp_path / "policy-boundary"
    run_experiment(_config(output, rounds=1), policy=CapturePolicy())
    assert len(contexts) == 1
    assert contexts[0].metrics == {}  # control-only RF search has no inner holdout metrics
    round_metrics = json.loads((output / "round_000" / "metrics.json").read_text())
    assert "mae" in round_metrics
    strategy = json.loads((output / "round_000" / "strategy.json").read_text())
    assert strategy["policy_context_metrics"] == {}


def test_nonempty_output_requires_resume(tmp_path: Path) -> None:
    output = tmp_path / "run"
    run_experiment(_config(output))
    with pytest.raises(FileExistsError):
        run_experiment(_config(output))


def test_resume_is_idempotent_after_completion(tmp_path: Path) -> None:
    output = tmp_path / "run"
    config = _config(output)
    first = run_experiment(config)
    second = run_experiment(config, resume=True)
    assert first.completed_rounds == second.completed_rounds == 2
    assert len(pd.read_csv(output / "summary.csv")) == 2


def test_resume_rejects_changed_config(tmp_path: Path) -> None:
    output = tmp_path / "run"
    run_experiment(_config(output))
    with pytest.raises(ValueError, match="hash changed"):
        run_experiment(_config(output, rounds=3), resume=True)


def test_resume_rejects_changed_software_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    config = _config(output)
    run_experiment(config)
    monkeypatch.setattr(engine, "software_fingerprint", lambda: "changed-source-hash")
    with pytest.raises(ValueError, match="hash changed"):
        run_experiment(config, resume=True)


def test_provider_runtime_fingerprint_is_secret_free_and_change_sensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _config(Path("outputs/provider")).model_dump()
    payload["agent"] = {"enabled": True, "provider": "openai_compatible"}
    config = AppConfig.model_validate(payload)
    monkeypatch.setenv("AGENTIC_AL_API_BASE", "https://provider-one.example/v1")
    monkeypatch.setenv("AGENTIC_AL_API_KEY", "first-private-value")
    monkeypatch.setenv("AGENTIC_AL_MODEL", "model-one")
    first = engine._provider_runtime_fingerprint(config)
    monkeypatch.setenv("AGENTIC_AL_API_KEY", "rotated-private-value")
    assert engine._provider_runtime_fingerprint(config) == first
    monkeypatch.setenv("AGENTIC_AL_MODEL", "model-two")
    assert engine._provider_runtime_fingerprint(config) != first
    assert "private" not in first


def test_resume_recovers_missing_commit_from_durable_state(tmp_path: Path) -> None:
    output = tmp_path / "run"
    config = _config(output)
    run_experiment(config)
    (output / "round_001" / "commit.json").unlink()
    run_experiment(config, resume=True)
    recovered = json.loads((output / "round_001" / "commit.json").read_text())
    assert recovered["recovered_from_state"] is True


def test_resume_rejects_missing_round_evidence(tmp_path: Path) -> None:
    output = tmp_path / "run"
    config = _config(output)
    run_experiment(config)
    (output / "round_001" / "selection.csv").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        run_experiment(config, resume=True)


def test_resume_rejects_missing_inner_loop_evidence(tmp_path: Path) -> None:
    output = tmp_path / "run"
    config = _config(output)
    run_experiment(config)
    (output / "round_001" / "inner_loop" / "summary.csv").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        run_experiment(config, resume=True)


def test_jsonl_dataset_runs_end_to_end(tmp_path: Path) -> None:
    dataset_path = tmp_path / "pool.jsonl"
    pd.DataFrame(
        {
            "sample_id": [f"row-{index:03d}" for index in range(100)],
            "feature": [float(index) for index in range(100)],
            "target": [float(index * 2 + 1) for index in range(100)],
        }
    ).to_json(dataset_path, orient="records", lines=True)
    payload = _config(tmp_path / "jsonl-run", rounds=1).model_dump()
    payload["dataset"] = {
        "kind": "jsonl",
        "path": dataset_path,
        "task": "regression",
        "target_column": "target",
        "feature_columns": ["feature"],
    }
    result = run_experiment(AppConfig.model_validate(payload))
    assert result.completed_rounds == 1
    assert result.final_metrics["r2"] > 0.8


def test_unlabeled_regression_pool_runs_with_external_oracle_and_validation(
    tmp_path: Path,
) -> None:
    pool_path = tmp_path / "pool.csv"
    pool = pd.DataFrame(
        {
            "sample_id": [f"pool-{index:03d}" for index in range(80)],
            "x": [float(index) for index in range(80)],
        }
    )
    pool.to_csv(pool_path, index=False)
    validation = pd.DataFrame(
        {
            "sample_id": [f"validation-{index:03d}" for index in range(20)],
            "x": [float(index) + 0.5 for index in range(20)],
            "target": [2.0 * (float(index) + 0.5) + 1.0 for index in range(20)],
        }
    )
    config = AppConfig.model_validate(
        {
            "project_name": "external-oracle-regression",
            "dataset": {
                "kind": "csv",
                "path": pool_path,
                "task": "regression",
                "target_column": "target",
                "feature_columns": ["x"],
            },
            "run": {
                "seed": 5,
                "rounds": 1,
                "initial_size": 20,
                "batch_size": 8,
                "output_dir": tmp_path / "external-run",
            },
            "model": {
                "name": "random_forest",
                "device": "cpu",
                "require_accelerator": False,
                "parameters": {"n_estimators": 32, "min_samples_leaf": 1, "n_jobs": 1},
            },
            "inner_loop": {"enabled": False, "trial_budget": 1},
            "acquisition": {"weights": {"uncertainty": 0.7, "diversity": 0.3}},
        }
    )

    def observe(rows, id_column, target_column):
        del target_column
        return {str(row[id_column]): 2.0 * float(row["x"]) + 1.0 for _, row in rows.iterrows()}

    result = run_experiment(
        config,
        oracle=CallableOracle(observe),
        validation_data=validation,
    )
    assert result.completed_rounds == 1
    assert result.final_metrics["r2"] > 0.8
    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    assert manifest["observation_mode"] == "external_oracle_with_validation_data"
    assert "target" not in pool.columns


def test_external_validation_requires_oracle_and_regression(tmp_path: Path) -> None:
    validation = pd.DataFrame({"sample_id": ["v-1", "v-2"], "x": [1.0, 2.0], "target": [0.0, 1.0]})
    with pytest.raises(ValueError, match="requires an explicit"):
        run_experiment(_config(tmp_path / "missing-oracle"), validation_data=validation)

    payload = _config(tmp_path / "classification-external").model_dump()
    payload["dataset"]["task"] = "classification"
    payload["acquisition"] = {"goal": "target_class", "target_class": 1}
    with pytest.raises(ValueError, match="regression only"):
        run_experiment(
            AppConfig.model_validate(payload),
            oracle=CallableOracle(lambda rows, id_column, target_column: {}),
            validation_data=validation,
        )
