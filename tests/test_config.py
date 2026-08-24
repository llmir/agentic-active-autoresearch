from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from agentic_al.config import AppConfig, dump_public_config, load_config


def test_default_is_gpu_first() -> None:
    config = AppConfig()
    assert config.model.name == "torch_mlp"
    assert config.model.device == "auto"
    assert config.model.require_accelerator is True
    assert config.agent.enabled is False
    assert config.inner_loop.enabled is True
    assert config.inner_loop.candidates_per_step == 1
    assert config.inner_loop.require_control is True
    assert config.inner_loop.trial_budget == 6
    assert config.inner_loop.candidate_paradigms == [
        "retrain_from_scratch",
        "finetune_global_best",
        "finetune_previous_round",
        "curriculum_training",
        "robust_loss_training",
    ]


def test_load_config_resolves_dataset_relative_to_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "nested" / "config.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {"kind": "csv", "path": "../data.csv"},
                "model": {
                    "name": "random_forest",
                    "require_accelerator": False,
                    "parameters": {"n_estimators": 8},
                },
            }
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.resolved_dataset_path() == (tmp_path / "data.csv").resolve()


def test_unknown_config_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"surprise": True})


def test_invalid_environment_variable_name_is_rejected() -> None:
    invalid = {"agent": {"api_key_env": "not-valid"}}  # pragma: allowlist secret
    with pytest.raises(ValidationError):
        AppConfig.model_validate(invalid)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("api_key_env", "AWS_SECRET_ACCESS_KEY"),  # pragma: allowlist secret
        ("base_url_env", "PRIVATE_PROVIDER_URL"),
        ("model_env", "UNRELATED_MODEL_NAME"),
    ],
)
def test_agent_config_cannot_select_unrelated_process_credentials(field: str, value: str) -> None:
    with pytest.raises(ValidationError, match="fixed"):
        AppConfig.model_validate({"agent": {field: value}})


def test_task_goal_mismatch_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate(
            {
                "dataset": {"task": "classification"},
                "acquisition": {"goal": "target_range", "target_range": [0, 1]},
            }
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"dataset": {"kind": "csv"}},
        {"dataset": {"kind": "jsonl"}},
        {"dataset": {"kind": "parquet"}},
        {"dataset": {"kind": "invalid kind"}},
        {"dataset": {"synthetic_features": 2, "synthetic_informative": 3}},
        {"acquisition": {"weights": {}}},
        {"acquisition": {"weights": {"random": 0}}},
        {"acquisition": {"weights": {"random": -1}}},
        {"acquisition": {"weights": {"random": float("nan")}}},
        {"acquisition": {"weights": {"random": float("inf")}}},
        {"acquisition": {"goal": "target_range"}},
        {"acquisition": {"goal": "target_range", "target_range": [2, 1]}},
        {"dataset": {"task": "classification"}, "acquisition": {"goal": "target_class"}},
        {"agent": {"temperature": float("nan")}},
    ],
)
def test_invalid_semantic_configs_are_rejected(payload: dict) -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate(payload)


def test_load_config_requires_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be a mapping"):
        load_config(path)


def test_public_config_contains_names_not_secret_values() -> None:
    config = AppConfig.model_validate(
        {
            "dataset": {"path": "/Volumes/Private Lab/pool.csv"},
            "run": {"output_dir": "/Users/example/private-run"},
        }
    )
    payload = dump_public_config(config)
    assert payload["agent"]["api_key_env"] == "AGENTIC_AL_API_KEY"  # pragma: allowlist secret
    assert "api_key" not in payload["agent"]
    assert payload["dataset"]["path"] == "${DATASET_PATH}"
    assert payload["run"]["output_dir"] == "${RUN_DIR}"


@pytest.mark.parametrize(
    "payload",
    [
        {"dataset": {"parameters": {"api_key": "do-not-store"}}},  # pragma: allowlist secret
        {  # pragma: allowlist secret
            "dataset": {"parameters": {"nested": {"access_token": "do-not-store"}}}
        },
        {"dataset": {"parameters": {"database_url": "postgresql://private"}}},
        {"dataset": {"parameters": {"session_cookie": "do-not-store"}}},
        {  # pragma: allowlist secret
            "model": {"parameters": {"password_file": "/private/location"}}
        },
    ],
)
def test_plugin_parameters_reject_secret_bearing_keys(payload: dict) -> None:
    with pytest.raises(ValidationError, match="looks secret-bearing"):
        AppConfig.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"dataset": {"parameters": {"threshold": float("inf")}}},
        {"model": {"parameters": {"learning_rate": float("nan")}}},
    ],
)
def test_plugin_and_model_parameters_reject_nonfinite_values(payload: dict) -> None:
    with pytest.raises(ValidationError, match="must be finite"):
        AppConfig.model_validate(payload)


def test_plugin_parameters_allow_environment_variable_references() -> None:
    parameters = {"api_key_env": "LAB_API_KEY"}  # pragma: allowlist secret
    config = AppConfig.model_validate({"dataset": {"parameters": parameters}})
    assert config.dataset.parameters["api_key_env"] == "LAB_API_KEY"  # pragma: allowlist secret


@pytest.mark.parametrize(
    "name", ["epochs", "warmup_epochs", "batch_size", "ensemble_size", "hidden_dims"]
)
def test_inner_loop_rejects_compute_budget_and_architecture_tuning(name: str) -> None:
    with pytest.raises(ValidationError, match="cannot be agent-tuned"):
        AppConfig.model_validate({"inner_loop": {"tunable_parameters": {name: [1, 10]}}})


@pytest.mark.parametrize(
    "payload",
    [
        {"candidate_generator": "not valid!"},
        {"selection_metric": "not valid!"},
        {"trial_budget": 1, "candidates_per_step": 2},
        {"tunable_parameters": {"learning_rate": [1.0, 1.0]}},
        {"tunable_parameters": {"learning_rate": [float("nan"), 1.0]}},
        {"candidate_paradigms": []},
        {"candidate_paradigms": ["robust_loss_training"]},
        {"candidate_paradigms": ["retrain_from_scratch", "retrain_from_scratch"]},
    ],
)
def test_invalid_inner_loop_contract_is_rejected(payload: dict) -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"inner_loop": payload})
