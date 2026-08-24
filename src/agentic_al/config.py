"""Typed configuration with conservative defaults and explicit extension points."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationInfo,
    field_validator,
    model_validator,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


_SENSITIVE_PARAMETER_NAMES = {
    "access_token",
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "client_secret",
    "connection_string",
    "cookie",
    "credential",
    "credentials",
    "database_url",
    "dsn",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "session_cookie",
    "token",
}


def _is_environment_name(value: str) -> bool:
    return bool(value) and value.replace("_", "A").isalnum() and value.upper() == value


def _sensitive_parameter_path(value: Any, path: tuple[str, ...] = ()) -> str | None:
    if isinstance(value, dict):
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized = key.strip().lower().replace("-", "_")
            current = (*path, key)
            if normalized.endswith("_env") and isinstance(item, str) and _is_environment_name(item):
                continue
            if any(
                normalized == name
                or normalized.startswith(f"{name}_")
                or normalized.endswith(f"_{name}")
                for name in _SENSITIVE_PARAMETER_NAMES
            ):
                return ".".join(current)
            nested = _sensitive_parameter_path(item, current)
            if nested:
                return nested
    elif isinstance(value, list):
        for index, item in enumerate(value):
            nested = _sensitive_parameter_path(item, (*path, str(index)))
            if nested:
                return nested
    return None


def _validate_public_parameters(value: dict[str, Any]) -> dict[str, Any]:
    sensitive_path = _sensitive_parameter_path(value)
    if sensitive_path:
        raise ValueError(
            f"parameter '{sensitive_path}' looks secret-bearing; store only an environment "
            "variable name in an *_env field"
        )
    nonfinite_path = _nonfinite_parameter_path(value)
    if nonfinite_path:
        raise ValueError(f"parameter '{nonfinite_path}' must be finite")
    return value


def _nonfinite_parameter_path(value: Any, path: tuple[str, ...] = ()) -> str | None:
    if isinstance(value, dict):
        for raw_key, item in value.items():
            nested = _nonfinite_parameter_path(item, (*path, str(raw_key)))
            if nested:
                return nested
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            nested = _nonfinite_parameter_path(item, (*path, str(index)))
            if nested:
                return nested
    elif isinstance(value, float) and not math.isfinite(value):
        return ".".join(path) or "<root>"
    return None


class DatasetConfig(StrictModel):
    kind: str = "synthetic"
    path: Path | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    task: Literal["regression", "classification"] = "regression"
    id_column: str = "sample_id"
    generate_id_if_missing: bool = True
    target_column: str = "target"
    feature_columns: list[str] | None = None
    group_column: str | None = None
    cost_column: str | None = None
    risk_column: str | None = None
    synthetic_samples: int = Field(default=240, ge=40, le=1_000_000)
    synthetic_features: int = Field(default=8, ge=2, le=1_000)
    synthetic_informative: int = Field(default=5, ge=1, le=1_000)
    synthetic_classes: int = Field(default=2, ge=2, le=100)
    noise: float = Field(default=8.0, ge=0.0)

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, value: str) -> str:
        normalized = value.strip().lower().replace("-", "_")
        if not normalized or not normalized.replace("_", "a").isalnum():
            raise ValueError("dataset.kind must contain only letters, digits, and underscores")
        return normalized

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_public_parameters(value)

    @model_validator(mode="after")
    def validate_source(self) -> DatasetConfig:
        if self.kind in {"csv", "jsonl", "parquet"} and self.path is None:
            raise ValueError(f"dataset.path is required when dataset.kind='{self.kind}'")
        if self.synthetic_informative > self.synthetic_features:
            raise ValueError("synthetic_informative cannot exceed synthetic_features")
        return self


class SplitConfig(StrictModel):
    validation_fraction: float = Field(default=0.2, gt=0.0, lt=0.5)


class RunConfig(StrictModel):
    seed: int = 42
    rounds: int = Field(default=4, ge=1, le=10_000)
    initial_size: int = Field(default=24, ge=4)
    batch_size: int = Field(default=12, ge=1)
    output_dir: Path = Path("outputs/quickstart")
    resume: bool = False
    store_observations: bool = True


class ModelConfig(StrictModel):
    name: str = "torch_mlp"
    featurizer: Literal["tabular", "morgan"] = "tabular"
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    require_accelerator: bool = True
    smiles_column: str = "smiles"
    fingerprint_radius: int = Field(default=2, ge=1, le=5)
    fingerprint_bits: int = Field(default=1024, ge=128, le=8192)
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {
            "hidden_dims": [256, 128],
            "dropout": 0.10,
            "epochs": 80,
            "batch_size": 64,
            "learning_rate": 0.001,
            "weight_decay": 0.00001,
            "ensemble_size": 3,
            "mc_dropout_passes": 20,
        }
    )

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_public_parameters(value)


_COMPUTE_BUDGET_PARAMETERS = {
    "batch_size",
    "depth",
    "ensemble_size",
    "epochs",
    "ffn_hidden_dim",
    "ffn_num_layers",
    "hidden_dims",
    "max_depth",
    "mc_dropout_passes",
    "message_hidden_dim",
    "n_estimators",
    "num_layers",
    "num_workers",
    "warmup_epochs",
    "width",
}


class InnerLoopConfig(StrictModel):
    enabled: bool = True
    candidate_generator: str = "auto"
    trial_budget: int = Field(default=6, ge=1, le=50)
    candidates_per_step: int = Field(default=1, ge=1, le=10)
    validation_fraction: float = Field(default=0.25, gt=0.0, lt=0.5)
    min_labeled_size: int = Field(default=12, ge=4)
    selection_metric: str = "auto"
    selection_mode: Literal["auto", "min", "max"] = "auto"
    require_control: bool = True
    reuse_completed_trials: bool = True
    candidate_paradigms: list[
        Literal[
            "retrain_from_scratch",
            "finetune_global_best",
            "finetune_previous_round",
            "curriculum_training",
            "robust_loss_training",
        ]
    ] = Field(
        default_factory=lambda: [
            "retrain_from_scratch",
            "finetune_global_best",
            "finetune_previous_round",
            "curriculum_training",
            "robust_loss_training",
        ]
    )
    tunable_parameters: dict[str, tuple[float, float]] = Field(
        default_factory=lambda: {
            "learning_rate": (0.00001, 0.005),
            "dropout": (0.0, 0.5),
            "weight_decay": (0.0, 0.01),
        }
    )

    @field_validator("candidate_generator")
    @classmethod
    def validate_candidate_generator(cls, value: str) -> str:
        normalized = value.strip().lower().replace("-", "_")
        if not normalized or not normalized.replace("_", "a").isalnum():
            raise ValueError(
                "inner_loop.candidate_generator must contain only letters, digits, and underscores"
            )
        return normalized

    @field_validator("selection_metric")
    @classmethod
    def validate_selection_metric(cls, value: str) -> str:
        normalized = value.strip().lower().replace("-", "_")
        if not normalized or not normalized.replace("_", "a").isalnum():
            raise ValueError(
                "inner_loop.selection_metric must contain only letters, digits, and underscores"
            )
        return normalized

    @field_validator("tunable_parameters")
    @classmethod
    def validate_tunable_parameters(
        cls, value: dict[str, tuple[float, float]]
    ) -> dict[str, tuple[float, float]]:
        normalized: dict[str, tuple[float, float]] = {}
        for raw_name, raw_bounds in value.items():
            name = str(raw_name).strip().lower().replace("-", "_")
            if not name or not name.replace("_", "a").isalnum():
                raise ValueError("inner-loop tunable parameter names must be identifiers")
            if name in _COMPUTE_BUDGET_PARAMETERS:
                raise ValueError(
                    f"inner-loop parameter '{name}' controls compute budget or architecture and "
                    "cannot be agent-tuned"
                )
            low, high = float(raw_bounds[0]), float(raw_bounds[1])
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ValueError(f"inner-loop bounds for '{name}' must be finite and increasing")
            normalized[name] = (low, high)
        return normalized

    @model_validator(mode="after")
    def validate_candidate_batch_size(self) -> InnerLoopConfig:
        if self.candidates_per_step > self.trial_budget:
            raise ValueError("inner_loop.candidates_per_step cannot exceed trial_budget")
        if not self.candidate_paradigms:
            raise ValueError("inner_loop.candidate_paradigms cannot be empty")
        if len(set(self.candidate_paradigms)) != len(self.candidate_paradigms):
            raise ValueError("inner_loop.candidate_paradigms cannot contain duplicates")
        if "retrain_from_scratch" not in self.candidate_paradigms:
            raise ValueError(
                "inner_loop.candidate_paradigms must include retrain_from_scratch as a fallback"
            )
        return self


class AcquisitionConfig(StrictModel):
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "uncertainty": 0.50,
            "diversity": 0.30,
            "random": 0.15,
            "target": 0.05,
        }
    )
    goal: Literal["minimize", "maximize", "target_range", "target_class"] = "minimize"
    target_range: tuple[float, float] | None = None
    target_class: str | int | float | None = None
    cost_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    diversity_chunk_size: int = Field(default=2048, ge=32, le=100_000)

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("acquisition.weights cannot be empty")
        if any(
            not isinstance(weight, int | float) or not math.isfinite(float(weight)) or weight < 0
            for weight in value.values()
        ):
            raise ValueError("acquisition weights must be finite non-negative numbers")
        if sum(float(weight) for weight in value.values()) <= 0:
            raise ValueError("at least one acquisition weight must be positive")
        return {str(name): float(weight) for name, weight in value.items()}

    @model_validator(mode="after")
    def validate_goal(self) -> AcquisitionConfig:
        if self.goal == "target_range" and (
            self.target_range is None or self.target_range[0] >= self.target_range[1]
        ):
            raise ValueError("goal='target_range' requires an increasing target_range")
        if self.goal == "target_class" and self.target_class is None:
            raise ValueError("goal='target_class' requires target_class")
        return self


class GuardrailConfig(StrictModel):
    enabled: bool = True
    max_agent_delta: float = Field(default=0.15, ge=0.0, le=1.0)
    max_component_weight: float = Field(default=0.80, gt=0.0, le=1.0)
    max_cost_penalty: float = Field(default=0.20, ge=0.0, le=1.0)
    max_risk_penalty: float = Field(default=0.20, ge=0.0, le=1.0)


class AgentConfig(StrictModel):
    enabled: bool = False
    provider: Literal["rule_based", "openai_compatible"] = "rule_based"
    base_url_env: str = "AGENTIC_AL_API_BASE"
    api_key_env: str = "AGENTIC_AL_API_KEY"
    model_env: str = "AGENTIC_AL_MODEL"
    model: str = ""
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=600.0)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_response_bytes: int = Field(default=1_000_000, ge=1_024, le=10_000_000)
    allow_local_http: bool = True
    fallback_to_rule_based: bool = True

    @field_validator("base_url_env", "api_key_env", "model_env")
    @classmethod
    def validate_env_name(cls, value: str, info: ValidationInfo) -> str:
        if not _is_environment_name(value):
            raise ValueError(
                "environment variable names must use uppercase letters, digits, and underscores"
            )
        expected = {
            "base_url_env": "AGENTIC_AL_API_BASE",
            "api_key_env": "AGENTIC_AL_API_KEY",  # pragma: allowlist secret
            "model_env": "AGENTIC_AL_MODEL",
        }[info.field_name]
        if value != expected:
            raise ValueError(
                f"{info.field_name} is fixed to {expected} so an untrusted config cannot select "
                "an unrelated process credential"
            )
        return value


class AppConfig(StrictModel):
    project_name: str = "agentic-active-autoresearch-demo"
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    run: RunConfig = Field(default_factory=RunConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    inner_loop: InnerLoopConfig = Field(default_factory=InnerLoopConfig)
    acquisition: AcquisitionConfig = Field(default_factory=AcquisitionConfig)
    guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    _config_dir: Path = PrivateAttr(default=Path.cwd())

    @model_validator(mode="after")
    def validate_task_goal(self) -> AppConfig:
        if self.dataset.task == "classification" and self.acquisition.goal == "target_range":
            raise ValueError("classification does not support goal='target_range'")
        if self.dataset.task == "regression" and self.acquisition.goal == "target_class":
            raise ValueError("regression does not support goal='target_class'")
        return self

    def resolved_dataset_path(self) -> Path | None:
        if self.dataset.path is None:
            return None
        path = self.dataset.path.expanduser()
        return path if path.is_absolute() else (self._config_dir / path).resolve()


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")
    config = AppConfig.model_validate(payload)
    config._config_dir = config_path.parent
    return config


def dump_public_config(config: AppConfig) -> dict[str, Any]:
    """Serialize configuration without reading or materializing secret values."""

    payload = config.model_dump(mode="json")
    if payload["dataset"].get("path") is not None:
        payload["dataset"]["path"] = "${DATASET_PATH}"
    payload["run"]["output_dir"] = "${RUN_DIR}"
    return payload
