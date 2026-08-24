"""Provider-independent surrogate model adapters."""

from __future__ import annotations

import os
import re
import shutil

# Security: fixed Chemprop executable, argument list, no shell, and allowlisted environment.
import subprocess  # nosec B404
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any, ClassVar, Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .checkpoints import load_checkpoint
from .config import ModelConfig
from .featurizers import Featurizer, build_featurizer, featurizer_from_checkpoint
from .registry import MODEL_FACTORIES, register_model
from .types import CheckpointReference, Prediction


class SurrogateModel(Protocol):
    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> None: ...

    def predict(self, frame: pd.DataFrame) -> Prediction: ...

    def featurize(self, frame: pd.DataFrame) -> np.ndarray: ...

    def metadata(self) -> dict[str, Any]: ...


class RandomForestSurrogate:
    """CPU baseline with per-tree uncertainty; not the formal GPU route."""

    def __init__(
        self,
        *,
        task: str,
        featurizer: Featurizer,
        seed: int,
        parameters: dict[str, Any],
    ) -> None:
        self.task = task
        self.featurizer = featurizer
        parameters = dict(parameters)
        parameters.setdefault("n_estimators", 128)
        parameters.setdefault("min_samples_leaf", 2)
        parameters.setdefault("n_jobs", 1)
        parameters["random_state"] = seed
        if task == "classification":
            self.estimator: RandomForestClassifier | RandomForestRegressor = RandomForestClassifier(
                **parameters
            )
        else:
            self.estimator = RandomForestRegressor(**parameters)

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> None:
        features = self.featurizer.fit_transform(frame)
        self.estimator.fit(features, target)

    def predict(self, frame: pd.DataFrame) -> Prediction:
        features = self.featurize(frame)
        if self.task == "classification":
            classifier = self.estimator
            if not isinstance(classifier, RandomForestClassifier):
                raise TypeError("classification surrogate is misconfigured")
            member_probabilities = np.asarray(
                [tree.predict_proba(features) for tree in classifier.estimators_], dtype=float
            )
            probabilities = _normalize_probabilities(member_probabilities.mean(axis=0))
            entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1)
            if probabilities.shape[1] > 1:
                entropy /= np.log(probabilities.shape[1])
            labels = classifier.classes_[np.argmax(probabilities, axis=1)]
            mean = (
                probabilities[:, -1] if probabilities.shape[1] == 2 else probabilities.max(axis=1)
            )
            return Prediction(
                mean=np.asarray(mean, dtype=float),
                uncertainty=np.asarray(entropy, dtype=float),
                labels=np.asarray(labels),
                probabilities=np.asarray(probabilities, dtype=float),
                classes=np.asarray(classifier.classes_),
            )
        regressor = self.estimator
        if not isinstance(regressor, RandomForestRegressor):
            raise TypeError("regression surrogate is misconfigured")
        members = np.asarray([tree.predict(features) for tree in regressor.estimators_])
        return Prediction(mean=members.mean(axis=0), uncertainty=members.std(axis=0, ddof=0))

    def featurize(self, frame: pd.DataFrame) -> np.ndarray:
        return self.featurizer.transform(frame)

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "scikit-learn",
            "model": type(self.estimator).__name__,
            "device": "cpu",
            "accelerator_used": False,
            "purpose": "baseline_or_smoke_test",
        }


class TorchMLPSurrogate:
    """Compact GPU-first MLP ensemble with real MC-dropout uncertainty."""

    _ALLOWED_PARAMETERS: ClassVar[set[str]] = {
        "hidden_dims",
        "dropout",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "ensemble_size",
        "mc_dropout_passes",
        "loss",
        "training_schedule",
        "curriculum_start_fraction",
    }

    def __init__(
        self,
        *,
        task: str,
        featurizer: Featurizer,
        seed: int,
        parameters: dict[str, Any],
        requested_device: str,
        require_accelerator: bool,
        feature_columns: list[str],
    ) -> None:
        unknown = sorted(set(parameters).difference(self._ALLOWED_PARAMETERS))
        if unknown:
            raise ValueError(f"unknown torch_mlp parameters: {unknown}")
        self.task = task
        self.featurizer = featurizer
        self.seed = seed
        self.parameters = dict(parameters)
        self.requested_device = requested_device
        self.require_accelerator = require_accelerator
        self.feature_columns = list(feature_columns)
        self.models: list[Any] = []
        self.classes: np.ndarray | None = None
        self.target_mean = 0.0
        self.target_scale = 1.0
        self.device = "unresolved"
        self.initialization: CheckpointReference | None = None
        self.initialization_public: dict[str, Any] | None = None
        self.lineage_training_fits = 1
        self.optimizer_steps = 0
        self.training_examples = 0
        self.training_schedule = str(self.parameters.get("training_schedule", "standard"))
        self.loss_name = (
            str(
                self.parameters.get(
                    "loss", "cross_entropy" if self.task == "classification" else "mse"
                )
            )
            .strip()
            .lower()
        )

    def set_checkpoint_initialization(self, reference: CheckpointReference) -> None:
        if self.models:
            raise RuntimeError("checkpoint initialization must be configured before fitting")
        self.initialization = reference
        self.initialization_public = reference.to_public_dict()

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> None:
        torch = _torch()
        self.device = _resolve_torch_device(
            self.requested_device, require_accelerator=self.require_accelerator
        )
        checkpoint_manifest: dict[str, Any] | None = None
        checkpoint_arrays: dict[str, np.ndarray] = {}
        if self.initialization is not None:
            checkpoint_manifest, checkpoint_arrays = load_checkpoint(self.initialization)
            checkpoint_model = checkpoint_manifest["model"]
            if (
                checkpoint_model.get("backend") != "torch_mlp"
                or checkpoint_model.get("task") != self.task
            ):
                raise ValueError("checkpoint backend or task does not match torch_mlp")
            self.featurizer = featurizer_from_checkpoint(
                checkpoint_model["featurizer"], expected_feature_columns=self.feature_columns
            )
            self.lineage_training_fits = int(checkpoint_model.get("lineage_training_fits", 1)) + 1
        features = self.featurizer.fit_transform(frame).astype(np.float32, copy=False)
        hidden_dims = [int(value) for value in self.parameters.get("hidden_dims", [256, 128])]
        dropout = float(self.parameters.get("dropout", 0.1))
        epochs = int(self.parameters.get("epochs", 80))
        batch_size = int(self.parameters.get("batch_size", 64))
        learning_rate = float(self.parameters.get("learning_rate", 1e-3))
        weight_decay = float(self.parameters.get("weight_decay", 1e-5))
        ensemble_size = int(self.parameters.get("ensemble_size", 3))
        self.training_schedule = (
            str(self.parameters.get("training_schedule", "standard")).strip().lower()
        )
        curriculum_start = float(self.parameters.get("curriculum_start_fraction", 0.35))
        if not hidden_dims or any(value < 1 for value in hidden_dims):
            raise ValueError("hidden_dims must contain positive integers")
        if epochs < 1 or batch_size < 1 or ensemble_size < 1:
            raise ValueError("epochs, batch_size, and ensemble_size must be positive")
        if self.training_schedule not in {"standard", "curriculum"}:
            raise ValueError("training_schedule must be 'standard' or 'curriculum'")
        if not 0.0 < curriculum_start <= 1.0:
            raise ValueError("curriculum_start_fraction must be in (0, 1]")
        if self.task == "classification":
            current_classes, encoded_target = np.unique(target, return_inverse=True)
            if checkpoint_manifest is not None:
                stored_classes = np.asarray(checkpoint_manifest["model"].get("classes", []))
                if not np.array_equal(current_classes, stored_classes):
                    raise ValueError("checkpoint classes do not match current labeled classes")
                self.classes = stored_classes
            else:
                self.classes = current_classes
            output_dim = len(self.classes)
            if output_dim < 2:
                raise ValueError("initial labeled set must contain at least two classes")
            target_tensor = torch.as_tensor(encoded_target, dtype=torch.long)
        else:
            numeric_target = np.asarray(target, dtype=np.float32)
            if checkpoint_manifest is not None:
                normalization = checkpoint_manifest["model"].get("target_normalization", {})
                self.target_mean = float(normalization["mean"])
                self.target_scale = float(normalization["scale"])
                if (
                    not np.isfinite(self.target_mean)
                    or not np.isfinite(self.target_scale)
                    or self.target_scale <= 0.0
                ):
                    raise ValueError("checkpoint target normalization is invalid")
            else:
                self.target_mean = float(numeric_target.mean())
                self.target_scale = max(float(numeric_target.std()), 1e-8)
            normalized_target = (numeric_target - self.target_mean) / self.target_scale
            output_dim = 1
            target_tensor = torch.as_tensor(normalized_target[:, None], dtype=torch.float32)
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        architecture = {
            "input_dim": int(features.shape[1]),
            "hidden_dims": hidden_dims,
            "output_dim": int(output_dim),
            "ensemble_size": ensemble_size,
        }
        if (
            checkpoint_manifest is not None
            and checkpoint_manifest["model"].get("architecture") != architecture
        ):
            raise ValueError("checkpoint architecture does not match current model contract")
        curriculum_order = (
            _curriculum_order(features, np.asarray(encoded_target))
            if (self.training_schedule == "curriculum" and self.task == "classification")
            else (
                _curriculum_order(features, None)
                if self.training_schedule == "curriculum"
                else np.arange(len(features))
            )
        )
        self.models = []
        self.optimizer_steps = 0
        self.training_examples = len(features)
        for member_index in range(ensemble_size):
            member_seed = self.seed + member_index
            _seed_torch(torch, member_seed)
            model = _build_mlp(
                torch,
                input_dim=features.shape[1],
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                dropout=dropout,
            ).to(self.device)
            if checkpoint_manifest is not None:
                _restore_member_state(
                    torch,
                    model,
                    checkpoint_manifest["model"],
                    checkpoint_arrays,
                    member_index,
                )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
            if self.task == "classification":
                if self.loss_name != "cross_entropy":
                    raise ValueError("torch_mlp classification loss must be 'cross_entropy'")
                loss_function = torch.nn.CrossEntropyLoss()
            elif self.loss_name == "mse":
                loss_function = torch.nn.MSELoss()
            elif self.loss_name == "huber":
                loss_function = torch.nn.SmoothL1Loss(beta=1.0)
            else:
                raise ValueError("torch_mlp regression loss must be 'mse' or 'huber'")
            model.train()
            for epoch_index in range(epochs):
                epoch_indices = _epoch_training_indices(
                    len(features),
                    curriculum_order,
                    epoch_index=epoch_index,
                    epochs=epochs,
                    start_fraction=curriculum_start,
                    schedule=self.training_schedule,
                    seed=member_seed,
                )
                dataset = torch.utils.data.TensorDataset(
                    feature_tensor[epoch_indices], target_tensor[epoch_indices]
                )
                loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=min(batch_size, len(dataset)),
                    shuffle=False,
                    num_workers=0,
                )
                for batch_features, batch_target in loader:
                    batch_features = batch_features.to(self.device)
                    batch_target = batch_target.to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_function(model(batch_features), batch_target)
                    loss.backward()
                    optimizer.step()
                    self.optimizer_steps += 1
            model.eval()
            self.models.append(model)

    def predict(self, frame: pd.DataFrame) -> Prediction:
        if not self.models:
            raise RuntimeError("model has not been fitted")
        torch = _torch()
        features = torch.as_tensor(self.featurize(frame), dtype=torch.float32, device=self.device)
        mc_passes = max(1, int(self.parameters.get("mc_dropout_passes", 20)))
        samples: list[np.ndarray] = []
        with torch.no_grad():
            for model in self.models:
                for _ in range(mc_passes):
                    _enable_dropout_only(torch, model)
                    raw = model(features)
                    if self.task == "classification":
                        raw = torch.softmax(raw, dim=1)
                    samples.append(raw.detach().cpu().numpy())
                model.eval()
        stacked = np.asarray(samples, dtype=float)
        if self.task == "classification":
            probabilities = _normalize_probabilities(stacked.mean(axis=0))
            entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1)
            if probabilities.shape[1] > 1:
                entropy /= np.log(probabilities.shape[1])
            if self.classes is None:
                raise RuntimeError("classification labels are unavailable")
            labels = self.classes[np.argmax(probabilities, axis=1)]
            mean = (
                probabilities[:, -1] if probabilities.shape[1] == 2 else probabilities.max(axis=1)
            )
            return Prediction(
                mean=np.asarray(mean),
                uncertainty=np.asarray(entropy),
                labels=np.asarray(labels),
                probabilities=np.asarray(probabilities),
                classes=np.asarray(self.classes),
            )
        denormalized = stacked[..., 0] * self.target_scale + self.target_mean
        return Prediction(mean=denormalized.mean(axis=0), uncertainty=denormalized.std(axis=0))

    def featurize(self, frame: pd.DataFrame) -> np.ndarray:
        return self.featurizer.transform(frame).astype(np.float32, copy=False)

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "pytorch",
            "model": "mlp_ensemble_mc_dropout",
            "requested_device": self.requested_device,
            "device": self.device,
            "accelerator_used": self.device in {"cuda", "mps"},
            "ensemble_size": int(self.parameters.get("ensemble_size", 3)),
            "mc_dropout_passes": int(self.parameters.get("mc_dropout_passes", 20)),
            "loss": self.loss_name,
            "training_schedule": self.training_schedule,
            "optimizer_steps": self.optimizer_steps,
            "training_examples": self.training_examples,
            "lineage_training_fits": self.lineage_training_fits,
            "checkpoint_initialization": self.initialization_public,
        }

    def checkpoint_payload(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """Return a JSON manifest fragment and non-pickle member arrays."""

        if not self.models:
            raise RuntimeError("cannot checkpoint an unfitted model")
        arrays: dict[str, np.ndarray] = {}
        members: list[list[dict[str, str]]] = []
        for member_index, model in enumerate(self.models):
            member_contract: list[dict[str, str]] = []
            for parameter_index, (state_name, value) in enumerate(model.state_dict().items()):
                array_name = f"member_{member_index:03d}_state_{parameter_index:03d}"
                arrays[array_name] = value.detach().cpu().numpy()
                member_contract.append({"state_name": state_name, "array_name": array_name})
            members.append(member_contract)
        first_linear = next(
            module for module in self.models[0].modules() if hasattr(module, "in_features")
        )
        last_linear = [
            module for module in self.models[0].modules() if hasattr(module, "out_features")
        ][-1]
        manifest = {
            "backend": "torch_mlp",
            "task": self.task,
            "architecture": {
                "input_dim": int(first_linear.in_features),
                "hidden_dims": [
                    int(value) for value in self.parameters.get("hidden_dims", [256, 128])
                ],
                "output_dim": int(last_linear.out_features),
                "ensemble_size": len(self.models),
            },
            "featurizer": self.featurizer.checkpoint_metadata(),
            "classes": self.classes.tolist() if self.classes is not None else None,
            "target_normalization": {
                "mean": self.target_mean,
                "scale": self.target_scale,
            },
            "members": members,
            "lineage_training_fits": self.lineage_training_fits,
            "training_schedule": self.training_schedule,
            "optimizer_steps_last_fit": self.optimizer_steps,
        }
        return manifest, arrays


class ChempropSurrogate:
    """Chemprop v2 GPU adapter for molecular regression and binary classification."""

    _ALLOWED_PARAMETERS: ClassVar[set[str]] = {
        "epochs",
        "batch_size",
        "depth",
        "message_hidden_dim",
        "ffn_hidden_dim",
        "ffn_num_layers",
        "dropout",
        "learning_rate",
        "ensemble_size",
        "mc_dropout_passes",
        "num_workers",
        "split_sizes",
        "warmup_epochs",
    }

    def __init__(
        self,
        *,
        task: str,
        featurizer: Featurizer,
        smiles_column: str,
        seed: int,
        parameters: dict[str, Any],
        requested_device: str,
        require_accelerator: bool,
    ) -> None:
        unknown = sorted(set(parameters).difference(self._ALLOWED_PARAMETERS))
        if unknown:
            raise ValueError(f"unknown chemprop parameters: {unknown}")
        executable = shutil.which("chemprop")
        if not executable:
            raise RuntimeError(
                "Chemprop CLI not found; install with pip install "
                "'agentic-active-autoresearch[chemprop]'"
            )
        self.executable = executable
        self.task = task
        self.featurizer = featurizer
        self.smiles_column = smiles_column
        self.seed = seed
        self.parameters = dict(parameters)
        self.requested_device = requested_device
        self.require_accelerator = require_accelerator
        self.device = "unresolved"
        self.classes: np.ndarray | None = None
        self._temporary = tempfile.TemporaryDirectory(prefix="agentic-autoresearch-chemprop-")
        self.work_dir = Path(self._temporary.name)
        self.checkpoints: list[Path] = []

    def close(self) -> None:
        """Release the private Chemprop workspace deterministically."""

        self._temporary.cleanup()

    def __del__(self) -> None:  # pragma: no cover - interpreter/GC timing is nondeterministic
        temporary = getattr(self, "_temporary", None)
        if temporary is not None:
            temporary.cleanup()

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> None:
        self.device = _resolve_torch_device(
            self.requested_device, require_accelerator=self.require_accelerator
        )
        self.featurizer.fit_transform(frame)
        training_target: np.ndarray
        task_type: str
        if self.task == "classification":
            self.classes, training_target = np.unique(target, return_inverse=True)
            if len(self.classes) != 2:
                raise ValueError("Chemprop adapter currently supports binary classification only")
            task_type = "classification"
        else:
            training_target = np.asarray(target, dtype=float)
            task_type = "regression"
        train_path = self.work_dir / "train.csv"
        model_dir = self.work_dir / "model"
        pd.DataFrame(
            {
                self.smiles_column: frame[self.smiles_column].astype(str),
                "__target__": training_target,
            }
        ).to_csv(train_path, index=False)
        split_sizes = self.parameters.get("split_sizes", [0.8, 0.1, 0.1])
        if len(split_sizes) != 3 or abs(sum(float(value) for value in split_sizes) - 1.0) > 1e-6:
            raise ValueError("chemprop split_sizes must contain three values summing to one")
        epochs = int(self.parameters.get("epochs", 50))
        warmup_epochs = int(self.parameters.get("warmup_epochs", min(2, max(0, epochs - 1))))
        if epochs < 1 or warmup_epochs < 0 or warmup_epochs >= epochs:
            raise ValueError("chemprop warmup_epochs must satisfy 0 <= warmup_epochs < epochs")
        command = [
            self.executable,
            "train",
            "--data-path",
            str(train_path),
            "--output-dir",
            str(model_dir),
            "--smiles-columns",
            self.smiles_column,
            "--target-columns",
            "__target__",
            "--task-type",
            task_type,
            "--epochs",
            str(epochs),
            "--warmup-epochs",
            str(warmup_epochs),
            "--batch-size",
            str(int(self.parameters.get("batch_size", 64))),
            "--depth",
            str(int(self.parameters.get("depth", 4))),
            "--message-hidden-dim",
            str(int(self.parameters.get("message_hidden_dim", 600))),
            "--ffn-hidden-dim",
            str(int(self.parameters.get("ffn_hidden_dim", 600))),
            "--ffn-num-layers",
            str(int(self.parameters.get("ffn_num_layers", 2))),
            "--dropout",
            str(float(self.parameters.get("dropout", 0.1))),
            "--max-lr",
            str(float(self.parameters.get("learning_rate", 5e-4))),
            "--ensemble-size",
            str(int(self.parameters.get("ensemble_size", 1))),
            "--num-workers",
            str(int(self.parameters.get("num_workers", 0))),
            "--split",
            "RANDOM",
            "--split-sizes",
            *(str(float(value)) for value in split_sizes),
            "--data-seed",
            str(self.seed),
            "--pytorch-seed",
            str(self.seed),
            "--accelerator",
            _lightning_accelerator(self.device),
            "--devices",
            "1",
            "-q",
        ]
        _run_checked(command)
        self.checkpoints = sorted(model_dir.rglob("best.pt"))
        if not self.checkpoints:
            self.checkpoints = sorted(model_dir.rglob("best*.ckpt"))
        if not self.checkpoints:
            raise RuntimeError("Chemprop training completed without a discoverable checkpoint")

    def predict(self, frame: pd.DataFrame) -> Prediction:
        if not self.checkpoints:
            raise RuntimeError("model has not been fitted")
        input_path = self.work_dir / "predict_input.csv"
        output_path = self.work_dir / "predict_output.csv"
        frame[[self.smiles_column]].to_csv(input_path, index=False)
        command = [
            self.executable,
            "predict",
            "--test-path",
            str(input_path),
            "--output",
            str(output_path),
            "--model-paths",
            *(str(path) for path in self.checkpoints),
            "--uncertainty-method",
            "dropout",
            "--dropout-sampling-size",
            str(int(self.parameters.get("mc_dropout_passes", 20))),
            "--uncertainty-dropout-p",
            str(float(self.parameters.get("dropout", 0.1))),
            "--accelerator",
            _lightning_accelerator(self.device),
            "--devices",
            "1",
            "-q",
        ]
        _run_checked(command)
        output = pd.read_csv(output_path)
        uncertainty_columns = [column for column in output if column.endswith("_unc")]
        prediction_columns = [
            column
            for column in output
            if column not in {self.smiles_column, *uncertainty_columns}
            and pd.api.types.is_numeric_dtype(output[column])
        ]
        if not prediction_columns:
            raise RuntimeError("Chemprop prediction output contains no numeric prediction column")
        mean = output[prediction_columns[0]].to_numpy(dtype=float)
        uncertainty = (
            output[uncertainty_columns[0]].to_numpy(dtype=float)
            if uncertainty_columns
            else np.zeros(len(output), dtype=float)
        )
        if self.task == "classification":
            probabilities = _normalize_probabilities(np.column_stack([1.0 - mean, mean]))
            mean = probabilities[:, -1]
            classes = self.classes if self.classes is not None else np.asarray([0, 1])
            labels = classes[(mean >= 0.5).astype(int)]
            return Prediction(
                mean=mean,
                uncertainty=uncertainty,
                labels=labels,
                probabilities=probabilities,
                classes=classes,
            )
        return Prediction(mean=mean, uncertainty=uncertainty)

    def featurize(self, frame: pd.DataFrame) -> np.ndarray:
        return self.featurizer.transform(frame)

    def metadata(self) -> dict[str, Any]:
        try:
            version = metadata.version("chemprop")
        except metadata.PackageNotFoundError:
            version = "unknown"
        return {
            "backend": "chemprop",
            "chemprop_version": version,
            "model": "directed_message_passing_neural_network",
            "requested_device": self.requested_device,
            "device": self.device,
            "accelerator_used": self.device in {"cuda", "mps"},
            "ensemble_size": int(self.parameters.get("ensemble_size", 1)),
            "uncertainty": "mc_dropout",
            "mc_dropout_passes": int(self.parameters.get("mc_dropout_passes", 20)),
        }


def build_model(
    config: ModelConfig,
    *,
    task: str,
    feature_columns: list[str],
    seed: int,
    initialization: CheckpointReference | None = None,
) -> SurrogateModel:
    name = config.name.strip().lower().replace("-", "_")
    if name in MODEL_FACTORIES:
        model = MODEL_FACTORIES[name](
            config=config, task=task, feature_columns=feature_columns, seed=seed
        )
        if initialization is not None:
            setter = getattr(model, "set_checkpoint_initialization", None)
            if not callable(setter):
                raise ValueError(f"model '{config.name}' does not support safe checkpoint loading")
            setter(initialization)
        return model
    raise ValueError(f"unknown model '{config.name}'. Registered models: {sorted(MODEL_FACTORIES)}")


def _random_forest_factory(
    *, config: ModelConfig, task: str, feature_columns: list[str], seed: int
) -> RandomForestSurrogate:
    return RandomForestSurrogate(
        task=task,
        featurizer=build_featurizer(config, feature_columns),
        seed=seed,
        parameters=config.parameters,
    )


def _torch_mlp_factory(
    *, config: ModelConfig, task: str, feature_columns: list[str], seed: int
) -> TorchMLPSurrogate:
    return TorchMLPSurrogate(
        task=task,
        featurizer=build_featurizer(config, feature_columns),
        seed=seed,
        parameters=config.parameters,
        requested_device=config.device,
        require_accelerator=config.require_accelerator,
        feature_columns=feature_columns,
    )


def _chemprop_factory(
    *, config: ModelConfig, task: str, feature_columns: list[str], seed: int
) -> ChempropSurrogate:
    if config.featurizer != "morgan":
        raise ValueError(
            "Chemprop requires model.featurizer='morgan' for acquisition-space features"
        )
    return ChempropSurrogate(
        task=task,
        featurizer=build_featurizer(config, feature_columns),
        smiles_column=config.smiles_column,
        seed=seed,
        parameters=config.parameters,
        requested_device=config.device,
        require_accelerator=config.require_accelerator,
    )


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch_mlp requires PyTorch 2.2 or newer") from exc
    return torch


def _resolve_torch_device(requested: str, *, require_accelerator: bool) -> str:
    torch = _torch()
    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return "cuda"
    if requested == "mps":
        if not mps_available:
            raise RuntimeError("Apple MPS was requested but is not available")
        _reject_mps_cpu_fallback(require_accelerator)
        return "mps"
    if requested == "cpu":
        if require_accelerator:
            raise RuntimeError("model.require_accelerator=true forbids CPU training")
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if mps_available:
        _reject_mps_cpu_fallback(require_accelerator)
        return "mps"
    if require_accelerator:
        raise RuntimeError("No CUDA or Apple MPS accelerator is available")
    return "cpu"


def _reject_mps_cpu_fallback(require_accelerator: bool) -> None:
    if require_accelerator and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "").strip() == "1":
        raise RuntimeError(
            "formal MPS training forbids PYTORCH_ENABLE_MPS_FALLBACK=1 because unsupported "
            "operations could run on CPU"
        )


def _seed_torch(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _build_mlp(
    torch: Any, *, input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float
) -> Any:
    layers: list[Any] = []
    current = input_dim
    for hidden in hidden_dims:
        layers.extend(
            [torch.nn.Linear(current, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
        )
        current = hidden
    layers.append(torch.nn.Linear(current, output_dim))
    return torch.nn.Sequential(*layers)


def _restore_member_state(
    torch: Any,
    model: Any,
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    member_index: int,
) -> None:
    members = manifest.get("members")
    if not isinstance(members, list) or member_index >= len(members):
        raise ValueError("checkpoint ensemble member contract is incomplete")
    state: dict[str, Any] = {}
    for entry in members[member_index]:
        state_name = str(entry["state_name"])
        array_name = str(entry["array_name"])
        if array_name not in arrays:
            raise ValueError(f"checkpoint state array is missing: {array_name}")
        state[state_name] = torch.as_tensor(arrays[array_name])
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("checkpoint state keys do not match the current network")
    for name, value in state.items():
        if tuple(value.shape) != tuple(expected[name].shape):
            raise ValueError(f"checkpoint tensor shape mismatch for '{name}'")
    model.load_state_dict(state, strict=True)


def _curriculum_order(features: np.ndarray, labels: np.ndarray | None) -> np.ndarray:
    """Return a deterministic easy-to-hard order using training features only."""

    values = np.asarray(features, dtype=np.float64)
    if labels is None:
        centroid = values.mean(axis=0, keepdims=True)
        difficulty = np.linalg.norm(values - centroid, axis=1)
        return np.lexsort((np.arange(len(values)), difficulty)).astype(int)
    class_orders: list[np.ndarray] = []
    labels = np.asarray(labels)
    for class_value in np.unique(labels):
        positions = np.flatnonzero(labels == class_value)
        centroid = values[positions].mean(axis=0, keepdims=True)
        difficulty = np.linalg.norm(values[positions] - centroid, axis=1)
        class_orders.append(positions[np.lexsort((positions, difficulty))])
    interleaved: list[int] = []
    for rank in range(max(len(values) for values in class_orders)):
        for order in class_orders:
            if rank < len(order):
                interleaved.append(int(order[rank]))
    return np.asarray(interleaved, dtype=int)


def _epoch_training_indices(
    sample_count: int,
    curriculum_order: np.ndarray,
    *,
    epoch_index: int,
    epochs: int,
    start_fraction: float,
    schedule: str,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed + epoch_index)
    if schedule == "standard":
        return rng.permutation(sample_count).astype(int)
    progress = 1.0 if epochs == 1 else epoch_index / (epochs - 1)
    fraction = start_fraction + (1.0 - start_fraction) * progress
    available = max(1, min(sample_count, int(np.ceil(sample_count * fraction))))
    repeated = np.resize(curriculum_order[:available], sample_count)
    return repeated[rng.permutation(sample_count)].astype(int)


def _enable_dropout_only(torch: Any, model: Any) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def _normalize_probabilities(values: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("classification probabilities must have at least two columns")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("classification probabilities must be finite")
    probabilities = np.clip(probabilities, 0.0, 1.0)
    totals = probabilities.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("classification probability rows must have positive mass")
    probabilities = probabilities / totals
    probabilities[:, -1] = 1.0 - probabilities[:, :-1].sum(axis=1)
    return probabilities


def _lightning_accelerator(device: str) -> str:
    return "gpu" if device == "cuda" else device


def _run_checked(command: list[str]) -> None:
    # Security: the executable is an absolute shutil.which result and shell execution is disabled.
    completed = subprocess.run(  # nosec B603
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_sanitized_subprocess_env(),
        timeout=24 * 60 * 60,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-2_000:]
        detail = re.sub(r"/(?:Users|home)/[^/\s]+", "${HOME}", detail)
        raise RuntimeError(
            f"Chemprop command failed with exit code {completed.returncode}: {detail}"
        )


def _sanitized_subprocess_env() -> dict[str, str]:
    # Chemprop is a trusted local dependency, but it does not need the parent process's arbitrary
    # credentials or experiment metadata. An allowlist is safer than trying to guess every secret
    # variable name. Keep only runtime, accelerator, locale, certificate, and cache settings.
    allowed = {
        "CONDA_DEFAULT_ENV",
        "CONDA_PREFIX",
        "CPATH",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "DYLD_LIBRARY_PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "MKL_NUM_THREADS",
        "MPLCONFIGDIR",
        "NVIDIA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PATH",
        "PYTHONNOUSERSITE",
        "PYTORCH_ENABLE_MPS_FALLBACK",
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TORCH_HOME",
        "VECLIB_MAXIMUM_THREADS",
        "VIRTUAL_ENV",
        "XDG_CACHE_HOME",
    }
    environment = {
        name: value
        for name, value in os.environ.items()
        if name in allowed or name.startswith("LC_")
    }
    # Do not let an optional experiment tracker silently export run metadata.
    environment["WANDB_MODE"] = "disabled"
    environment["WANDB_SILENT"] = "true"
    return environment


register_model("random_forest", _random_forest_factory)
register_model("torch_mlp", _torch_mlp_factory)
register_model("chemprop", _chemprop_factory)
