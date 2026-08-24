from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentic_al import models
from agentic_al.config import ModelConfig
from agentic_al.models import ChempropSurrogate, build_model


class DummyFeaturizer:
    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return np.zeros((len(frame), 2), dtype=np.float32)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return np.zeros((len(frame), 2), dtype=np.float32)


def test_random_forest_regression_prediction_shape() -> None:
    frame = pd.DataFrame({"x": np.linspace(0, 1, 30), "kind": ["a", "b"] * 15})
    model = build_model(
        ModelConfig(
            name="random_forest",
            device="cpu",
            require_accelerator=False,
            parameters={"n_estimators": 12, "min_samples_leaf": 1, "n_jobs": 1},
        ),
        task="regression",
        feature_columns=["x", "kind"],
        seed=42,
    )
    model.fit(frame, np.linspace(0, 2, 30))
    prediction = model.predict(frame.iloc[:5])
    assert prediction.mean.shape == (5,)
    assert prediction.uncertainty.shape == (5,)
    assert model.metadata()["purpose"] == "baseline_or_smoke_test"


def test_torch_mlp_runs_small_cpu_unit_path() -> None:
    frame = pd.DataFrame({"x": np.linspace(-1, 1, 40), "x2": np.linspace(1, -1, 40)})
    model = build_model(
        ModelConfig(
            name="torch_mlp",
            device="cpu",
            require_accelerator=False,
            parameters={
                "hidden_dims": [8],
                "dropout": 0.1,
                "epochs": 2,
                "batch_size": 16,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "ensemble_size": 1,
                "mc_dropout_passes": 2,
            },
        ),
        task="regression",
        feature_columns=["x", "x2"],
        seed=42,
    )
    model.fit(frame, (frame["x"] * 2).to_numpy())
    prediction = model.predict(frame.iloc[:4])
    assert prediction.mean.shape == (4,)
    assert model.metadata()["device"] == "cpu"


def test_torch_mlp_supports_a_real_huber_loss_candidate() -> None:
    frame = pd.DataFrame({"x": np.linspace(-1, 1, 24)})
    model = build_model(
        ModelConfig(
            name="torch_mlp",
            device="cpu",
            require_accelerator=False,
            parameters={
                "hidden_dims": [6],
                "loss": "huber",
                "epochs": 1,
                "batch_size": 8,
                "ensemble_size": 1,
                "mc_dropout_passes": 1,
            },
        ),
        task="regression",
        feature_columns=["x"],
        seed=8,
    )
    model.fit(frame, (2.0 * frame["x"]).to_numpy())
    assert model.metadata()["loss"] == "huber"


def test_curriculum_preserves_the_standard_optimizer_step_budget() -> None:
    frame = pd.DataFrame({"x": np.linspace(-2, 2, 25)})
    common = {
        "hidden_dims": [6],
        "dropout": 0.0,
        "epochs": 3,
        "batch_size": 8,
        "ensemble_size": 2,
        "mc_dropout_passes": 1,
    }
    standard = build_model(
        ModelConfig(
            name="torch_mlp",
            device="cpu",
            require_accelerator=False,
            parameters=common,
        ),
        task="regression",
        feature_columns=["x"],
        seed=19,
    )
    curriculum = build_model(
        ModelConfig(
            name="torch_mlp",
            device="cpu",
            require_accelerator=False,
            parameters={
                **common,
                "training_schedule": "curriculum",
                "curriculum_start_fraction": 0.35,
            },
        ),
        task="regression",
        feature_columns=["x"],
        seed=19,
    )
    target = (frame["x"] ** 2).to_numpy()
    standard.fit(frame, target)
    curriculum.fit(frame, target)
    assert standard.metadata()["optimizer_steps"] == curriculum.metadata()["optimizer_steps"]
    assert curriculum.metadata()["optimizer_steps"] == 3 * 4 * 2
    assert curriculum.metadata()["training_schedule"] == "curriculum"


def test_torch_mlp_rejects_loss_task_mismatches() -> None:
    frame = pd.DataFrame({"x": np.linspace(-1, 1, 12)})
    classification = build_model(
        ModelConfig(
            name="torch_mlp",
            device="cpu",
            require_accelerator=False,
            parameters={"hidden_dims": [4], "loss": "huber", "epochs": 1},
        ),
        task="classification",
        feature_columns=["x"],
        seed=1,
    )
    with pytest.raises(ValueError, match="classification loss"):
        classification.fit(frame, np.asarray([0, 1] * 6))

    regression = build_model(
        ModelConfig(
            name="torch_mlp",
            device="cpu",
            require_accelerator=False,
            parameters={"hidden_dims": [4], "loss": "unknown", "epochs": 1},
        ),
        task="regression",
        feature_columns=["x"],
        seed=1,
    )
    with pytest.raises(ValueError, match="regression loss"):
        regression.fit(frame, frame["x"].to_numpy())


def test_random_forest_classification_contract() -> None:
    frame = pd.DataFrame({"x": np.arange(40), "kind": ["a", "b"] * 20})
    model = build_model(
        ModelConfig(
            name="random_forest",
            device="cpu",
            require_accelerator=False,
            parameters={"n_estimators": 12, "n_jobs": 1},
        ),
        task="classification",
        feature_columns=["x", "kind"],
        seed=4,
    )
    model.fit(frame, np.asarray([0, 1] * 20))
    prediction = model.predict(frame.iloc[:3])
    assert prediction.probabilities is not None
    assert prediction.probabilities.shape == (3, 2)
    np.testing.assert_allclose(prediction.probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-15)
    assert prediction.labels is not None


def test_torch_classification_and_gpu_guard() -> None:
    frame = pd.DataFrame({"x": np.linspace(-1, 1, 24)})
    model = build_model(
        ModelConfig(
            name="torch_mlp",
            device="cpu",
            require_accelerator=False,
            parameters={
                "hidden_dims": [6],
                "epochs": 1,
                "batch_size": 8,
                "ensemble_size": 1,
                "mc_dropout_passes": 1,
            },
        ),
        task="classification",
        feature_columns=["x"],
        seed=2,
    )
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.predict(frame.iloc[:2])
    model.fit(frame, np.asarray([0, 1] * 12))
    prediction = model.predict(frame.iloc[:2])
    assert prediction.labels is not None
    assert prediction.probabilities is not None
    np.testing.assert_allclose(prediction.probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-15)

    guarded = ModelConfig(
        name="torch_mlp",
        device="cpu",
        require_accelerator=True,
        parameters={"hidden_dims": [4]},
    )
    guarded_model = build_model(guarded, task="regression", feature_columns=["x"], seed=1)
    with pytest.raises(RuntimeError, match="forbids CPU"):
        guarded_model.fit(frame, frame["x"].to_numpy())


def test_invalid_model_configuration_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="unknown model"):
        build_model(ModelConfig(name="missing"), task="regression", feature_columns=["x"], seed=1)
    with pytest.raises(ValueError, match="unknown torch_mlp parameters"):
        build_model(
            ModelConfig(name="torch_mlp", parameters={"surprise": 1}),
            task="regression",
            feature_columns=["x"],
            seed=1,
        )
    with pytest.raises(ValueError, match="unknown chemprop parameters"):
        ChempropSurrogate(
            task="regression",
            featurizer=DummyFeaturizer(),
            smiles_column="smiles",
            seed=1,
            parameters={"executable": "/tmp/not-chemprop"},
            requested_device="cpu",
            require_accelerator=False,
        )
    with pytest.raises(ValueError, match=r"requires model\.featurizer"):
        build_model(
            ModelConfig(name="chemprop", featurizer="tabular", parameters={}),
            task="regression",
            feature_columns=["smiles"],
            seed=1,
        )
    monkeypatch.setattr(models.shutil, "which", lambda executable: None)
    with pytest.raises(RuntimeError, match="CLI not found"):
        ChempropSurrogate(
            task="regression",
            featurizer=DummyFeaturizer(),
            smiles_column="smiles",
            seed=1,
            parameters={},
            requested_device="cpu",
            require_accelerator=False,
        )


def test_formal_mps_path_rejects_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(RuntimeError, match="unsupported operations could run on CPU"):
        models._reject_mps_cpu_fallback(True)
    models._reject_mps_cpu_fallback(False)


def test_probability_normalization_rejects_invalid_rows() -> None:
    normalized = models._normalize_probabilities(np.asarray([[0.2, 0.2], [0.1, 0.2]]))
    np.testing.assert_allclose(normalized.sum(axis=1), 1.0, rtol=0.0, atol=1e-15)
    with pytest.raises(ValueError, match="positive mass"):
        models._normalize_probabilities(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        models._normalize_probabilities(np.asarray([[float("nan"), 1.0]]))


def test_chemprop_adapter_command_and_prediction_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models.shutil, "which", lambda executable: "/usr/local/bin/chemprop")
    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> None:
        commands.append(command)
        if command[1] == "train":
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir(parents=True)
            (output_dir / "best.pt").write_text("checkpoint", encoding="utf-8")
        else:
            output_path = command[command.index("--output") + 1]
            pd.DataFrame(
                {
                    "smiles": ["CC", "CO"],
                    "prediction": [0.25, 0.75],
                    "prediction_unc": [0.05, 0.10],
                }
            ).to_csv(output_path, index=False)

    monkeypatch.setattr(models, "_run_checked", fake_run)
    model = ChempropSurrogate(
        task="regression",
        featurizer=DummyFeaturizer(),
        smiles_column="smiles",
        seed=3,
        parameters={"epochs": 1, "split_sizes": [0.7, 0.2, 0.1]},
        requested_device="cpu",
        require_accelerator=False,
    )
    frame = pd.DataFrame({"smiles": ["CC", "CO"]})
    model.fit(frame, np.asarray([1.0, 2.0]))
    prediction = model.predict(frame)
    assert prediction.mean.tolist() == [0.25, 0.75]
    assert prediction.uncertainty.tolist() == [0.05, 0.1]
    assert commands[0][1] == "train"
    assert "--accelerator" in commands[0]
    assert commands[0][commands[0].index("--warmup-epochs") + 1] == "0"
    assert model.metadata()["backend"] == "chemprop"
    model.close()


def test_chemprop_validation_and_runner_redaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models.shutil, "which", lambda executable: "/usr/local/bin/chemprop")
    model = ChempropSurrogate(
        task="classification",
        featurizer=DummyFeaturizer(),
        smiles_column="smiles",
        seed=3,
        parameters={"split_sizes": [0.5, 0.5]},
        requested_device="cpu",
        require_accelerator=False,
    )
    frame = pd.DataFrame({"smiles": ["CC", "CO", "CN"]})
    with pytest.raises(ValueError, match="binary classification"):
        model.fit(frame, np.asarray([0, 1, 2]))
    model.close()

    regression = ChempropSurrogate(
        task="regression",
        featurizer=DummyFeaturizer(),
        smiles_column="smiles",
        seed=3,
        parameters={"split_sizes": [0.5, 0.5]},
        requested_device="cpu",
        require_accelerator=False,
    )
    with pytest.raises(ValueError, match="split_sizes"):
        regression.fit(frame, np.asarray([0.0, 1.0, 2.0]))
    regression.close()

    invalid_warmup = ChempropSurrogate(
        task="regression",
        featurizer=DummyFeaturizer(),
        smiles_column="smiles",
        seed=3,
        parameters={"epochs": 2, "warmup_epochs": 2},
        requested_device="cpu",
        require_accelerator=False,
    )
    with pytest.raises(ValueError, match="warmup_epochs"):
        invalid_warmup.fit(frame, np.asarray([0.0, 1.0, 2.0]))
    invalid_warmup.close()

    class Failed:
        returncode = 2
        stderr = "/Users/alice/private failed"
        stdout = ""

    monkeypatch.setattr(models.subprocess, "run", lambda *args, **kwargs: Failed())
    with pytest.raises(RuntimeError, match=r"\$\{HOME\}/private"):
        models._run_checked(["chemprop", "train"])
