import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentic_al.checkpoints import (
    discover_checkpoint_catalog,
    load_checkpoint,
    save_selected_checkpoint,
)
from agentic_al.config import ModelConfig
from agentic_al.models import build_model


def _model_config() -> ModelConfig:
    return ModelConfig(
        name="torch_mlp",
        device="cpu",
        require_accelerator=False,
        parameters={
            "hidden_dims": [8],
            "dropout": 0.0,
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "ensemble_size": 1,
            "mc_dropout_passes": 1,
        },
    )


def _committed_checkpoint(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "x": np.linspace(-1, 1, 24),
            "kind": ["a", "b"] * 12,
        }
    )
    target = (2.0 * frame["x"]).to_numpy()
    model = build_model(_model_config(), task="regression", feature_columns=["x", "kind"], seed=3)
    model.fit(frame, target)
    inner_dir = tmp_path / "round_000" / "inner_loop"
    manifest = save_selected_checkpoint(
        model,
        inner_dir,
        round_index=0,
        phase="active_round",
        selection_metric="mae",
        selection_mode="min",
        selection_value=0.2,
        candidate_name="baseline_equivalent_control",
        candidate_paradigm="baseline_equivalent_control",
    )
    assert manifest is not None
    (tmp_path / "round_000" / "commit.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "round": 0,
                "selected_checkpoint_sha256": manifest["weights_sha256"],
                "selected_checkpoint_manifest_sha256": manifest["manifest_sha256"],
            }
        ),
        encoding="utf-8",
    )
    catalog = discover_checkpoint_catalog(
        tmp_path,
        current_round=1,
        selection_metric="mae",
        selection_mode="min",
    )
    return frame, target, catalog


def test_safe_checkpoint_roundtrip_and_finetune_lineage(tmp_path: Path) -> None:
    frame, _target, catalog = _committed_checkpoint(tmp_path)
    assert set(catalog.sources) == {"global_best", "previous_round"}
    reference = catalog.sources["previous_round"]
    manifest, arrays = load_checkpoint(reference)
    assert manifest["format"] == "numpy_npz_allow_pickle_false"
    assert arrays

    extended = pd.concat(
        [frame, pd.DataFrame({"x": [1.5, 2.0], "kind": ["new", "a"]})],
        ignore_index=True,
    )
    extended_target = (2.0 * extended["x"]).to_numpy()
    model = build_model(
        _model_config(),
        task="regression",
        feature_columns=["x", "kind"],
        seed=4,
        initialization=reference,
    )
    model.fit(extended, extended_target)
    metadata = model.metadata()
    assert metadata["lineage_training_fits"] == 2
    assert metadata["checkpoint_initialization"]["source"] == "previous_round"
    assert metadata["optimizer_steps"] == 4
    assert model.predict(extended.iloc[:3]).mean.shape == (3,)


def test_corrupt_checkpoint_is_rejected_before_candidate_generation(tmp_path: Path) -> None:
    _, _, catalog = _committed_checkpoint(tmp_path)
    weights_path = catalog.sources["previous_round"].weights_path
    with weights_path.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        discover_checkpoint_catalog(
            tmp_path,
            current_round=1,
            selection_metric="mae",
            selection_mode="min",
        )


def test_global_best_uses_only_compatible_inner_metric(tmp_path: Path) -> None:
    _committed_checkpoint(tmp_path)
    catalog = discover_checkpoint_catalog(
        tmp_path,
        current_round=1,
        selection_metric="rmse",
        selection_mode="min",
    )
    assert set(catalog.sources) == {"previous_round"}


def test_manifest_array_bomb_is_rejected_before_npz_decompression(tmp_path: Path) -> None:
    _, _, catalog = _committed_checkpoint(tmp_path)
    manifest_path = catalog.sources["previous_round"].manifest_path
    manifest = json.loads(manifest_path.read_text())
    first_name = next(iter(manifest["array_contract"]))
    manifest["array_contract"][first_name]["shape"] = [100_000_001]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    commit_path = tmp_path / "round_000" / "commit.json"
    commit = json.loads(commit_path.read_text())
    commit["selected_checkpoint_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    commit_path.write_text(json.dumps(commit), encoding="utf-8")
    with pytest.raises(ValueError, match="uncompressed array limit"):
        discover_checkpoint_catalog(
            tmp_path,
            current_round=1,
            selection_metric="mae",
            selection_mode="min",
        )


def test_committed_manifest_tampering_is_rejected(tmp_path: Path) -> None:
    _, _, catalog = _committed_checkpoint(tmp_path)
    manifest_path = catalog.sources["previous_round"].manifest_path
    manifest = json.loads(manifest_path.read_text())
    manifest["origin"]["candidate_name"] = "silently_replaced"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest does not match the committed digest"):
        discover_checkpoint_catalog(
            tmp_path,
            current_round=1,
            selection_metric="mae",
            selection_mode="min",
        )


@pytest.mark.parametrize(
    ("arrays", "message"),
    [
        ({"../escape": np.ones(1)}, "array name is invalid"),
        ({"text": np.asarray(["unsafe"])}, "array dtype is unsafe"),
        ({1: np.ones(1), "1": np.ones(1)}, "duplicate normalized array names"),
    ],
)
def test_checkpoint_export_rejects_unsafe_plugin_arrays_before_writing(
    tmp_path: Path, arrays: dict[object, np.ndarray], message: str
) -> None:
    class UnsafeExporter:
        def checkpoint_payload(self):
            return {"backend": "unsafe_plugin"}, arrays

    with pytest.raises(ValueError, match=message):
        save_selected_checkpoint(
            UnsafeExporter(),
            tmp_path,
            round_index=0,
            phase="active_round",
            selection_metric="mae",
            selection_mode="min",
            selection_value=0.5,
            candidate_name="plugin",
            candidate_paradigm="retrain_from_scratch",
        )
    assert not (tmp_path / "selected_checkpoint.npz").exists()
