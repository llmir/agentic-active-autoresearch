from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentic_al.config import AppConfig
from agentic_al.data import choose_initial_ids, load_dataset, split_for_evaluation


def _csv_config(path: Path, **dataset_overrides: object) -> AppConfig:
    dataset = {
        "kind": "csv",
        "path": path,
        "task": "regression",
        "target_column": "y",
        "feature_columns": ["x"],
        **dataset_overrides,
    }
    return AppConfig.model_validate(
        {
            "dataset": dataset,
            "model": {
                "name": "random_forest",
                "require_accelerator": False,
                "parameters": {"n_estimators": 8},
            },
        }
    )


def test_csv_generates_stable_ids(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]}).to_csv(path, index=False)
    bundle = load_dataset(_csv_config(path))
    assert bundle.frame["sample_id"].tolist() == [
        "row-00000000",
        "row-00000001",
        "row-00000002",
        "row-00000003",
    ]
    assert "y" not in bundle.feature_columns


def test_unlabeled_regression_pool_can_be_loaded_for_an_external_oracle(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pool.csv"
    pd.DataFrame({"sample_id": ["p-1", "p-2"], "x": [1.0, 2.0]}).to_csv(path, index=False)
    bundle = load_dataset(_csv_config(path), require_target=False)
    assert "y" not in bundle.frame.columns
    validation = pd.DataFrame({"sample_id": ["v-1", "v-2"], "x": [3.0, 4.0], "y": [6.0, 8.0]})
    candidates, held_out = split_for_evaluation(bundle, 0.2, 7, external_validation=validation)
    assert candidates["sample_id"].tolist() == ["p-1", "p-2"]
    assert held_out["sample_id"].tolist() == ["v-1", "v-2"]


def test_external_validation_must_be_disjoint_and_complete(tmp_path: Path) -> None:
    path = tmp_path / "pool.csv"
    pd.DataFrame({"sample_id": ["p-1", "p-2"], "x": [1.0, 2.0]}).to_csv(path, index=False)
    bundle = load_dataset(_csv_config(path), require_target=False)
    with pytest.raises(ValueError, match="IDs must be disjoint"):
        split_for_evaluation(
            bundle,
            0.2,
            7,
            external_validation=pd.DataFrame(
                {"sample_id": ["p-1", "v-2"], "x": [3.0, 4.0], "y": [6.0, 8.0]}
            ),
        )
    with pytest.raises(ValueError, match="missing required"):
        split_for_evaluation(
            bundle,
            0.2,
            7,
            external_validation=pd.DataFrame({"sample_id": ["v-1", "v-2"], "x": [3.0, 4.0]}),
        )


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame({"sample_id": ["a", "a"], "x": [1, 2], "y": [2, 4]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_dataset(_csv_config(path))


def test_ids_must_remain_unique_after_string_normalization(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    pd.DataFrame(
        {
            "sample_id": [1, "1"],
            "x": [1.0, 2.0],
            "y": [2.0, 4.0],
        }
    ).to_json(path, orient="records", lines=True)
    with pytest.raises(ValueError, match="string normalization"):
        load_dataset(_csv_config(path, kind="jsonl"))


def test_missing_groups_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "grouped.csv"
    pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "x": [1.0, 2.0, 3.0],
            "y": [2.0, 4.0, 6.0],
            "group": ["one", None, "two"],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="group_column"):
        load_dataset(_csv_config(path, group_column="group"))


def test_group_split_has_no_overlap() -> None:
    config = AppConfig.model_validate(
        {
            "dataset": {"kind": "synthetic", "synthetic_samples": 100},
            "model": {"require_accelerator": False},
        }
    )
    bundle = load_dataset(config)
    bundle.frame["group"] = [f"g-{index // 5}" for index in range(len(bundle.frame))]
    bundle.group_column = "group"
    candidate, validation = split_for_evaluation(bundle, 0.2, 42)
    assert set(candidate["group"]).isdisjoint(validation["group"])


def test_group_classification_split_retains_every_class() -> None:
    config = AppConfig.model_validate(
        {
            "dataset": {
                "kind": "synthetic",
                "task": "classification",
                "synthetic_samples": 90,
                "synthetic_classes": 3,
            },
            "model": {"require_accelerator": False},
            "acquisition": {"goal": "target_class", "target_class": 1},
        }
    )
    bundle = load_dataset(config)
    bundle.frame = pd.DataFrame(
        {
            "sample_id": [f"row-{index:03d}" for index in range(30)],
            "feature": np.arange(30, dtype=float),
            "target": [value for _ in range(10) for value in (0, 1, 2)],
            "group": [f"group-{index // 3}" for index in range(30)],
        }
    )
    bundle.feature_columns = ["feature"]
    bundle.target_column = "target"
    bundle.group_column = "group"
    candidates, validation = split_for_evaluation(bundle, 0.2, 7)
    assert set(candidates["target"]) == {0, 1, 2}
    assert set(validation["target"]) == {0, 1, 2}
    assert set(candidates["group"]).isdisjoint(validation["group"])


def test_group_classification_split_fails_when_classes_are_group_confounded() -> None:
    config = AppConfig.model_validate(
        {
            "dataset": {"kind": "synthetic", "task": "classification"},
            "model": {"require_accelerator": False},
            "acquisition": {"goal": "target_class", "target_class": 1},
        }
    )
    bundle = load_dataset(config)
    bundle.frame = pd.DataFrame(
        {
            "sample_id": [f"row-{index:03d}" for index in range(20)],
            "feature": np.arange(20, dtype=float),
            "target": [0] * 10 + [1] * 10,
            "group": ["only-class-zero"] * 10 + ["only-class-one"] * 10,
        }
    )
    bundle.feature_columns = ["feature"]
    bundle.target_column = "target"
    bundle.group_column = "group"
    with pytest.raises(ValueError, match="every class"):
        split_for_evaluation(bundle, 0.2, 7)


def test_classification_initial_set_contains_each_class() -> None:
    config = AppConfig.model_validate(
        {
            "dataset": {"kind": "synthetic", "task": "classification", "synthetic_samples": 100},
            "model": {"require_accelerator": False},
            "acquisition": {"goal": "target_class", "target_class": 1},
        }
    )
    bundle = load_dataset(config)
    ids = choose_initial_ids(
        bundle.frame,
        bundle.id_column,
        20,
        42,
        task="classification",
        target_column=bundle.target_column,
    )
    chosen = bundle.frame[bundle.frame[bundle.id_column].isin(ids)]
    assert chosen[bundle.target_column].nunique() == 2


@pytest.mark.parametrize(
    ("frame", "overrides", "message"),
    [
        (pd.DataFrame({"x": [1], "y": [2]}), {"feature_columns": ["y"]}, "target_column"),
        (pd.DataFrame({"sample_id": [None], "x": [1], "y": [2]}), {}, "non-null"),
        (pd.DataFrame({"x": [1], "y": [None]}), {}, "missing values"),
        (
            pd.DataFrame({"x": [1], "y": [2], "cost": ["bad"]}),
            {"cost_column": "cost"},
            "finite numeric",
        ),
        (pd.DataFrame({"x": [1], "y": [2]}), {"feature_columns": ["missing"]}, "missing"),
    ],
)
def test_csv_validation_errors(
    tmp_path: Path, frame: pd.DataFrame, overrides: dict, message: str
) -> None:
    path = tmp_path / "invalid.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        load_dataset(_csv_config(path, **overrides))


def test_missing_csv_and_oversized_initial_set_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(_csv_config(tmp_path / "missing.csv"))
    frame = pd.DataFrame({"sample_id": ["a", "b"], "y": [1, 2]})
    with pytest.raises(ValueError, match="leave at least one"):
        choose_initial_ids(frame, "sample_id", 2, 1, task="regression", target_column="y")


def test_non_group_split_is_deterministic_and_disjoint() -> None:
    config = AppConfig.model_validate(
        {
            "dataset": {"kind": "synthetic", "synthetic_samples": 80},
            "model": {"require_accelerator": False},
        }
    )
    bundle = load_dataset(config)
    candidates, validation = split_for_evaluation(bundle, 0.2, 13)
    assert len(candidates) == 64
    assert set(candidates[bundle.id_column]).isdisjoint(validation[bundle.id_column])
    assert candidates[bundle.id_column].tolist() == sorted(candidates[bundle.id_column])


def test_jsonl_loader_uses_the_same_validation_contract(tmp_path: Path) -> None:
    path = tmp_path / "pool.jsonl"
    pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]}).to_json(path, orient="records", lines=True)
    bundle = load_dataset(_csv_config(path, kind="jsonl"))
    assert bundle.frame["sample_id"].tolist() == [
        "row-00000000",
        "row-00000001",
        "row-00000002",
    ]


def test_parquet_loader_is_optional_and_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pool.parquet"
    path.touch()
    monkeypatch.setattr(pd, "read_parquet", lambda source: pd.DataFrame({"x": [1], "y": [2]}))
    assert len(load_dataset(_csv_config(path, kind="parquet")).frame) == 1

    def unavailable(source):
        raise ImportError("no parquet engine")

    monkeypatch.setattr(pd, "read_parquet", unavailable)
    with pytest.raises(RuntimeError, match=r"agentic-active-autoresearch\[parquet\]"):
        load_dataset(_csv_config(path, kind="parquet"))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("x", np.inf, "numeric feature column 'x' contains infinite values"),
        ("y", np.inf, "regression target_column must contain finite numeric values"),
    ],
)
def test_tabular_loader_rejects_infinite_training_values(
    tmp_path: Path, column: str, value: float, message: str
) -> None:
    path = tmp_path / "invalid.csv"
    frame = pd.DataFrame({"sample_id": ["a", "b", "c"], "x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
    frame.loc[1, column] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        load_dataset(_csv_config(path))


def test_classification_loader_requires_multiple_classes(tmp_path: Path) -> None:
    path = tmp_path / "single-class.csv"
    pd.DataFrame({"sample_id": ["a", "b"], "x": [1.0, 2.0], "y": [1, 1]}).to_csv(path, index=False)
    config = _csv_config(path).model_copy(
        update={"dataset": _csv_config(path).dataset.model_copy(update={"task": "classification"})}
    )
    with pytest.raises(ValueError, match="at least two classes"):
        load_dataset(config)
