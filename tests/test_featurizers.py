import numpy as np
import pandas as pd
import pytest

from agentic_al.config import ModelConfig
from agentic_al.featurizers import (
    FrozenTabularFeaturizer,
    MorganFeaturizer,
    TabularFeaturizer,
    build_featurizer,
    featurizer_from_checkpoint,
)


def test_tabular_featurizer_handles_numeric_categorical_and_unknown() -> None:
    train = pd.DataFrame({"x": [1.0, np.nan, 3.0], "kind": ["a", "b", None]})
    featurizer = TabularFeaturizer(["x", "kind"])
    encoded = featurizer.fit_transform(train)
    transformed = featurizer.transform(pd.DataFrame({"x": [2.0], "kind": ["new"]}))
    assert encoded.shape[0] == 3
    assert transformed.shape == (1, encoded.shape[1])
    assert np.isfinite(encoded).all()


def test_transform_before_fit_and_invalid_morgan_config() -> None:
    with pytest.raises(RuntimeError, match="not been fitted"):
        TabularFeaturizer(["x"]).transform(pd.DataFrame({"x": [1]}))
    with pytest.raises(RuntimeError, match="not been fitted"):
        TabularFeaturizer(["x"]).checkpoint_metadata()
    with pytest.raises(ValueError, match="smiles_column"):
        build_featurizer(ModelConfig(featurizer="morgan"), ["x"])


def test_frozen_tabular_checkpoint_reproduces_original_transform() -> None:
    train = pd.DataFrame(
        {
            "x": [1.0, np.nan, 4.0, 8.0],
            "count": [1, 2, 3, 4],
            "kind": ["a", "b", None, "a"],
        }
    )
    probe = pd.DataFrame({"x": [np.nan, 6.0], "count": [5, 1], "kind": ["new", None]})
    original = TabularFeaturizer(["x", "count", "kind"])
    original.fit_transform(train)
    metadata = original.checkpoint_metadata()
    frozen = featurizer_from_checkpoint(metadata, expected_feature_columns=["x", "count", "kind"])

    assert isinstance(frozen, FrozenTabularFeaturizer)
    np.testing.assert_allclose(frozen.fit_transform(probe), original.transform(probe), atol=1e-6)
    assert frozen.checkpoint_metadata() == metadata


def test_checkpoint_featurizer_contract_rejects_mismatch_and_unknown_kind() -> None:
    base = {
        "kind": "tabular_frozen_v1",
        "feature_columns": ["x"],
        "numeric_columns": ["x"],
        "numeric_imputer_statistics": [0.0],
        "numeric_means": [0.0],
        "numeric_scales": [1.0],
        "categorical_columns": [],
        "categorical_imputer_statistics": [],
        "categorical_categories": [],
        "output_dim": 1,
    }
    with pytest.raises(ValueError, match="feature columns"):
        featurizer_from_checkpoint(base, expected_feature_columns=["other"])
    with pytest.raises(ValueError, match="unsupported checkpoint featurizer"):
        featurizer_from_checkpoint(
            {**base, "kind": "external_pickle"}, expected_feature_columns=["x"]
        )
    morgan = featurizer_from_checkpoint(
        {
            "kind": "morgan_v1",
            "feature_columns": ["smiles"],
            "smiles_column": "smiles",
            "radius": 3,
            "bits": 128,
        },
        expected_feature_columns=["smiles"],
    )
    assert isinstance(morgan, MorganFeaturizer)
    assert morgan.checkpoint_metadata()["output_dim"] == 128


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"numeric_scales": []}, "numeric feature contract"),
        ({"numeric_scales": [0.0]}, "scales must be finite and positive"),
        ({"numeric_means": [float("nan")]}, "non-finite values"),
        (
            {
                "categorical_columns": ["kind"],
                "categorical_imputer_statistics": [],
                "categorical_categories": [[{"type": "str", "value": "a"}]],
            },
            "categorical feature contract",
        ),
    ],
)
def test_frozen_tabular_rejects_invalid_numeric_and_categorical_contracts(
    updates: dict[str, object], message: str
) -> None:
    metadata: dict[str, object] = {
        "kind": "tabular_frozen_v1",
        "feature_columns": ["x"],
        "numeric_columns": ["x"],
        "numeric_imputer_statistics": [0.0],
        "numeric_means": [0.0],
        "numeric_scales": [1.0],
        "categorical_columns": [],
        "categorical_imputer_statistics": [],
        "categorical_categories": [],
        "output_dim": 1,
    }
    metadata.update(updates)
    with pytest.raises(ValueError, match=message):
        FrozenTabularFeaturizer(metadata)


def test_frozen_tabular_validates_scalar_and_output_contracts() -> None:
    metadata = {
        "kind": "tabular_frozen_v1",
        "feature_columns": ["flag", "rank", "ratio", "kind"],
        "numeric_columns": [],
        "numeric_imputer_statistics": [],
        "numeric_means": [],
        "numeric_scales": [],
        "categorical_columns": ["flag", "rank", "ratio", "kind"],
        "categorical_imputer_statistics": [
            {"type": "bool", "value": True},
            {"type": "int", "value": 2},
            {"type": "float", "value": 1.5},
            {"type": "str", "value": "missing"},
        ],
        "categorical_categories": [
            [{"type": "bool", "value": True}],
            [{"type": "int", "value": 2}],
            [{"type": "float", "value": 1.5}],
            [{"type": "str", "value": "missing"}],
        ],
        "output_dim": 4,
    }
    frozen = FrozenTabularFeaturizer(metadata)
    frame = pd.DataFrame({"flag": [None], "rank": [None], "ratio": [None], "kind": [None]})
    np.testing.assert_array_equal(frozen.transform(frame), np.ones((1, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="columns are missing"):
        frozen.transform(frame.drop(columns="kind"))
    frozen.metadata["output_dim"] = 5
    with pytest.raises(ValueError, match="output dimension"):
        frozen.transform(frame)

    invalid_scalar = {
        **metadata,
        "categorical_imputer_statistics": [{"type": "bool", "value": 1}] * 4,
    }
    with pytest.raises(ValueError, match="boolean scalar"):
        FrozenTabularFeaturizer(invalid_scalar)
