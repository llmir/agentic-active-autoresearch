import numpy as np
import pandas as pd
import pytest

from agentic_al.config import AppConfig
from agentic_al.data import load_dataset
from agentic_al.registry import (
    ACQUISITION_FUNCTIONS,
    DATASET_LOADERS,
    register_acquisition,
    register_dataset,
)


def test_register_custom_acquisition() -> None:
    def custom(context, config):
        del config
        return np.ones(len(context.pool))

    register_acquisition("unit_custom", custom, replace=True)
    assert ACQUISITION_FUNCTIONS["unit_custom"] is custom


def test_register_custom_dataset_loader() -> None:
    def loader(config):
        assert config.dataset.parameters == {"source": "unit"}
        return pd.DataFrame({"sample_id": ["a", "b"], "x": [1, 2], "target": [3, 4]})

    register_dataset("unit-rows", loader, replace=True)
    config = AppConfig.model_validate(
        {
            "dataset": {
                "kind": "unit-rows",
                "parameters": {"source": "unit"},
                "feature_columns": ["x"],
            },
            "model": {"require_accelerator": False},
        }
    )
    assert config.dataset.kind == "unit_rows"
    assert load_dataset(config).frame["sample_id"].tolist() == ["a", "b"]
    assert "unit_rows" in DATASET_LOADERS


def test_dataset_loader_must_return_dataframe() -> None:
    register_dataset("unit_bad", lambda config: [], replace=True)
    config = AppConfig.model_validate({"dataset": {"kind": "unit_bad"}})
    with pytest.raises(TypeError, match="pandas DataFrame"):
        load_dataset(config)


def test_dataset_loader_frame_is_not_mutated() -> None:
    source = pd.DataFrame({"x": [1, 2], "target": [3, 4]})
    register_dataset("unit_shared", lambda config: source, replace=True)
    config = AppConfig.model_validate(
        {"dataset": {"kind": "unit_shared"}, "model": {"require_accelerator": False}}
    )
    bundle = load_dataset(config)
    assert "sample_id" in bundle.frame
    assert "sample_id" not in source
