import pandas as pd
import pytest

from agentic_al.oracle import CallableOracle, TableOracle, validate_observations


def test_table_and_callable_oracles() -> None:
    frame = pd.DataFrame({"id": ["a", "b"], "target": [1.0, 2.0]})
    assert TableOracle().observe(frame, id_column="id", target_column="target") == {
        "a": 1.0,
        "b": 2.0,
    }
    oracle = CallableOracle(
        lambda rows, id_column, target_column: {"a": rows[target_column].iloc[0]}
    )
    assert oracle.observe(frame.iloc[:1], id_column="id", target_column="target") == {"a": 1.0}


def test_observation_validation_rejects_id_and_value_errors() -> None:
    assert validate_observations({1: 2.0}, ["1"]) == {"1": 2.0}
    with pytest.raises(ValueError, match="ID mismatch"):
        validate_observations({"extra": 1}, ["expected"])
    with pytest.raises(ValueError, match="missing observations"):
        validate_observations({"a": None}, ["a"])
    assert validate_observations({"a": "2.5"}, ["a"], task="regression") == {"a": 2.5}
    for invalid in ("not-numeric", float("inf")):
        with pytest.raises(ValueError, match="finite numeric"):
            validate_observations({"a": invalid}, ["a"], task="regression")
