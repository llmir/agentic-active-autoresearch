"""Oracle contracts for simulation and user-supplied experimental systems."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import pandas as pd


class Oracle(Protocol):
    def observe(
        self, rows: pd.DataFrame, *, id_column: str, target_column: str
    ) -> Mapping[str, Any]:
        """Return an id-to-observation mapping; implementations should be idempotent by ID."""


class TableOracle:
    """Reveal labels already present in a benchmark table."""

    def observe(
        self, rows: pd.DataFrame, *, id_column: str, target_column: str
    ) -> Mapping[str, Any]:
        return dict(zip(rows[id_column].astype(str), rows[target_column], strict=True))


class CallableOracle:
    """Wrap a trusted Python callback as an Oracle implementation."""

    def __init__(
        self,
        callback: Callable[[pd.DataFrame, str, str], Mapping[str, Any]],
    ) -> None:
        self.callback = callback

    def observe(
        self, rows: pd.DataFrame, *, id_column: str, target_column: str
    ) -> Mapping[str, Any]:
        return self.callback(rows.copy(), id_column, target_column)


def validate_observations(
    observations: Mapping[str, Any], expected_ids: list[str], *, task: str | None = None
) -> dict[str, Any]:
    normalized = {str(key): value for key, value in observations.items()}
    missing = sorted(set(expected_ids).difference(normalized))
    extra = sorted(set(normalized).difference(expected_ids))
    if missing or extra:
        raise ValueError(f"oracle result ID mismatch: missing={missing[:5]} extra={extra[:5]}")
    if any(pd.isna(value) for value in normalized.values()):
        raise ValueError("oracle returned missing observations")
    if task == "regression":
        try:
            numeric = {key: float(value) for key, value in normalized.items()}
        except (TypeError, ValueError) as error:
            raise ValueError(
                "regression oracle observations must be finite numeric values"
            ) from error
        if not all(math.isfinite(value) for value in numeric.values()):
            raise ValueError("regression oracle observations must be finite numeric values")
        return numeric
    return normalized
