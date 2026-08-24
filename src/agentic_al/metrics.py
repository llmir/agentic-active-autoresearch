"""Task metrics and uncertainty diagnostics without hidden global state."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from .types import Prediction


def evaluate(target: np.ndarray, prediction: Prediction, task: str) -> dict[str, float]:
    prediction = validate_prediction(prediction, len(target), task)
    if task == "classification":
        if prediction.labels is None:
            raise ValueError("classification predictions do not contain labels")
        result = {
            "accuracy": float(accuracy_score(target, prediction.labels)),
            "balanced_accuracy": float(balanced_accuracy_score(target, prediction.labels)),
        }
        if prediction.probabilities is not None and prediction.classes is not None:
            result["log_loss"] = float(
                log_loss(target, prediction.probabilities, labels=prediction.classes)
            )
        errors = (np.asarray(prediction.labels) != np.asarray(target)).astype(float)
    else:
        result = {
            "mae": float(mean_absolute_error(target, prediction.mean)),
            "rmse": float(math.sqrt(mean_squared_error(target, prediction.mean))),
            "r2": float(r2_score(target, prediction.mean)),
        }
        errors = np.abs(np.asarray(target, dtype=float) - prediction.mean)
    result["uncertainty_error_spearman"] = _rank_correlation(prediction.uncertainty, errors)
    return {key: _finite_float(value) for key, value in result.items()}


def prediction_summary(prediction: Prediction) -> dict[str, float]:
    return {
        "prediction_mean": _finite_float(np.mean(prediction.mean)),
        "prediction_std": _finite_float(np.std(prediction.mean)),
        "uncertainty_mean": _finite_float(np.mean(prediction.uncertainty)),
        "uncertainty_std": _finite_float(np.std(prediction.uncertainty)),
        "uncertainty_max": _finite_float(np.max(prediction.uncertainty)),
    }


def validate_prediction(prediction: Prediction, expected_rows: int, task: str) -> Prediction:
    """Fail fast on malformed or non-finite output from any surrogate backend."""

    mean = np.asarray(prediction.mean)
    uncertainty = np.asarray(prediction.uncertainty)
    if mean.shape != (expected_rows,) or uncertainty.shape != (expected_rows,):
        raise ValueError(
            "prediction mean and uncertainty must be one-dimensional and aligned with input rows"
        )
    if (
        not np.isfinite(mean.astype(float)).all()
        or not np.isfinite(uncertainty.astype(float)).all()
    ):
        raise ValueError("prediction mean and uncertainty must be finite")
    if np.any(uncertainty.astype(float) < 0.0):
        raise ValueError("prediction uncertainty must be non-negative")
    if task != "classification":
        return prediction
    if prediction.labels is None or np.asarray(prediction.labels).shape != (expected_rows,):
        raise ValueError("classification predictions must contain one label per input row")
    if prediction.probabilities is None and prediction.classes is None:
        return prediction
    if prediction.probabilities is None or prediction.classes is None:
        raise ValueError("classification probabilities and classes must be provided together")
    probabilities = np.asarray(prediction.probabilities, dtype=float)
    classes = np.asarray(prediction.classes)
    if probabilities.ndim != 2 or probabilities.shape[0] != expected_rows:
        raise ValueError("classification probability rows must align with input rows")
    if probabilities.shape[1] < 2 or classes.shape != (probabilities.shape[1],):
        raise ValueError("classification probability columns must align with at least two classes")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise ValueError("classification probabilities must be finite and non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("classification probability rows must sum to one")
    return prediction


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return 0.0
    left_rank = pd.Series(np.asarray(left, dtype=float)).rank(method="average").to_numpy()
    right_rank = pd.Series(np.asarray(right, dtype=float)).rank(method="average").to_numpy()
    if np.std(left_rank) <= 1e-12 or np.std(right_rank) <= 1e-12:
        return 0.0
    return _finite_float(np.corrcoef(left_rank, right_rank)[0, 1])


def _finite_float(value: float | np.floating) -> float:
    number = float(value)
    return number if math.isfinite(number) else 0.0
