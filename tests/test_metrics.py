import numpy as np
import pytest

from agentic_al.metrics import evaluate, prediction_summary, validate_prediction
from agentic_al.types import Prediction


def test_regression_metrics_and_summary_are_finite() -> None:
    prediction = Prediction(
        mean=np.asarray([1.0, 2.0, 4.0]), uncertainty=np.asarray([0.1, 0.2, 0.9])
    )
    metrics = evaluate(np.asarray([1.0, 3.0, 4.0]), prediction, "regression")
    summary = prediction_summary(prediction)
    assert metrics["mae"] == pytest.approx(1 / 3)
    assert all(np.isfinite(value) for value in {**metrics, **summary}.values())


def test_classification_metrics_include_log_loss() -> None:
    prediction = Prediction(
        mean=np.asarray([0.1, 0.8, 0.7]),
        uncertainty=np.asarray([0.2, 0.3, 0.4]),
        labels=np.asarray([0, 1, 1]),
        probabilities=np.asarray([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]]),
        classes=np.asarray([0, 1]),
    )
    metrics = evaluate(np.asarray([0, 1, 0]), prediction, "classification")
    assert set(metrics) >= {"accuracy", "balanced_accuracy", "log_loss"}


def test_classification_requires_labels_and_constant_rank_is_safe() -> None:
    with pytest.raises(ValueError, match="one label"):
        evaluate(
            np.asarray([0, 1]),
            Prediction(mean=np.zeros(2), uncertainty=np.zeros(2)),
            "classification",
        )
    with pytest.raises(ValueError, match="finite"):
        evaluate(
            np.asarray([1.0, 2.0]),
            Prediction(mean=np.asarray([1.0, 2.0]), uncertainty=np.asarray([np.nan, np.nan])),
            "regression",
        )
    metrics = evaluate(
        np.asarray([1.0, 2.0]),
        Prediction(mean=np.asarray([1.0, 2.0]), uncertainty=np.asarray([0.0, 0.0])),
        "regression",
    )
    assert metrics["uncertainty_error_spearman"] == 0.0


@pytest.mark.parametrize(
    ("prediction", "task", "message"),
    [
        (Prediction(mean=np.ones(2), uncertainty=np.ones(1)), "regression", "aligned"),
        (
            Prediction(mean=np.asarray([np.nan]), uncertainty=np.zeros(1)),
            "regression",
            "finite",
        ),
        (Prediction(mean=np.ones(1), uncertainty=np.asarray([-1.0])), "regression", "non-negative"),
        (
            Prediction(mean=np.ones(2), uncertainty=np.zeros(2), labels=np.asarray([0])),
            "classification",
            "one label",
        ),
        (
            Prediction(
                mean=np.ones(2),
                uncertainty=np.zeros(2),
                labels=np.asarray([0, 1]),
                probabilities=np.asarray([[0.8, 0.8], [0.2, 0.8]]),
                classes=np.asarray([0, 1]),
            ),
            "classification",
            "sum to one",
        ),
    ],
)
def test_prediction_contract_rejects_malformed_backend_output(
    prediction: Prediction, task: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_prediction(prediction, len(prediction.mean), task)
