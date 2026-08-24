import numpy as np
import pandas as pd
import pytest

from agentic_al.acquisition import (
    aggregate_scores,
    component_scores,
    diversity_component,
    group_coverage_component,
    normalize,
    representativeness_component,
    select_top_k,
    target_component,
)
from agentic_al.config import AcquisitionConfig
from agentic_al.registry import register_acquisition
from agentic_al.types import AcquisitionContext, Prediction


def _context() -> AcquisitionContext:
    pool = pd.DataFrame({"id": ["b", "a", "c"], "cost": [1.0, 10.0, 2.0]})
    labeled = pd.DataFrame({"id": ["z"]})
    return AcquisitionContext(
        pool=pool,
        labeled=labeled,
        pool_features=np.asarray([[0.0], [1.0], [3.0]]),
        labeled_features=np.asarray([[0.0]]),
        prediction=Prediction(
            mean=np.asarray([3.0, 2.0, 1.0]), uncertainty=np.asarray([0.1, 0.5, 0.2])
        ),
        id_column="id",
        round_index=0,
        seed=42,
        task="regression",
        cost_column="cost",
    )


def test_normalize_constant_is_zero() -> None:
    assert np.array_equal(normalize(np.ones(4)), np.zeros(4))


def test_normalize_empty_and_nonfinite_values() -> None:
    assert normalize(np.asarray([])).size == 0
    assert np.isfinite(normalize(np.asarray([np.nan, np.inf, -np.inf, 1.0]))).all()


def test_hybrid_scores_are_finite_and_cost_aware() -> None:
    config = AcquisitionConfig(weights={"uncertainty": 1.0}, goal="minimize", cost_penalty=0.5)
    score, components = aggregate_scores(_context(), config, config.weights)
    assert np.isfinite(score).all()
    assert set(components) == {"uncertainty"}
    assert score[1] < 1.0


def test_tie_break_uses_stable_string_id() -> None:
    assert select_top_k(np.asarray([1.0, 1.0, 1.0]), ["b", "a", "c"], 2) == [1, 0]


def test_target_minimize_prefers_low_prediction() -> None:
    config = AcquisitionConfig(weights={"target": 1.0}, goal="minimize")
    score, _ = aggregate_scores(_context(), config, config.weights)
    assert int(np.argmax(score)) == 2


def test_target_modes_cover_range_maximize_and_class() -> None:
    context = _context()
    assert int(np.argmax(target_component(context, AcquisitionConfig(goal="maximize")))) == 0
    ranged = target_component(
        context,
        AcquisitionConfig(goal="target_range", target_range=(1.5, 2.5)),
    )
    assert int(np.argmax(ranged)) == 1

    context.prediction.probabilities = np.asarray([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    context.prediction.classes = np.asarray(["no", "yes"])
    classified = target_component(
        context,
        AcquisitionConfig(goal="target_class", target_class="yes"),
    )
    assert int(np.argmax(classified)) == 1
    missing = target_component(
        context,
        AcquisitionConfig(goal="target_class", target_class="unknown"),
    )
    assert np.array_equal(missing, np.zeros(3))


def test_diversity_handles_empty_references_and_real_distances() -> None:
    context = _context()
    context.labeled_features = np.empty((0, 1))
    assert np.array_equal(diversity_component(context, AcquisitionConfig()), np.ones(3))
    context.labeled_features = np.asarray([[0.0]])
    scores = diversity_component(context, AcquisitionConfig(diversity_chunk_size=32))
    assert int(np.argmax(scores)) == 2


def test_diversity_uses_stable_precision_for_large_float32_features() -> None:
    context = _context()
    largest = np.finfo(np.float32).max
    context.pool_features = np.asarray([[largest], [-largest], [0.0]], dtype=np.float32)
    context.labeled_features = np.asarray([[largest]], dtype=np.float32)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        scores = diversity_component(context, AcquisitionConfig(diversity_chunk_size=32))
    assert np.isfinite(scores).all()
    assert int(np.argmax(scores)) == 1


def test_diversity_rejects_nonfinite_plugin_features() -> None:
    context = _context()
    context.pool_features[0, 0] = np.inf
    with pytest.raises(ValueError, match="diversity features must contain only finite values"):
        diversity_component(context, AcquisitionConfig())


def test_representativeness_prefers_the_robust_pool_center() -> None:
    context = _context()
    context.pool_features = np.asarray([[0.0], [1.0], [20.0]])
    scores = representativeness_component(context, AcquisitionConfig())
    assert np.isfinite(scores).all()
    assert int(np.argmax(scores)) == 1


def test_group_and_scaffold_coverage_prefer_unseen_groups() -> None:
    context = _context()
    context.group_column = "series"
    context.labeled["series"] = ["seen"]
    context.pool["series"] = ["seen", "new", "seen"]
    scores = group_coverage_component(context, AcquisitionConfig())
    assert int(np.argmax(scores)) == 1
    registered = component_scores(
        context,
        AcquisitionConfig(),
        {"group_coverage": 0.5, "scaffold_coverage": 0.5},
    )
    assert set(registered) == {"group_coverage", "scaffold_coverage"}


def test_group_coverage_without_a_declared_group_is_neutral() -> None:
    assert np.array_equal(
        group_coverage_component(_context(), AcquisitionConfig()),
        np.zeros(3),
    )


def test_component_contract_and_selection_validation() -> None:
    with pytest.raises(ValueError, match="unknown acquisition"):
        component_scores(_context(), AcquisitionConfig(), {"does_not_exist": 1.0})

    register_acquisition("bad_shape", lambda context, config: np.ones(1), replace=True)
    with pytest.raises(ValueError, match="returned shape"):
        component_scores(_context(), AcquisitionConfig(), {"bad_shape": 1.0})
    register_acquisition(
        "bad_finite",
        lambda context, config: np.full(len(context.pool), np.nan),
        replace=True,
    )
    with pytest.raises(ValueError, match="non-finite"):
        component_scores(_context(), AcquisitionConfig(), {"bad_finite": 1.0})
    with pytest.raises(ValueError, match="lengths differ"):
        select_top_k(np.ones(2), ["only-one"], 1)


def test_risk_penalty_is_applied() -> None:
    context = _context()
    context.pool["risk"] = [0.0, 1.0, 0.0]
    context.risk_column = "risk"
    config = AcquisitionConfig(weights={"random": 1.0}, risk_penalty=1.0)
    scores, _ = aggregate_scores(context, config, config.weights)
    assert scores[1] < 0.0
