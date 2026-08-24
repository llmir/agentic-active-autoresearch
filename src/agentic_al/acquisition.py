"""Composable acquisition functions and deterministic batch selection."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .config import AcquisitionConfig
from .registry import ACQUISITION_FUNCTIONS, register_acquisition
from .types import AcquisitionContext


def normalize(values: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if array.size == 0:
        return array
    low = float(array.min())
    high = float(array.max())
    if high - low <= 1e-12:
        return np.zeros_like(array)
    return (array - low) / (high - low)


def random_component(context: AcquisitionContext, _: AcquisitionConfig) -> np.ndarray:
    seed = int(context.seed + 1_000_003 * context.round_index)
    return np.random.default_rng(seed).random(len(context.pool))


def uncertainty_component(context: AcquisitionContext, _: AcquisitionConfig) -> np.ndarray:
    return normalize(context.prediction.uncertainty)


def diversity_component(context: AcquisitionContext, config: AcquisitionConfig) -> np.ndarray:
    # Accumulate squared distances in float64. Feature matrices are commonly float32, whose dot
    # products can overflow for otherwise finite inputs on some NumPy/BLAS combinations.
    pool = np.asarray(context.pool_features, dtype=np.float64)
    labeled = np.asarray(context.labeled_features, dtype=np.float64)
    if len(pool) == 0:
        return np.zeros(0, dtype=float)
    if len(labeled) == 0:
        return np.ones(len(pool), dtype=float)
    if not np.isfinite(pool).all() or not np.isfinite(labeled).all():
        raise ValueError("diversity features must contain only finite values")
    scale = max(float(np.max(np.abs(pool))), float(np.max(np.abs(labeled))), 1.0)
    pool = pool / scale
    labeled = labeled / scale
    best = np.full(len(pool), np.inf, dtype=np.float64)
    chunk_size = config.diversity_chunk_size
    for start in range(0, len(pool), chunk_size):
        stop = min(start + chunk_size, len(pool))
        chunk = pool[start:stop]
        chunk_norm = np.sum(chunk * chunk, axis=1, keepdims=True)
        reference_norm = np.sum(labeled * labeled, axis=1, keepdims=True).T
        # NumPy 2.2 linked to Apple Accelerate can emit spurious divide/overflow warnings for a
        # finite matmul. The operands are scaled above, and the result is checked immediately.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            squared = np.maximum(chunk_norm + reference_norm - 2.0 * (chunk @ labeled.T), 0.0)
        if not np.isfinite(squared).all():
            raise ValueError("diversity distance computation produced non-finite values")
        best[start:stop] = np.sqrt(squared).min(axis=1)
    return normalize(best)


def representativeness_component(context: AcquisitionContext, _: AcquisitionConfig) -> np.ndarray:
    """Prefer points near the robust center of the current candidate pool."""

    features = np.asarray(context.pool_features, dtype=np.float64)
    if len(features) == 0:
        return np.zeros(0, dtype=float)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("representativeness features must be a finite two-dimensional matrix")
    scale = np.maximum(np.nanstd(features, axis=0), 1e-12)
    center = np.nanmedian(features, axis=0)
    distance = np.sqrt(np.mean(((features - center) / scale) ** 2, axis=1))
    return 1.0 - normalize(distance)


def group_coverage_component(context: AcquisitionContext, _: AcquisitionConfig) -> np.ndarray:
    """Prefer pool rows from groups that are rare or absent in the labeled set."""

    column = context.group_column
    if not column:
        return np.zeros(len(context.pool), dtype=float)
    if column not in context.pool.columns or column not in context.labeled.columns:
        raise ValueError(f"group coverage column is unavailable: {column}")
    labeled_counts = context.labeled[column].astype(str).value_counts().to_dict()
    raw = np.asarray(
        [
            1.0 / (1.0 + float(labeled_counts.get(value, 0)))
            for value in context.pool[column].astype(str)
        ],
        dtype=float,
    )
    return normalize(raw)


def target_component(context: AcquisitionContext, config: AcquisitionConfig) -> np.ndarray:
    prediction = context.prediction
    if config.goal == "target_class":
        if prediction.probabilities is None or prediction.classes is None:
            return np.zeros(len(context.pool), dtype=float)
        class_index = next(
            (
                index
                for index, value in enumerate(prediction.classes)
                if str(value) == str(config.target_class)
            ),
            None,
        )
        if class_index is None:
            return np.zeros(len(context.pool), dtype=float)
        return normalize(prediction.probabilities[:, class_index])
    values = np.asarray(prediction.mean, dtype=float)
    if config.goal == "maximize":
        return normalize(values)
    if config.goal == "target_range" and config.target_range is not None:
        low, high = config.target_range
        center = 0.5 * (low + high)
        half_width = max(1e-9, 0.5 * (high - low))
        return normalize(np.exp(-0.5 * ((values - center) / half_width) ** 2))
    return 1.0 - normalize(values)


def component_scores(
    context: AcquisitionContext,
    config: AcquisitionConfig,
    weights: dict[str, float],
) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}
    for name, weight in weights.items():
        normalized_name = name.strip().lower().replace("-", "_")
        if weight <= 0:
            continue
        function = ACQUISITION_FUNCTIONS.get(normalized_name)
        if function is None:
            raise ValueError(
                f"unknown acquisition component '{name}'. "
                f"Registered components: {sorted(ACQUISITION_FUNCTIONS)}"
            )
        value = np.asarray(function(context, config), dtype=float)
        if value.shape != (len(context.pool),):
            raise ValueError(
                f"acquisition component '{name}' returned shape {value.shape}; "
                f"expected {(len(context.pool),)}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"acquisition component '{name}' returned non-finite scores")
        scores[normalized_name] = normalize(value)
    return scores


def aggregate_scores(
    context: AcquisitionContext,
    config: AcquisitionConfig,
    weights: dict[str, float],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    scores = component_scores(context, config, weights)
    final = np.zeros(len(context.pool), dtype=float)
    for name, score in scores.items():
        final += float(weights.get(name, 0.0)) * score
    if context.cost_column:
        final -= config.cost_penalty * normalize(
            context.pool[context.cost_column].to_numpy(dtype=float)
        )
    if context.risk_column:
        final -= config.risk_penalty * normalize(
            context.pool[context.risk_column].to_numpy(dtype=float)
        )
    return np.nan_to_num(final, nan=-math.inf), scores


def select_top_k(
    final_scores: np.ndarray,
    row_ids: list[Any],
    batch_size: int,
) -> list[int]:
    if len(final_scores) != len(row_ids):
        raise ValueError("score and row-id lengths differ")
    order = sorted(
        range(len(row_ids)),
        key=lambda index: (-float(final_scores[index]), str(row_ids[index])),
    )
    return order[: min(batch_size, len(order))]


register_acquisition("random", random_component)
register_acquisition("uncertainty", uncertainty_component)
register_acquisition("diversity", diversity_component)
register_acquisition("representativeness", representativeness_component)
register_acquisition("group_coverage", group_coverage_component)
register_acquisition("scaffold_coverage", group_coverage_component)
register_acquisition("target", target_component)
