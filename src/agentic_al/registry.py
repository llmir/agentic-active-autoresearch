"""Small in-process registries for trusted user extensions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .types import AcquisitionContext

ModelFactory = Callable[..., Any]
AcquisitionFunction = Callable[[AcquisitionContext, Any], np.ndarray]
DatasetLoader = Callable[[Any], Any]
TrainingCandidateGeneratorFactory = Callable[[Any], Any]

MODEL_FACTORIES: dict[str, ModelFactory] = {}
ACQUISITION_FUNCTIONS: dict[str, AcquisitionFunction] = {}
DATASET_LOADERS: dict[str, DatasetLoader] = {}
TRAINING_CANDIDATE_GENERATORS: dict[str, TrainingCandidateGeneratorFactory] = {}


def register_model(name: str, factory: ModelFactory, *, replace: bool = False) -> None:
    _register(MODEL_FACTORIES, name, factory, replace=replace)


def register_acquisition(
    name: str, function: AcquisitionFunction, *, replace: bool = False
) -> None:
    _register(ACQUISITION_FUNCTIONS, name, function, replace=replace)


def register_dataset(name: str, loader: DatasetLoader, *, replace: bool = False) -> None:
    """Register a trusted loader that returns a pandas DataFrame for one dataset kind."""

    _register(DATASET_LOADERS, name, loader, replace=replace)


def register_training_candidate_generator(
    name: str, factory: TrainingCandidateGeneratorFactory, *, replace: bool = False
) -> None:
    """Register a trusted factory for an inner-loop candidate generator."""

    _register(TRAINING_CANDIDATE_GENERATORS, name, factory, replace=replace)


def _register(registry: dict[str, Any], name: str, value: Any, *, replace: bool) -> None:
    normalized = name.strip().lower().replace("-", "_")
    if not normalized or not normalized.replace("_", "a").isalnum():
        raise ValueError("registry names must contain only letters, digits, and underscores")
    if normalized in registry and not replace:
        raise ValueError(f"component already registered: {normalized}")
    registry[normalized] = value
