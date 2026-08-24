"""Small, stable data contracts shared by Agentic Active AutoResearch components."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class DatasetBundle:
    frame: pd.DataFrame
    id_column: str
    target_column: str
    feature_columns: list[str]
    task: str
    group_column: str | None = None
    cost_column: str | None = None
    risk_column: str | None = None


@dataclass(slots=True)
class Prediction:
    mean: np.ndarray
    uncertainty: np.ndarray
    labels: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    classes: np.ndarray | None = None


@dataclass(slots=True)
class AcquisitionContext:
    pool: pd.DataFrame
    labeled: pd.DataFrame
    pool_features: np.ndarray
    labeled_features: np.ndarray
    prediction: Prediction
    id_column: str
    round_index: int
    seed: int
    task: str
    group_column: str | None = None
    cost_column: str | None = None
    risk_column: str | None = None


@dataclass(slots=True)
class PolicyContext:
    round_index: int
    total_rounds: int
    labeled_size: int
    pool_size: int
    batch_size: int
    remaining_budget: int
    task: str
    metrics: dict[str, float]
    previous_metrics: dict[str, float] = field(default_factory=dict)
    prediction_summary: dict[str, float] = field(default_factory=dict)
    previous_weights: dict[str, float] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        """Return aggregate-only context safe to send to an external policy service."""

        return {
            "round_index": self.round_index,
            "total_rounds": self.total_rounds,
            "labeled_size": self.labeled_size,
            "pool_size": self.pool_size,
            "batch_size": self.batch_size,
            "remaining_budget": self.remaining_budget,
            "task": self.task,
            "metrics": self.metrics,
            "previous_metrics": self.previous_metrics,
            "prediction_summary": self.prediction_summary,
            "previous_weights": self.previous_weights,
        }


@dataclass(slots=True)
class StrategyProposal:
    weights: dict[str, float]
    rationale: str
    source: str
    fallback_used: bool = False
    fallback_reason: str = ""


@dataclass(slots=True)
class GuardrailResult:
    proposal: StrategyProposal
    original_weights: dict[str, float]
    repairs: list[str]


@dataclass(slots=True)
class TrainingCandidate:
    name: str
    parameters: dict[str, float]
    rationale: str
    source: str
    paradigm: str = "retrain_from_scratch"
    repairs: list[str] = field(default_factory=list)
    fallback_used: bool = False
    fallback_reason: str = ""


@dataclass(frozen=True, slots=True)
class CheckpointReference:
    """Validated local checkpoint plus the small public lineage policy may inspect."""

    source: str
    weights_path: Path
    manifest_path: Path
    origin_round: int
    origin_phase: str
    selection_metric: str
    selection_mode: str
    selection_value: float | None
    weights_sha256: str
    manifest_sha256: str
    lineage_training_fits: int

    def to_public_dict(self) -> dict[str, Any]:
        """Exclude paths and weights while exposing enough evidence for policy decisions."""

        return {
            "source": self.source,
            "origin_round": self.origin_round,
            "origin_phase": self.origin_phase,
            "selection_metric": self.selection_metric,
            "selection_mode": self.selection_mode,
            "selection_value": self.selection_value,
            "weights_sha256": self.weights_sha256,
            "manifest_sha256": self.manifest_sha256,
            "lineage_training_fits": self.lineage_training_fits,
        }


@dataclass(slots=True)
class CheckpointCatalog:
    """Typed checkpoint choices resolved locally before candidate generation."""

    sources: dict[str, CheckpointReference] = field(default_factory=dict)

    def public_sources(self) -> dict[str, dict[str, Any]]:
        return {name: reference.to_public_dict() for name, reference in self.sources.items()}


@dataclass(slots=True)
class TrainingCandidateContext:
    round_index: int
    inner_step_index: int
    phase: str
    task: str
    labeled_size: int
    trial_budget: int
    remaining_trials: int
    selection_metric: str
    selection_mode: str
    base_parameters: dict[str, float]
    tunable_parameters: dict[str, tuple[float, float]]
    supported_paradigms: list[str]
    checkpoint_sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    completed_trials: list[dict[str, Any]] = field(default_factory=list)
    previous_reflection: dict[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        """Return aggregate, secret-free state for a training-candidate policy."""

        return {
            "round_index": self.round_index,
            "inner_step_index": self.inner_step_index,
            "phase": self.phase,
            "task": self.task,
            "labeled_size": self.labeled_size,
            "trial_budget": self.trial_budget,
            "remaining_trials": self.remaining_trials,
            "selection_metric": self.selection_metric,
            "selection_mode": self.selection_mode,
            "base_parameters": self.base_parameters,
            "tunable_parameters": self.tunable_parameters,
            "supported_paradigms": self.supported_paradigms,
            "checkpoint_sources": self.checkpoint_sources,
            "completed_trials": self.completed_trials,
            "previous_reflection": self.previous_reflection,
        }


@dataclass(slots=True)
class InnerLoopResult:
    model: Any
    best_candidate: TrainingCandidate
    trial_results: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass(slots=True)
class RunResult:
    run_dir: Path
    summary_path: Path
    report_path: Path
    final_metrics: dict[str, Any]
    completed_rounds: int
