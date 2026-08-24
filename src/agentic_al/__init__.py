"""Public API for Agentic Active AutoResearch (stable import: agentic_al)."""

from .config import AppConfig, load_config
from .engine import RunResult, run_experiment
from .oracle import CallableOracle, TableOracle
from .registry import (
    register_acquisition,
    register_dataset,
    register_model,
    register_training_candidate_generator,
)

__all__ = [
    "AppConfig",
    "CallableOracle",
    "RunResult",
    "TableOracle",
    "load_config",
    "register_acquisition",
    "register_dataset",
    "register_model",
    "register_training_candidate_generator",
    "run_experiment",
]

__version__ = "0.1.0"
