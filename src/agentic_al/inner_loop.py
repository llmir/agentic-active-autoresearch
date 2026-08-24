"""Sequential, auditable inner-loop search over bounded training candidates."""

from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .artifacts import (
    atomic_write_csv,
    atomic_write_json,
    dataframe_hash,
    safe_error,
    software_fingerprint,
    stable_hash,
)
from .checkpoints import paradigm_checkpoint_source, save_selected_checkpoint
from .config import AppConfig, ModelConfig
from .metrics import evaluate
from .models import build_model
from .policy import request_policy_json
from .registry import TRAINING_CANDIDATE_GENERATORS, register_training_candidate_generator
from .types import (
    CheckpointCatalog,
    CheckpointReference,
    InnerLoopResult,
    TrainingCandidate,
    TrainingCandidateContext,
)


class TrainingCandidateGenerator(Protocol):
    def propose(self, context: TrainingCandidateContext, count: int) -> list[TrainingCandidate]: ...


class RuleBasedTrainingCandidateGenerator:
    """Deterministic sequential proposals anchored to the configured control."""

    def propose(self, context: TrainingCandidateContext, count: int) -> list[TrainingCandidate]:
        completed_signatures = {
            _candidate_signature(
                str(item.get("paradigm", "retrain_from_scratch")),
                item.get("parameters", {}),
            )
            for item in context.completed_trials
        }
        options = _rule_based_options(context)
        unique = [
            candidate
            for candidate in options
            if _candidate_signature(candidate.paradigm, candidate.parameters)
            not in completed_signatures
        ]
        return unique[:count]


class OpenAICompatibleTrainingCandidateGenerator:
    """Aggregate-only external planner with deterministic local fallback."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.fallback = RuleBasedTrainingCandidateGenerator()

    def propose(self, context: TrainingCandidateContext, count: int) -> list[TrainingCandidate]:
        try:
            system_prompt = (
                "Propose bounded model-training candidates for an active-learning inner loop. "
                "Return one JSON object with key 'candidates', an array of at most "
                f"{count} objects. Each object may contain only 'name', 'parameters', "
                "'paradigm', and 'rationale'. Parameters must use only the supplied tunable "
                "parameter names and numeric bounds. The paradigm must be one of the supplied "
                "supported_paradigms. Do not change epochs, batch size, ensemble size, model "
                "architecture, data split, label budget, or trial budget. Do not request data, "
                "row IDs, labels, SMILES, paths, tools, code, or checkpoints."
            )
            parsed = request_policy_json(self.config.agent, system_prompt, context.to_public_dict())
            raw_candidates = parsed.get("candidates")
            if not isinstance(raw_candidates, list):
                raise ValueError("training policy response 'candidates' must be an array")
            candidates: list[TrainingCandidate] = []
            for index, raw in enumerate(raw_candidates[:count]):
                if not isinstance(raw, dict):
                    continue
                raw_parameters = raw.get("parameters", {})
                if not isinstance(raw_parameters, dict):
                    continue
                parameters: dict[str, float] = {}
                for name, value in raw_parameters.items():
                    if isinstance(value, bool) or not isinstance(value, int | float):
                        continue
                    parameters[str(name)] = float(value)
                candidates.append(
                    TrainingCandidate(
                        name=str(raw.get("name", f"agent_candidate_{index + 1}")),
                        parameters=parameters,
                        paradigm=str(raw.get("paradigm", "retrain_from_scratch")),
                        rationale=str(raw.get("rationale", "External bounded proposal."))[:500],
                        source="openai_compatible",
                    )
                )
            if not candidates:
                raise ValueError("training policy returned no usable candidates")
            return candidates
        except Exception as error:
            if not self.config.agent.fallback_to_rule_based:
                raise
            fallback = self.fallback.propose(context, count)
            return [
                replace(
                    candidate,
                    fallback_used=True,
                    fallback_reason=safe_error(error),
                    rationale=f"Provider unavailable or invalid; {candidate.rationale}",
                )
                for candidate in fallback
            ]


def run_inner_loop(
    config: AppConfig,
    *,
    labeled: pd.DataFrame,
    target: np.ndarray,
    task: str,
    feature_columns: list[str],
    seed: int,
    round_index: int,
    phase: str,
    artifact_dir: Path,
    checkpoint_catalog: CheckpointCatalog | None = None,
) -> InnerLoopResult:
    """Select one bounded training plan on an inner split, then refit on all labels."""

    inner = config.inner_loop
    checkpoint_catalog = checkpoint_catalog or CheckpointCatalog()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    generator_name = _resolved_generator_name(config)
    selection_metric, selection_mode = selection_contract(config, task)
    tunable = {
        name: bounds
        for name, bounds in inner.tunable_parameters.items()
        if name in config.model.parameters
        and isinstance(config.model.parameters[name], int | float)
        and not isinstance(config.model.parameters[name], bool)
    }
    public_config = {
        **inner.model_dump(mode="json"),
        "resolved_candidate_generator": generator_name,
        "resolved_selection_metric": selection_metric,
        "resolved_selection_mode": selection_mode,
        "effective_tunable_parameters": tunable,
        "effective_candidate_paradigms": _supported_paradigms(config, task, checkpoint_catalog),
        "checkpoint_sources": checkpoint_catalog.public_sources(),
        "outer_validation_used_for_selection": False,
    }
    atomic_write_json(artifact_dir / "inner_loop_config.json", public_config)

    skip_reason = _skip_reason(config, labeled, target, task, tunable, checkpoint_catalog)
    if skip_reason:
        return _fit_control_only(
            config,
            labeled=labeled,
            target=target,
            task=task,
            feature_columns=feature_columns,
            seed=seed,
            round_index=round_index,
            phase=phase,
            artifact_dir=artifact_dir,
            generator_name=generator_name,
            selection_metric=selection_metric,
            selection_mode=selection_mode,
            reason=skip_reason,
            checkpoint_catalog=checkpoint_catalog,
        )

    group_column = config.dataset.group_column
    groups = (
        labeled[group_column].astype(str).to_numpy()
        if group_column is not None and group_column in labeled.columns
        else None
    )
    try:
        train_positions, validation_positions = _inner_split_positions(
            target,
            task=task,
            fraction=inner.validation_fraction,
            seed=seed,
            groups=groups,
        )
    except ValueError as error:
        return _fit_control_only(
            config,
            labeled=labeled,
            target=target,
            task=task,
            feature_columns=feature_columns,
            seed=seed,
            round_index=round_index,
            phase=phase,
            artifact_dir=artifact_dir,
            generator_name=generator_name,
            selection_metric=selection_metric,
            selection_mode=selection_mode,
            reason=f"inner_split_unavailable:{safe_error(error)}",
            checkpoint_catalog=checkpoint_catalog,
        )
    inner_train = labeled.iloc[train_positions].reset_index(drop=True)
    inner_validation = labeled.iloc[validation_positions].reset_index(drop=True)
    train_target = np.asarray(target)[train_positions]
    validation_target = np.asarray(target)[validation_positions]
    atomic_write_json(
        artifact_dir / "split_summary.json",
        {
            "method": (
                "deterministic_labeled_only_group_holdout"
                if groups is not None
                else "deterministic_labeled_only_holdout"
            ),
            "train_size": len(inner_train),
            "validation_size": len(inner_validation),
            "stratified": task == "classification" and groups is None,
            "group_aware": groups is not None,
            "group_column": group_column if groups is not None else None,
            "train_group_count": (
                len(np.unique(groups[train_positions])) if groups is not None else None
            ),
            "validation_group_count": (
                len(np.unique(groups[validation_positions])) if groups is not None else None
            ),
            "outer_validation_used": False,
        },
    )

    generator = _build_generator(config, generator_name)
    trials: list[dict[str, Any]] = []
    step_index = 0
    previous_reflection: dict[str, Any] = {}
    if inner.require_control:
        control = TrainingCandidate(
            name="baseline_equivalent_control",
            parameters={},
            rationale="Mandatory control using the configured model parameters.",
            source="mandatory_control",
            paradigm="baseline_equivalent_control",
        )
        step_dir = artifact_dir / f"step_{step_index:03d}"
        atomic_write_json(step_dir / "candidate_batch.json", [_candidate_payload(control)])
        trials.append(
            _run_trial(
                config,
                candidate=control,
                trial_index=0,
                step_index=step_index,
                train_frame=inner_train,
                train_target=train_target,
                validation_frame=inner_validation,
                validation_target=validation_target,
                task=task,
                feature_columns=feature_columns,
                seed=seed,
                selection_metric=selection_metric,
                artifact_dir=step_dir / "trial_000",
                checkpoint_catalog=checkpoint_catalog,
            )
        )
        previous_reflection = _reflect(
            trials,
            selection_metric,
            selection_mode,
            remaining_trials=inner.trial_budget - len(trials),
        )
        atomic_write_json(step_dir / "reflection.json", previous_reflection)
        step_index += 1

    while len(trials) < inner.trial_budget:
        remaining = inner.trial_budget - len(trials)
        count = min(inner.candidates_per_step, remaining)
        context = _candidate_context(
            config,
            trials=trials,
            previous_reflection=previous_reflection,
            round_index=round_index,
            step_index=step_index,
            phase=phase,
            task=task,
            labeled_size=len(labeled),
            remaining=remaining,
            selection_metric=selection_metric,
            selection_mode=selection_mode,
            tunable=tunable,
            checkpoint_catalog=checkpoint_catalog,
        )
        step_dir = artifact_dir / f"step_{step_index:03d}"
        atomic_write_json(step_dir / "agent_context.json", context.to_public_dict())
        proposals = generator.propose(context, count)
        candidates = _validate_candidate_batch(proposals, context)
        candidates = _remove_duplicate_candidates(candidates, trials)
        if not candidates:
            fallback = RuleBasedTrainingCandidateGenerator().propose(context, count)
            candidates = _remove_duplicate_candidates(
                _validate_candidate_batch(fallback, context), trials
            )
        if not candidates:
            break
        candidates = candidates[:count]
        atomic_write_json(
            step_dir / "candidate_batch.json",
            [_candidate_payload(candidate) for candidate in candidates],
        )
        for candidate in candidates:
            trial_index = len(trials)
            trials.append(
                _run_trial(
                    config,
                    candidate=candidate,
                    trial_index=trial_index,
                    step_index=step_index,
                    train_frame=inner_train,
                    train_target=train_target,
                    validation_frame=inner_validation,
                    validation_target=validation_target,
                    task=task,
                    feature_columns=feature_columns,
                    seed=seed,
                    selection_metric=selection_metric,
                    artifact_dir=step_dir / f"trial_{trial_index:03d}",
                    checkpoint_catalog=checkpoint_catalog,
                )
            )
        previous_reflection = _reflect(
            trials,
            selection_metric,
            selection_mode,
            remaining_trials=inner.trial_budget - len(trials),
        )
        atomic_write_json(step_dir / "reflection.json", previous_reflection)
        step_index += 1

    best = _best_trial(trials, selection_metric, selection_mode)
    if best is None:
        raise RuntimeError("inner loop completed without a selectable training trial")
    best_candidate = _candidate_from_payload(best["candidate"])
    selected_config = _model_config_for_candidate(config.model, best_candidate)
    selected_initialization = _checkpoint_for_candidate(best_candidate, checkpoint_catalog)
    selected_model = build_model(
        selected_config,
        task=task,
        feature_columns=feature_columns,
        seed=seed + 100_000 + int(best["trial_index"]),
        initialization=selected_initialization,
    )
    selected_model.fit(labeled, np.asarray(target))
    checkpoint_manifest = save_selected_checkpoint(
        selected_model,
        artifact_dir,
        round_index=round_index,
        phase=phase,
        selection_metric=selection_metric,
        selection_mode=selection_mode,
        selection_value=float(best["selection_value"]),
        candidate_name=best_candidate.name,
        candidate_paradigm=best_candidate.paradigm,
    )
    _write_trial_summary(artifact_dir, trials, selection_metric, best)
    requested_trials = inner.trial_budget
    summary = {
        "enabled": True,
        "phase": phase,
        "round_index": round_index,
        "candidate_generator": generator_name,
        "trial_budget": requested_trials,
        "attempted_trials": len(trials),
        "unfilled_trial_slots": max(0, requested_trials - len(trials)),
        "candidate_generation_exhausted": len(trials) < requested_trials,
        "completed_trials": sum(item.get("status") == "completed" for item in trials),
        "failed_trials": sum(item.get("status") != "completed" for item in trials),
        "inner_steps": step_index,
        "candidate_generation_steps": step_index - int(inner.require_control),
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
        "best_trial_index": int(best["trial_index"]),
        "best_candidate": best_candidate.name,
        "best_selection_value": float(best["selection_value"]),
        "best_parameters": best["resolved_parameters"],
        "compute_multiplier_vs_single_fit": len(trials) + 1,
        "selected_plan_refit_on_all_labeled": True,
        "outer_validation_used_for_selection": False,
        "selected_checkpoint_exported": checkpoint_manifest is not None,
        "selected_checkpoint": checkpoint_manifest,
        "checkpoint_initialization": (
            selected_initialization.to_public_dict()
            if selected_initialization is not None
            else None
        ),
        "lineage_training_fits": selected_model.metadata().get("lineage_training_fits", 1),
        "device": selected_model.metadata().get("device", "unknown"),
        "accelerator_used": selected_model.metadata().get("accelerator_used", False),
    }
    atomic_write_json(artifact_dir / "best_trial.json", {**best, "summary": summary})
    atomic_write_json(artifact_dir / "inner_loop_summary.json", summary)
    return InnerLoopResult(
        model=selected_model,
        best_candidate=best_candidate,
        trial_results=trials,
        summary=summary,
    )


def _fit_control_only(
    config: AppConfig,
    *,
    labeled: pd.DataFrame,
    target: np.ndarray,
    task: str,
    feature_columns: list[str],
    seed: int,
    round_index: int,
    phase: str,
    artifact_dir: Path,
    generator_name: str,
    selection_metric: str,
    selection_mode: str,
    reason: str,
    checkpoint_catalog: CheckpointCatalog,
) -> InnerLoopResult:
    candidate = TrainingCandidate(
        name="baseline_equivalent_control",
        parameters={},
        rationale=f"Inner search skipped: {reason}",
        source="mandatory_control",
        paradigm="baseline_equivalent_control",
    )
    model = build_model(
        config.model, task=task, feature_columns=feature_columns, seed=seed + 100_000
    )
    model.fit(labeled, np.asarray(target))
    checkpoint_manifest = save_selected_checkpoint(
        model,
        artifact_dir,
        round_index=round_index,
        phase=phase,
        selection_metric=selection_metric,
        selection_mode=selection_mode,
        selection_value=None,
        candidate_name=candidate.name,
        candidate_paradigm=candidate.paradigm,
    )
    step_dir = artifact_dir / "step_000"
    trial_dir = step_dir / "trial_000"
    resolved_parameters = dict(config.model.parameters)
    locked_parameters = dict(resolved_parameters)
    plan_fingerprint = {
        "model": config.model.model_dump(mode="json"),
        "task": task,
        "seed": seed + 100_000,
        "training_data_hash": dataframe_hash(labeled),
        "training_target_hash": stable_hash(np.asarray(target).tolist()),
        "fit_scope": "all_labeled",
        "software_hash": software_fingerprint(),
    }
    plan_hash = stable_hash(plan_fingerprint)
    trial = {
        "trial_index": 0,
        "inner_step_index": 0,
        "status": "completed",
        "candidate": _candidate_payload(candidate),
        "resolved_parameters": resolved_parameters,
        "locked_parameters": locked_parameters,
        "plan_hash": plan_hash,
        "metrics": {},
        "selection_metric": selection_metric,
        "selection_value": None,
        "reused": False,
        "model": model.metadata(),
    }
    summary = {
        "enabled": bool(config.inner_loop.enabled),
        "skipped": True,
        "skip_reason": reason,
        "phase": phase,
        "round_index": round_index,
        "candidate_generator": generator_name,
        "trial_budget": config.inner_loop.trial_budget,
        "attempted_trials": 1,
        "completed_trials": 1,
        "failed_trials": 0,
        "inner_steps": 1,
        "candidate_generation_steps": 0,
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
        "best_trial_index": 0,
        "best_candidate": candidate.name,
        "best_selection_value": None,
        "best_parameters": resolved_parameters,
        "unfilled_trial_slots": max(0, config.inner_loop.trial_budget - 1),
        "candidate_generation_exhausted": False,
        "compute_multiplier_vs_single_fit": 1,
        "selected_plan_refit_on_all_labeled": True,
        "outer_validation_used_for_selection": False,
        "selected_checkpoint_exported": checkpoint_manifest is not None,
        "selected_checkpoint": checkpoint_manifest,
        "checkpoint_initialization": None,
        "checkpoint_sources": checkpoint_catalog.public_sources(),
        "lineage_training_fits": model.metadata().get("lineage_training_fits", 1),
        "device": model.metadata().get("device", "unknown"),
        "accelerator_used": model.metadata().get("accelerator_used", False),
    }
    atomic_write_json(
        artifact_dir / "split_summary.json",
        {
            "method": "not_created",
            "reason": reason,
            "train_size": len(labeled),
            "validation_size": 0,
            "outer_validation_used": False,
        },
    )
    atomic_write_json(step_dir / "candidate_batch.json", [_candidate_payload(candidate)])
    atomic_write_json(
        trial_dir / "training_plan.json",
        {
            "trial_index": 0,
            "inner_step_index": 0,
            "candidate": _candidate_payload(candidate),
            "resolved_parameters": resolved_parameters,
            "locked_parameters": locked_parameters,
            "fit_scope": "all_labeled",
            "plan_hash": plan_hash,
        },
    )
    atomic_write_json(trial_dir / "trial_result.json", trial)
    atomic_write_json(
        step_dir / "reflection.json",
        {
            "completed_trials": 1,
            "failed_trials": 0,
            "best_trial_index": 0,
            "best_candidate": candidate.name,
            "best_selection_value": None,
            "improvement_vs_control": None,
            "next_action": "search_skipped",
        },
    )
    atomic_write_json(artifact_dir / "best_trial.json", {**trial, "summary": summary})
    atomic_write_json(artifact_dir / "inner_loop_summary.json", summary)
    _write_trial_summary(artifact_dir, [trial], selection_metric, trial)
    return InnerLoopResult(
        model=model, best_candidate=candidate, trial_results=[trial], summary=summary
    )


def _run_trial(
    config: AppConfig,
    *,
    candidate: TrainingCandidate,
    trial_index: int,
    step_index: int,
    train_frame: pd.DataFrame,
    train_target: np.ndarray,
    validation_frame: pd.DataFrame,
    validation_target: np.ndarray,
    task: str,
    feature_columns: list[str],
    seed: int,
    selection_metric: str,
    artifact_dir: Path,
    checkpoint_catalog: CheckpointCatalog,
) -> dict[str, Any]:
    resolved_config = _model_config_for_candidate(config.model, candidate)
    paradigm_parameters = _paradigm_parameter_overrides(config.model, candidate)
    locked_parameters = {
        name: value
        for name, value in resolved_config.parameters.items()
        if name not in candidate.parameters and name not in paradigm_parameters
    }
    initialization = _checkpoint_for_candidate(candidate, checkpoint_catalog)
    plan = {
        "trial_index": trial_index,
        "inner_step_index": step_index,
        "candidate": _candidate_payload(candidate),
        "resolved_parameters": resolved_config.parameters,
        "paradigm_parameters": paradigm_parameters,
        "locked_model_contract": {
            "name": resolved_config.name,
            "featurizer": resolved_config.featurizer,
            "device": resolved_config.device,
            "require_accelerator": resolved_config.require_accelerator,
        },
        "locked_parameters": locked_parameters,
        "locked_compute_parameters": {
            name: resolved_config.parameters.get(name)
            for name in (
                "epochs",
                "warmup_epochs",
                "batch_size",
                "ensemble_size",
                "mc_dropout_passes",
            )
            if name in resolved_config.parameters
        },
        "checkpoint_initialization": (
            initialization.to_public_dict() if initialization is not None else None
        ),
    }
    plan_hash = stable_hash(
        {
            "resolved_model": resolved_config.model_dump(mode="json"),
            "task": task,
            "seed": seed + trial_index,
            "selection_metric": selection_metric,
            "training_data_hash": dataframe_hash(train_frame),
            "training_target_hash": stable_hash(np.asarray(train_target).tolist()),
            "validation_data_hash": dataframe_hash(validation_frame),
            "validation_target_hash": stable_hash(np.asarray(validation_target).tolist()),
            "software_hash": software_fingerprint(),
            "checkpoint_initialization": (
                initialization.to_public_dict() if initialization is not None else None
            ),
        }
    )
    plan_path = artifact_dir / "training_plan.json"
    result_path = artifact_dir / "trial_result.json"
    atomic_write_json(plan_path, {**plan, "plan_hash": plan_hash})
    if config.inner_loop.reuse_completed_trials and result_path.is_file():
        try:
            cached = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, TypeError):
            cached = {}
        if (
            cached.get("plan_hash") == plan_hash
            and cached.get("status") == "completed"
            and isinstance(cached.get("selection_value"), int | float)
        ):
            cached["reused"] = True
            atomic_write_json(result_path, cached)
            return cached
    result: dict[str, Any] = {
        **plan,
        "plan_hash": plan_hash,
        "status": "failed",
        "metrics": {},
        "selection_metric": selection_metric,
        "selection_value": None,
        "reused": False,
    }
    try:
        model = build_model(
            resolved_config,
            task=task,
            feature_columns=feature_columns,
            seed=seed + trial_index,
            initialization=initialization,
        )
        model.fit(train_frame, train_target)
        metrics = evaluate(validation_target, model.predict(validation_frame), task)
        if selection_metric not in metrics:
            raise ValueError(
                f"inner-loop selection metric '{selection_metric}' is unavailable; "
                f"metrics: {sorted(metrics)}"
            )
        selection_value = float(metrics[selection_metric])
        if not math.isfinite(selection_value):
            raise ValueError("inner-loop selection metric is not finite")
        result.update(
            {
                "status": "completed",
                "metrics": metrics,
                "selection_value": selection_value,
                "model": model.metadata(),
            }
        )
    except Exception as error:
        result["error"] = safe_error(error)
        if trial_index == 0 and config.inner_loop.require_control:
            atomic_write_json(result_path, result)
            raise RuntimeError(
                f"mandatory inner-loop control failed: {safe_error(error)}"
            ) from error
    atomic_write_json(result_path, result)
    return result


def selection_contract(config: AppConfig, task: str) -> tuple[str, str]:
    """Resolve the labeled-only metric and direction used by trials and checkpoint lineage."""

    metric = config.inner_loop.selection_metric
    if metric == "auto":
        metric = "mae" if task == "regression" else "balanced_accuracy"
    mode = config.inner_loop.selection_mode
    if mode == "auto":
        mode = (
            "max"
            if metric in {"accuracy", "balanced_accuracy", "r2", "uncertainty_error_spearman"}
            else "min"
        )
    return metric, mode


def _skip_reason(
    config: AppConfig,
    labeled: pd.DataFrame,
    target: np.ndarray,
    task: str,
    tunable: dict[str, tuple[float, float]],
    checkpoint_catalog: CheckpointCatalog,
) -> str:
    inner = config.inner_loop
    if not inner.enabled:
        return "inner_loop_disabled"
    if inner.trial_budget <= 1:
        return "trial_budget_is_one"
    if not tunable and _supported_paradigms(config, task, checkpoint_catalog) == [
        "retrain_from_scratch"
    ]:
        return "model_exposes_no_configured_tunable_parameters"
    if len(labeled) < inner.min_labeled_size:
        return f"labeled_size_below_minimum:{len(labeled)}<{inner.min_labeled_size}"
    if task == "classification":
        _, counts = np.unique(target, return_counts=True)
        if len(counts) < 2 or int(counts.min()) < 2:
            return "classification_inner_split_requires_two_samples_per_class"
    return ""


def _inner_split_positions(
    target: np.ndarray,
    *,
    task: str,
    fraction: float,
    seed: int,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(len(target))
    if groups is not None:
        if len(groups) != len(target):
            raise ValueError("inner-loop groups must align with the labeled target")
        if len(np.unique(groups)) < 2:
            raise ValueError("group-aware inner split requires at least two groups")
        attempts = 32 if task == "classification" else 1
        expected_classes = set(np.asarray(target).tolist()) if task == "classification" else set()
        for attempt in range(attempts):
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=fraction,
                random_state=seed + attempt,
            )
            train, validation = next(splitter.split(positions, target, groups))
            if task != "classification" or (
                set(np.asarray(target)[train].tolist()) == expected_classes
                and set(np.asarray(target)[validation].tolist()) == expected_classes
            ):
                return np.sort(train), np.sort(validation)
        raise ValueError("group-aware classification inner split must retain every class")
    stratify = target if task == "classification" else None
    train, validation = train_test_split(
        positions,
        test_size=fraction,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return np.sort(train), np.sort(validation)


def _candidate_context(
    config: AppConfig,
    *,
    trials: list[dict[str, Any]],
    previous_reflection: dict[str, Any],
    round_index: int,
    step_index: int,
    phase: str,
    task: str,
    labeled_size: int,
    remaining: int,
    selection_metric: str,
    selection_mode: str,
    tunable: dict[str, tuple[float, float]],
    checkpoint_catalog: CheckpointCatalog,
) -> TrainingCandidateContext:
    base = {
        name: float(config.model.parameters[name])
        for name in tunable
        if isinstance(config.model.parameters.get(name), int | float)
    }
    completed = [
        {
            "trial_index": int(item["trial_index"]),
            "name": item.get("candidate", {}).get("name", "unknown"),
            "status": item.get("status"),
            "selection_value": item.get("selection_value"),
            "parameters": item.get("candidate", {}).get("parameters", {}),
            "paradigm": item.get("candidate", {}).get("paradigm", "retrain_from_scratch"),
            "metrics": {
                str(name): float(value)
                for name, value in item.get("metrics", {}).items()
                if isinstance(value, int | float) and math.isfinite(float(value))
            },
        }
        for item in trials
    ]
    return TrainingCandidateContext(
        round_index=round_index,
        inner_step_index=step_index,
        phase=phase,
        task=task,
        labeled_size=labeled_size,
        trial_budget=config.inner_loop.trial_budget,
        remaining_trials=remaining,
        selection_metric=selection_metric,
        selection_mode=selection_mode,
        base_parameters=base,
        tunable_parameters=tunable,
        supported_paradigms=_supported_paradigms(config, task, checkpoint_catalog),
        checkpoint_sources=checkpoint_catalog.public_sources(),
        completed_trials=completed,
        previous_reflection=previous_reflection,
    )


def _rule_based_options(context: TrainingCandidateContext) -> list[TrainingCandidate]:
    base = context.base_parameters
    bounds = context.tunable_parameters
    options: list[TrainingCandidate] = []
    if "finetune_global_best" in context.supported_paradigms:
        origin = context.checkpoint_sources.get("global_best", {})
        options.append(
            TrainingCandidate(
                name="finetune_global_best",
                parameters={},
                paradigm="finetune_global_best",
                rationale=(
                    "Continue the best committed checkpoint selected only by the labeled-only "
                    f"inner metric (origin round {origin.get('origin_round', 'unknown')})."
                ),
                source="rule_based",
            )
        )
    if "finetune_previous_round" in context.supported_paradigms:
        origin = context.checkpoint_sources.get("previous_round", {})
        options.append(
            TrainingCandidate(
                name="finetune_previous_round",
                parameters={},
                paradigm="finetune_previous_round",
                rationale=(
                    "Continue the immediately previous committed active-round checkpoint "
                    f"(origin round {origin.get('origin_round', 'unknown')})."
                ),
                source="rule_based",
            )
        )
    if "robust_loss_training" in context.supported_paradigms:
        options.append(
            TrainingCandidate(
                name="robust_huber_loss",
                parameters={},
                paradigm="robust_loss_training",
                rationale=(
                    "Use Huber loss under the same architecture, epochs, batch, ensemble, and "
                    "device budget as the control."
                ),
                source="rule_based",
            )
        )
    if "curriculum_training" in context.supported_paradigms:
        options.append(
            TrainingCandidate(
                name="easy_to_hard_curriculum",
                parameters={},
                paradigm="curriculum_training",
                rationale=(
                    "Expand a deterministic easy-to-hard training subset while preserving "
                    "the control's optimizer-step budget."
                ),
                source="rule_based",
            )
        )
    paradigm_option_count = len(options)
    if "learning_rate" in base:
        low, high = bounds["learning_rate"]
        options.extend(
            [
                TrainingCandidate(
                    name="lower_learning_rate",
                    parameters={"learning_rate": _clip(base["learning_rate"] * 0.5, low, high)},
                    rationale="Conservative scratch retrain with a lower learning rate.",
                    source="rule_based",
                ),
                TrainingCandidate(
                    name="higher_learning_rate",
                    parameters={"learning_rate": _clip(base["learning_rate"] * 1.5, low, high)},
                    rationale="Scratch retrain probing a moderately higher learning rate.",
                    source="rule_based",
                ),
            ]
        )
    regularized: dict[str, float] = {}
    if "dropout" in base:
        low, high = bounds["dropout"]
        regularized["dropout"] = _clip(base["dropout"] + 0.10, low, high)
    if "weight_decay" in base:
        low, high = bounds["weight_decay"]
        proposed = max(base["weight_decay"] * 5.0, 0.00001)
        regularized["weight_decay"] = _clip(proposed, low, high)
    if regularized:
        options.insert(
            paradigm_option_count,
            TrainingCandidate(
                name="stronger_regularization",
                parameters=regularized,
                rationale="Scratch retrain with stronger dropout or weight decay.",
                source="rule_based",
            ),
        )
    if "learning_rate" in base and regularized:
        low, high = bounds["learning_rate"]
        options.append(
            TrainingCandidate(
                name="low_lr_regularized",
                parameters={
                    **regularized,
                    "learning_rate": _clip(base["learning_rate"] * 0.7, low, high),
                },
                rationale="Combine conservative optimization with regularization after prior trials.",
                source="rule_based",
            )
        )
    completed = [
        item
        for item in context.completed_trials
        if item.get("status") == "completed"
        and isinstance(item.get("selection_value"), int | float)
        and math.isfinite(float(item["selection_value"]))
    ]
    if len(completed) >= 2:
        direction = 1.0 if context.selection_mode == "min" else -1.0
        best = min(completed, key=lambda item: direction * float(item["selection_value"]))
        control = next(
            (item for item in completed if item.get("name") == "baseline_equivalent_control"),
            None,
        )
        best_parameters = best.get("parameters", {})
        if isinstance(best_parameters, dict) and best_parameters:
            improved = control is None or direction * float(
                best["selection_value"]
            ) < direction * float(control["selection_value"])
            refined: dict[str, float] = {}
            for name, raw_value in best_parameters.items():
                if name not in base or name not in bounds:
                    continue
                value = float(raw_value)
                proposed = (
                    value + 0.5 * (value - base[name]) if improved else 0.5 * (value + base[name])
                )
                refined[name] = _clip(proposed, *bounds[name])
            if refined:
                options.insert(
                    paradigm_option_count,
                    TrainingCandidate(
                        name=f"refine_{best.get('name', 'best_candidate')}",
                        parameters=refined,
                        rationale=(
                            "Continue the best observed bounded direction after sequential "
                            "reflection."
                            if improved
                            else "Interpolate toward the control after the prior candidate did "
                            "not improve the selection metric."
                        ),
                        source="rule_based_reflection",
                    ),
                )
    return options


def _validate_candidate_batch(
    proposals: list[TrainingCandidate], context: TrainingCandidateContext
) -> list[TrainingCandidate]:
    guarded: list[TrainingCandidate] = []
    for index, proposal in enumerate(proposals):
        repairs = list(proposal.repairs)
        name = _candidate_name(proposal.name or f"candidate_{index + 1}")
        paradigm = proposal.paradigm.strip().lower().replace("-", "_")
        allowed_paradigms = {"baseline_equivalent_control", *context.supported_paradigms}
        if paradigm not in allowed_paradigms:
            repairs.append(f"unsupported_paradigm:{paradigm}->retrain_from_scratch")
            paradigm = "retrain_from_scratch"
        parameters: dict[str, float] = {}
        for raw_name, raw_value in proposal.parameters.items():
            parameter = str(raw_name).strip().lower().replace("-", "_")
            bounds = context.tunable_parameters.get(parameter)
            if bounds is None:
                repairs.append(f"dropped_non_tunable_parameter:{parameter}")
                continue
            if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
                repairs.append(f"dropped_nonnumeric_parameter:{parameter}")
                continue
            value = float(raw_value)
            if not math.isfinite(value):
                repairs.append(f"dropped_nonfinite_parameter:{parameter}")
                continue
            clipped = _clip(value, *bounds)
            if abs(clipped - value) > 1e-15:
                repairs.append(f"clipped_parameter:{parameter}:{value:.6g}->{clipped:.6g}")
            if abs(clipped - context.base_parameters[parameter]) <= 1e-15:
                repairs.append(f"dropped_unchanged_parameter:{parameter}")
                continue
            parameters[parameter] = clipped
        if not parameters and paradigm not in {
            "baseline_equivalent_control",
            "finetune_global_best",
            "finetune_previous_round",
            "curriculum_training",
            "robust_loss_training",
        }:
            repairs.append("dropped_candidate_without_tunable_changes")
            continue
        guarded.append(
            replace(
                proposal,
                name=name,
                parameters=parameters,
                paradigm=paradigm,
                rationale=proposal.rationale[:500],
                repairs=repairs,
            )
        )
    return guarded


def _remove_duplicate_candidates(
    candidates: list[TrainingCandidate], trials: list[dict[str, Any]]
) -> list[TrainingCandidate]:
    signatures = {
        _candidate_signature(
            str(item.get("candidate", {}).get("paradigm", "retrain_from_scratch")),
            item.get("candidate", {}).get("parameters", {}),
        )
        for item in trials
    }
    unique: list[TrainingCandidate] = []
    for candidate in candidates:
        signature = _candidate_signature(candidate.paradigm, candidate.parameters)
        if signature in signatures:
            continue
        signatures.add(signature)
        unique.append(candidate)
    return unique


def _best_trial(trials: list[dict[str, Any]], metric: str, mode: str) -> dict[str, Any] | None:
    valid = [
        item
        for item in trials
        if item.get("status") == "completed"
        and item.get("selection_metric") == metric
        and isinstance(item.get("selection_value"), int | float)
        and math.isfinite(float(item["selection_value"]))
    ]
    if not valid:
        return None
    direction = 1.0 if mode == "min" else -1.0
    return min(
        valid,
        key=lambda item: (direction * float(item["selection_value"]), int(item["trial_index"])),
    )


def _reflect(
    trials: list[dict[str, Any]],
    selection_metric: str,
    selection_mode: str,
    *,
    remaining_trials: int,
) -> dict[str, Any]:
    best = _best_trial(trials, selection_metric, selection_mode)
    control = next(
        (
            item
            for item in trials
            if item.get("candidate", {}).get("name") == "baseline_equivalent_control"
            and item.get("status") == "completed"
        ),
        None,
    )
    improvement: float | None = None
    if best is not None and control is not None:
        best_value = float(best["selection_value"])
        control_value = float(control["selection_value"])
        improvement = (
            control_value - best_value if selection_mode == "min" else best_value - control_value
        )
    return {
        "completed_trials": len(trials),
        "failed_trials": sum(item.get("status") != "completed" for item in trials),
        "best_trial_index": int(best["trial_index"]) if best else None,
        "best_candidate": best.get("candidate", {}).get("name") if best else None,
        "best_selection_value": best.get("selection_value") if best else None,
        "improvement_vs_control": improvement,
        "remaining_trials": max(0, remaining_trials),
        "next_action": (
            "propose_nonduplicate_bounded_retrain_candidate"
            if remaining_trials > 0
            else "select_best_and_refit"
        ),
    }


def _write_trial_summary(
    artifact_dir: Path,
    trials: list[dict[str, Any]],
    selection_metric: str,
    best: dict[str, Any],
) -> None:
    rows: list[dict[str, Any]] = []
    best_index = int(best["trial_index"])
    for item in trials:
        candidate = item.get("candidate", {})
        row = {
            "trial_index": int(item["trial_index"]),
            "inner_step_index": int(item.get("inner_step_index", 0)),
            "candidate": candidate.get("name"),
            "paradigm": candidate.get("paradigm"),
            "source": candidate.get("source"),
            "status": item.get("status"),
            "selection_metric": selection_metric,
            "selection_value": item.get("selection_value"),
            "is_best_trial": int(int(item["trial_index"]) == best_index),
            "reused": bool(item.get("reused", False)),
        }
        for name, value in item.get("resolved_parameters", {}).items():
            if isinstance(value, int | float | str | bool):
                row[f"parameter_{name}"] = value
        rows.append(row)
    atomic_write_csv(artifact_dir / "summary.csv", pd.DataFrame(rows))


def _model_config_for_candidate(base: ModelConfig, candidate: TrainingCandidate) -> ModelConfig:
    parameters = {
        **base.parameters,
        **_paradigm_parameter_overrides(base, candidate),
        **candidate.parameters,
    }
    payload = base.model_dump()
    payload["parameters"] = parameters
    return ModelConfig.model_validate(payload)


def _candidate_payload(candidate: TrainingCandidate) -> dict[str, Any]:
    return {
        "name": candidate.name,
        "parameters": candidate.parameters,
        "rationale": candidate.rationale,
        "source": candidate.source,
        "paradigm": candidate.paradigm,
        "repairs": candidate.repairs,
        "fallback_used": candidate.fallback_used,
        "fallback_reason": candidate.fallback_reason,
    }


def _candidate_from_payload(payload: dict[str, Any]) -> TrainingCandidate:
    return TrainingCandidate(
        name=str(payload["name"]),
        parameters={
            str(name): float(value) for name, value in payload.get("parameters", {}).items()
        },
        rationale=str(payload.get("rationale", "")),
        source=str(payload.get("source", "unknown")),
        paradigm=str(payload.get("paradigm", "retrain_from_scratch")),
        repairs=[str(value) for value in payload.get("repairs", [])],
        fallback_used=bool(payload.get("fallback_used", False)),
        fallback_reason=str(payload.get("fallback_reason", "")),
    )


def _resolved_generator_name(config: AppConfig) -> str:
    requested = config.inner_loop.candidate_generator
    if requested == "auto":
        if config.agent.enabled and config.agent.provider == "openai_compatible":
            return "openai_compatible"
        return "rule_based"
    return requested


def _build_generator(config: AppConfig, name: str) -> TrainingCandidateGenerator:
    factory = TRAINING_CANDIDATE_GENERATORS.get(name)
    if factory is None:
        raise ValueError(
            f"unknown training candidate generator '{name}'. "
            f"Registered generators: {sorted(TRAINING_CANDIDATE_GENERATORS)}"
        )
    generator = factory(config)
    if not hasattr(generator, "propose"):
        raise TypeError("training candidate generator must provide propose(context, count)")
    return generator


def _candidate_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", value.strip().lower().replace("-", "_"))
    normalized = normalized.strip("_")[:80]
    return normalized or "training_candidate"


def _candidate_signature(paradigm: str, parameters: dict[str, Any]) -> str:
    return stable_hash(
        {
            "paradigm": paradigm,
            "parameters": {
                str(name): round(float(value), 15)
                for name, value in parameters.items()
                if isinstance(value, int | float) and not isinstance(value, bool)
            },
        }
    )


def _supported_paradigms(
    config: AppConfig, task: str, checkpoint_catalog: CheckpointCatalog
) -> list[str]:
    supported = ["retrain_from_scratch"]
    model_name = config.model.name.strip().lower().replace("-", "_")
    if model_name == "torch_mlp":
        if "global_best" in checkpoint_catalog.sources:
            supported.append("finetune_global_best")
        if "previous_round" in checkpoint_catalog.sources:
            supported.append("finetune_previous_round")
        supported.append("curriculum_training")
        if task == "regression":
            supported.append("robust_loss_training")
    configured = set(config.inner_loop.candidate_paradigms)
    return [name for name in supported if name in configured]


def _paradigm_parameter_overrides(
    base: ModelConfig, candidate: TrainingCandidate
) -> dict[str, Any]:
    if candidate.paradigm == "robust_loss_training":
        model_name = base.name.strip().lower().replace("-", "_")
        if model_name != "torch_mlp":
            raise ValueError("robust_loss_training requires the torch_mlp backend")
        return {"loss": "huber"}
    if candidate.paradigm == "curriculum_training":
        model_name = base.name.strip().lower().replace("-", "_")
        if model_name != "torch_mlp":
            raise ValueError("curriculum_training requires the torch_mlp backend")
        return {"training_schedule": "curriculum", "curriculum_start_fraction": 0.35}
    return {}


def _checkpoint_for_candidate(
    candidate: TrainingCandidate, catalog: CheckpointCatalog
) -> CheckpointReference | None:
    source = paradigm_checkpoint_source(candidate.paradigm)
    if source is None:
        return None
    reference = catalog.sources.get(source)
    if reference is None:
        raise ValueError(
            f"candidate paradigm '{candidate.paradigm}' has no validated checkpoint source"
        )
    return reference


def _clip(value: float, low: float, high: float) -> float:
    return min(max(float(value), float(low)), float(high))


register_training_candidate_generator(
    "rule_based", lambda config: RuleBasedTrainingCandidateGenerator()
)
register_training_candidate_generator(
    "openai_compatible", lambda config: OpenAICompatibleTrainingCandidateGenerator(config)
)
