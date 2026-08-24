"""Auditable closed-loop active-learning engine."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .acquisition import aggregate_scores, select_top_k
from .artifacts import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_yaml,
    dataframe_hash,
    environment_manifest,
    software_fingerprint,
    stable_hash,
)
from .checkpoints import checkpoint_commit_evidence, discover_checkpoint_catalog
from .config import AppConfig, dump_public_config
from .data import choose_initial_ids, load_dataset, split_for_evaluation
from .inner_loop import run_inner_loop, selection_contract
from .metrics import evaluate, prediction_summary, validate_prediction
from .oracle import Oracle, TableOracle, validate_observations
from .policy import Policy, apply_guardrails, build_policy
from .report import render_report
from .types import AcquisitionContext, PolicyContext, RunResult


def run_experiment(
    config: AppConfig,
    *,
    oracle: Oracle | None = None,
    policy: Policy | None = None,
    validation_data: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
    resume: bool | None = None,
) -> RunResult:
    external_observation_mode = validation_data is not None
    if external_observation_mode and oracle is None:
        raise ValueError("validation_data requires an explicit external Oracle")
    if external_observation_mode and config.dataset.task != "regression":
        raise ValueError(
            "fully unlabeled external pools currently support regression only; classification "
            "requires a benchmark target table in version 0.1.0"
        )
    bundle = load_dataset(config, require_target=not external_observation_mode)
    candidates, validation = split_for_evaluation(
        bundle,
        config.split.validation_fraction,
        config.run.seed,
        external_validation=validation_data,
    )
    run_dir = Path(output_dir or config.run.output_dir).expanduser().resolve()
    should_resume = config.run.resume if resume is None else resume
    public_config = dump_public_config(config)
    fingerprint_payload = json.loads(json.dumps(public_config))
    fingerprint_payload["run"]["resume"] = False
    fingerprint_payload["run"]["output_dir"] = "${RUN_DIR}"
    runtime_contract = {
        "software_hash": software_fingerprint(),
        "provider_runtime_hash": _provider_runtime_fingerprint(config),
        "observation_mode": (
            "external_oracle_with_validation_data"
            if external_observation_mode
            else "target_table_oracle"
        ),
    }
    fingerprint_payload["runtime_contract"] = runtime_contract
    config_hash = stable_hash(fingerprint_payload)
    data_hash = stable_hash(
        {
            "dataset": dataframe_hash(bundle.frame),
            "external_validation": (
                dataframe_hash(validation) if external_observation_mode else None
            ),
        }
    )
    state_path = run_dir / "state.json"

    if run_dir.exists() and any(run_dir.iterdir()) and not should_resume:
        raise FileExistsError(
            f"run directory is not empty: {run_dir}. Choose a new directory or pass --resume."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config_hash") != config_hash or state.get("dataset_hash") != data_hash:
            raise ValueError("resume refused: configuration or dataset hash changed")
        initial_ids = [str(value) for value in state["initial_ids"]]
        selected_ids = [str(value) for value in state.get("selected_ids", [])]
        completed_rounds = int(state.get("completed_rounds", 0))
        observations = _load_observations(run_dir, bundle.id_column, bundle.target_column)
        _recover_completed_commits(
            run_dir,
            completed_rounds,
            requires_checkpoint=(
                config.model.name.strip().lower().replace("-", "_") == "torch_mlp"
            ),
        )
    else:
        initial_ids = choose_initial_ids(
            candidates,
            bundle.id_column,
            config.run.initial_size,
            config.run.seed,
            task=bundle.task,
            target_column=bundle.target_column,
        )
        selected_ids = []
        completed_rounds = 0
        active_oracle = oracle or TableOracle()
        initial_rows = _rows_by_ids(candidates, bundle.id_column, initial_ids)
        observations = validate_observations(
            active_oracle.observe(
                initial_rows, id_column=bundle.id_column, target_column=bundle.target_column
            ),
            initial_ids,
            task=bundle.task,
        )
        _write_static_manifests(
            run_dir,
            public_config=public_config,
            config_hash=config_hash,
            data_hash=data_hash,
            bundle=bundle,
            candidates=candidates,
            validation=validation,
            runtime_contract=runtime_contract,
        )
        if config.run.store_observations:
            _write_observations(run_dir, observations, bundle.id_column, bundle.target_column)
        _write_state(
            state_path,
            config_hash=config_hash,
            data_hash=data_hash,
            initial_ids=initial_ids,
            selected_ids=selected_ids,
            completed_rounds=completed_rounds,
        )

    if should_resume and not config.run.store_observations:
        raise ValueError("resume requires run.store_observations=true")
    active_oracle = oracle or TableOracle()
    active_policy = policy or build_policy(config.agent, config.acquisition.weights)
    history = _load_summary(run_dir, completed_rounds)
    previous_policy_metrics, previous_weights = _load_policy_state(
        run_dir,
        completed_rounds,
        anchor_weights=config.acquisition.weights,
    )

    for round_index in range(completed_rounds, config.run.rounds):
        labeled_ids = [*initial_ids, *selected_ids]
        labeled = _rows_by_ids(candidates, bundle.id_column, labeled_ids)
        pool = candidates[~candidates[bundle.id_column].astype(str).isin(labeled_ids)].reset_index(
            drop=True
        )
        if pool.empty:
            break
        target = np.asarray([observations[row_id] for row_id in labeled_ids])
        round_dir = run_dir / f"round_{round_index:03d}"
        selection_metric, selection_mode = selection_contract(config, bundle.task)
        checkpoint_catalog = discover_checkpoint_catalog(
            run_dir,
            current_round=round_index,
            selection_metric=selection_metric,
            selection_mode=selection_mode,
        )
        inner_result = run_inner_loop(
            config,
            labeled=labeled,
            target=target,
            task=bundle.task,
            feature_columns=bundle.feature_columns,
            seed=config.run.seed + round_index,
            round_index=round_index,
            phase="active_round",
            artifact_dir=round_dir / "inner_loop",
            checkpoint_catalog=checkpoint_catalog,
        )
        model = inner_result.model
        policy_metrics = _inner_policy_metrics(inner_result)
        validation_prediction = model.predict(validation)
        round_metrics = evaluate(
            validation[bundle.target_column].to_numpy(), validation_prediction, bundle.task
        )
        pool_prediction = validate_prediction(model.predict(pool), len(pool), bundle.task)
        batch_size = min(config.run.batch_size, len(pool))
        remaining_budget = min(len(pool), (config.run.rounds - round_index) * config.run.batch_size)
        policy_context = PolicyContext(
            round_index=round_index,
            total_rounds=config.run.rounds,
            labeled_size=len(labeled),
            pool_size=len(pool),
            batch_size=batch_size,
            remaining_budget=remaining_budget,
            task=bundle.task,
            metrics=policy_metrics,
            previous_metrics=previous_policy_metrics,
            prediction_summary=prediction_summary(pool_prediction),
            previous_weights=previous_weights,
        )
        guardrail = apply_guardrails(
            active_policy.propose(policy_context),
            config.acquisition.weights,
            config.guardrails,
        )
        proposal = guardrail.proposal
        acquisition_context = AcquisitionContext(
            pool=pool,
            labeled=labeled,
            pool_features=model.featurize(pool),
            labeled_features=model.featurize(labeled),
            prediction=pool_prediction,
            id_column=bundle.id_column,
            round_index=round_index,
            seed=config.run.seed,
            task=bundle.task,
            group_column=bundle.group_column,
            cost_column=bundle.cost_column,
            risk_column=bundle.risk_column,
        )
        final_score, component_values = aggregate_scores(
            acquisition_context, config.acquisition, proposal.weights
        )
        positions = select_top_k(final_score, pool[bundle.id_column].tolist(), batch_size)
        selected = pool.iloc[positions].copy().reset_index(drop=True)
        round_selected_ids = selected[bundle.id_column].astype(str).tolist()
        observed = validate_observations(
            active_oracle.observe(
                selected, id_column=bundle.id_column, target_column=bundle.target_column
            ),
            round_selected_ids,
            task=bundle.task,
        )
        observations.update(observed)
        selected_ids.extend(round_selected_ids)
        selection = pd.DataFrame(
            {
                bundle.id_column: round_selected_ids,
                "prediction": pool_prediction.mean[positions],
                "uncertainty": pool_prediction.uncertainty[positions],
                "final_score": final_score[positions],
                "observed_target": [observed[row_id] for row_id in round_selected_ids],
            }
        )
        for name, values in component_values.items():
            selection[f"score_{name}"] = values[positions]
        atomic_write_csv(round_dir / "selection.csv", selection)
        strategy_payload = {
            "source": proposal.source,
            "weights": proposal.weights,
            "rationale": proposal.rationale,
            "fallback_used": proposal.fallback_used,
            "fallback_reason": proposal.fallback_reason,
            "guardrail_repairs": guardrail.repairs,
            "original_weights": guardrail.original_weights,
            "policy_metric_source": "labeled_only_inner_validation",
            "policy_context_metrics": policy_metrics,
        }
        atomic_write_json(round_dir / "strategy.json", strategy_payload)
        metrics_payload: dict[str, Any] = {
            **round_metrics,
            "round": round_index,
            "labeled_size_before": len(labeled),
            "labeled_size_after": len(labeled) + len(selected),
            "pool_size_before": len(pool),
            "pool_size_after": len(pool) - len(selected),
            "selected_count": len(selected),
            "model": model.metadata(),
            "inner_loop": inner_result.summary,
        }
        atomic_write_json(round_dir / "metrics.json", metrics_payload)
        completed_rounds = round_index + 1
        if config.run.store_observations:
            _write_observations(run_dir, observations, bundle.id_column, bundle.target_column)
        row = {
            "round": round_index,
            **round_metrics,
            "labeled_size_before": len(labeled),
            "labeled_size_after": len(labeled) + len(selected),
            "pool_size_before": len(pool),
            "pool_size_after": len(pool) - len(selected),
            "selected_count": len(selected),
            "policy_source": proposal.source,
            "fallback_used": proposal.fallback_used,
            "device": model.metadata().get("device", "unknown"),
            "accelerator_used": model.metadata().get("accelerator_used", False),
            "inner_training_trials": inner_result.summary["attempted_trials"],
            "best_training_candidate": inner_result.summary["best_candidate"],
            "inner_loop_compute_multiplier": inner_result.summary[
                "compute_multiplier_vs_single_fit"
            ],
        }
        row.update({f"weight_{name}": value for name, value in proposal.weights.items()})
        history.append(row)
        atomic_write_csv(run_dir / "summary.csv", pd.DataFrame(history))
        _write_state(
            state_path,
            config_hash=config_hash,
            data_hash=data_hash,
            initial_ids=initial_ids,
            selected_ids=selected_ids,
            completed_rounds=completed_rounds,
        )
        atomic_write_json(
            round_dir / "commit.json",
            {
                "status": "complete",
                "round": round_index,
                "selected_checkpoint_exported": inner_result.summary.get(
                    "selected_checkpoint_exported", False
                ),
                "selected_checkpoint_sha256": (
                    (inner_result.summary.get("selected_checkpoint") or {}).get("weights_sha256")
                ),
                "selected_checkpoint_manifest_sha256": (
                    (inner_result.summary.get("selected_checkpoint") or {}).get("manifest_sha256")
                ),
            },
        )
        previous_policy_metrics = policy_metrics
        previous_weights = proposal.weights

    final_metrics = _final_evaluation(
        config,
        bundle=bundle,
        candidates=candidates,
        validation=validation,
        initial_ids=initial_ids,
        selected_ids=selected_ids,
        observations=observations,
        completed_rounds=completed_rounds,
        run_dir=run_dir,
    )
    atomic_write_json(run_dir / "final_metrics.json", final_metrics)
    report_path = render_report(run_dir)
    return RunResult(
        run_dir=run_dir,
        summary_path=run_dir / "summary.csv",
        report_path=report_path,
        final_metrics=final_metrics,
        completed_rounds=completed_rounds,
    )


def _final_evaluation(
    config: AppConfig,
    *,
    bundle: Any,
    candidates: pd.DataFrame,
    validation: pd.DataFrame,
    initial_ids: list[str],
    selected_ids: list[str],
    observations: dict[str, Any],
    completed_rounds: int,
    run_dir: Path,
) -> dict[str, Any]:
    labeled_ids = [*initial_ids, *selected_ids]
    labeled = _rows_by_ids(candidates, bundle.id_column, labeled_ids)
    target = np.asarray([observations[row_id] for row_id in labeled_ids])
    selection_metric, selection_mode = selection_contract(config, bundle.task)
    checkpoint_catalog = discover_checkpoint_catalog(
        run_dir,
        current_round=completed_rounds,
        selection_metric=selection_metric,
        selection_mode=selection_mode,
    )
    inner_result = run_inner_loop(
        config,
        labeled=labeled,
        target=target,
        task=bundle.task,
        feature_columns=bundle.feature_columns,
        seed=config.run.seed + completed_rounds,
        round_index=completed_rounds,
        phase="final_evaluation",
        artifact_dir=run_dir / "final_inner_loop",
        checkpoint_catalog=checkpoint_catalog,
    )
    model = inner_result.model
    values: dict[str, Any] = evaluate(
        validation[bundle.target_column].to_numpy(), model.predict(validation), bundle.task
    )
    values.update(
        {
            "completed_rounds": completed_rounds,
            "final_labeled_size": len(labeled),
            "validation_size": len(validation),
            "device": model.metadata().get("device", "unknown"),
            "accelerator_used": model.metadata().get("accelerator_used", False),
            "inner_training_trials": inner_result.summary["attempted_trials"],
            "best_training_candidate": inner_result.summary["best_candidate"],
            "inner_loop_compute_multiplier": inner_result.summary[
                "compute_multiplier_vs_single_fit"
            ],
        }
    )
    return values


def _write_static_manifests(
    run_dir: Path,
    *,
    public_config: dict[str, Any],
    config_hash: str,
    data_hash: str,
    bundle: Any,
    candidates: pd.DataFrame,
    validation: pd.DataFrame,
    runtime_contract: dict[str, str],
) -> None:
    atomic_write_yaml(run_dir / "config.resolved.yaml", public_config)
    atomic_write_json(run_dir / "environment.json", environment_manifest())
    atomic_write_json(
        run_dir / "manifest.json",
        {
            "schema_version": 1,
            "config_hash": config_hash,
            "dataset_hash": data_hash,
            "dataset_rows": len(bundle.frame),
            "candidate_rows": len(candidates),
            "validation_rows": len(validation),
            "task": bundle.task,
            "feature_count": len(bundle.feature_columns),
            **runtime_contract,
        },
    )


def _provider_runtime_fingerprint(config: AppConfig) -> str:
    uses_external = (
        config.agent.enabled and config.agent.provider == "openai_compatible"
    ) or config.inner_loop.candidate_generator == "openai_compatible"
    if not uses_external:
        return stable_hash({"provider": "offline"})
    base_url = os.environ.get(config.agent.base_url_env, "").rstrip("/")
    model = config.agent.model or os.environ.get(config.agent.model_env, "")
    return stable_hash(
        {
            "provider": "openai_compatible",
            "base_url": base_url,
            "model": model,
            "api_key_configured": bool(os.environ.get(config.agent.api_key_env, "")),
        }
    )


def _rows_by_ids(frame: pd.DataFrame, id_column: str, row_ids: list[str]) -> pd.DataFrame:
    indexed = frame.assign(__id__=frame[id_column].astype(str)).set_index("__id__", drop=False)
    missing = sorted(set(row_ids).difference(indexed.index))
    if missing:
        raise ValueError(f"row IDs not found: {missing[:5]}")
    return indexed.loc[row_ids].drop(columns="__id__").reset_index(drop=True)


def _write_observations(
    run_dir: Path,
    observations: dict[str, Any],
    id_column: str,
    target_column: str,
) -> None:
    frame = pd.DataFrame(
        [{id_column: row_id, target_column: value} for row_id, value in observations.items()]
    ).sort_values(id_column, key=lambda s: s.astype(str))
    atomic_write_csv(run_dir / "observations.csv", frame)


def _load_observations(run_dir: Path, id_column: str, target_column: str) -> dict[str, Any]:
    path = run_dir / "observations.csv"
    if not path.is_file():
        raise FileNotFoundError("resume requires observations.csv")
    frame = pd.read_csv(path)
    return dict(zip(frame[id_column].astype(str), frame[target_column], strict=True))


def _write_state(
    path: Path,
    *,
    config_hash: str,
    data_hash: str,
    initial_ids: list[str],
    selected_ids: list[str],
    completed_rounds: int,
) -> None:
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "config_hash": config_hash,
            "dataset_hash": data_hash,
            "initial_ids": initial_ids,
            "selected_ids": selected_ids,
            "completed_rounds": completed_rounds,
        },
    )


def _load_summary(run_dir: Path, completed_rounds: int) -> list[dict[str, Any]]:
    path = run_dir / "summary.csv"
    if not path.is_file():
        if completed_rounds:
            raise ValueError("resume state references completed rounds but summary.csv is missing")
        return []
    rows = pd.read_csv(path).to_dict(orient="records")
    committed = [row for row in rows if int(row.get("round", -1)) < completed_rounds]
    seen = {int(row["round"]) for row in committed}
    if seen != set(range(completed_rounds)):
        raise ValueError("resume summary does not match completed-round state")
    return committed


def _inner_policy_metrics(inner_result: Any) -> dict[str, float]:
    best_index = int(inner_result.summary.get("best_trial_index", 0))
    best = next(
        (
            item
            for item in inner_result.trial_results
            if int(item.get("trial_index", -1)) == best_index
        ),
        None,
    )
    if not isinstance(best, dict):
        return {}
    return {
        str(name): float(value)
        for name, value in best.get("metrics", {}).items()
        if isinstance(value, int | float) and math.isfinite(float(value))
    }


def _load_policy_state(
    run_dir: Path,
    completed_rounds: int,
    *,
    anchor_weights: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    if completed_rounds <= 0:
        return {}, dict(anchor_weights)
    round_dir = run_dir / f"round_{completed_rounds - 1:03d}"
    best_path = round_dir / "inner_loop" / "best_trial.json"
    strategy_path = round_dir / "strategy.json"
    try:
        best = json.loads(best_path.read_text(encoding="utf-8"))
        strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as error:
        raise ValueError("resume cannot reconstruct the previous policy state") from error
    metrics = {
        str(name): float(value)
        for name, value in best.get("metrics", {}).items()
        if isinstance(value, int | float) and math.isfinite(float(value))
    }
    raw_weights = strategy.get("weights", {})
    if not isinstance(raw_weights, dict) or not raw_weights:
        raise ValueError("resume strategy does not contain previous guarded weights")
    weights = {
        str(name): float(value)
        for name, value in raw_weights.items()
        if isinstance(value, int | float) and math.isfinite(float(value))
    }
    if not weights:
        raise ValueError("resume strategy does not contain finite previous guarded weights")
    return metrics, weights


def _recover_completed_commits(
    run_dir: Path, completed_rounds: int, *, requires_checkpoint: bool
) -> None:
    required_names = [
        "selection.csv",
        "strategy.json",
        "metrics.json",
        "inner_loop/inner_loop_summary.json",
        "inner_loop/summary.csv",
    ]
    if requires_checkpoint:
        required_names.extend(
            [
                "inner_loop/selected_checkpoint.json",
                "inner_loop/selected_checkpoint.npz",
            ]
        )
    for round_index in range(completed_rounds):
        round_dir = run_dir / f"round_{round_index:03d}"
        missing = [name for name in required_names if not (round_dir / name).is_file()]
        if missing:
            raise ValueError(
                f"resume round {round_index} is incomplete; missing artifacts: {missing}"
            )
        commit_path = round_dir / "commit.json"
        if not commit_path.is_file():
            commit: dict[str, Any] = {
                "status": "complete",
                "round": round_index,
                "recovered_from_state": True,
            }
            if requires_checkpoint:
                commit.update(checkpoint_commit_evidence(round_dir / "inner_loop"))
            atomic_write_json(
                commit_path,
                commit,
            )
