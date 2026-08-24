"""Dataset loading, validation, and leakage-aware evaluation splitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .config import AppConfig
from .registry import DATASET_LOADERS, register_dataset
from .types import DatasetBundle


def load_dataset(config: AppConfig, *, require_target: bool = True) -> DatasetBundle:
    cfg = config.dataset
    loader = DATASET_LOADERS.get(cfg.kind)
    if loader is None:
        raise ValueError(
            f"unknown dataset kind '{cfg.kind}'. Registered loaders: {sorted(DATASET_LOADERS)}"
        )
    frame = loader(config)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"dataset loader '{cfg.kind}' must return a pandas DataFrame")
    # A plugin may cache or reuse its DataFrame. Isolate the structural mutations below without
    # paying for a full copy of large scientific tables.
    frame = frame.copy(deep=False)

    if cfg.id_column not in frame.columns and cfg.generate_id_if_missing:
        frame.insert(0, cfg.id_column, [f"row-{index:08d}" for index in range(len(frame))])
    if (
        cfg.group_column == "scaffold"
        and "scaffold" not in frame.columns
        and config.model.smiles_column in frame.columns
    ):
        frame["scaffold"] = _murcko_scaffolds(frame[config.model.smiles_column])

    reserved = {
        cfg.id_column,
        cfg.target_column,
        cfg.group_column,
        cfg.cost_column,
        cfg.risk_column,
    }
    feature_columns = cfg.feature_columns or [
        column for column in frame.columns if column not in reserved
    ]
    required = [cfg.id_column, *feature_columns]
    if require_target:
        required.append(cfg.target_column)
    required.extend(
        column
        for column in (cfg.group_column, cfg.cost_column, cfg.risk_column)
        if column is not None
    )
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")
    if cfg.target_column in feature_columns:
        raise ValueError("target_column cannot also be a feature_column")
    if not feature_columns:
        raise ValueError("dataset must expose at least one feature column")
    if frame[cfg.id_column].isna().any():
        raise ValueError("id_column must be non-null and unique")
    normalized_ids = frame[cfg.id_column].astype(str)
    if normalized_ids.str.strip().eq("").any() or normalized_ids.duplicated().any():
        raise ValueError("id_column must remain non-empty and unique after string normalization")
    frame[cfg.id_column] = normalized_ids
    if cfg.group_column and frame[cfg.group_column].isna().any():
        raise ValueError("group_column must not contain missing values")
    if require_target:
        _validate_target(frame[cfg.target_column], cfg.task)
    numeric_features = frame[feature_columns].select_dtypes(include=[np.number, "bool"])
    for name in numeric_features.columns:
        values = pd.to_numeric(numeric_features[name], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        if np.isinf(values).any():
            raise ValueError(f"numeric feature column '{name}' contains infinite values")
    for name in (cfg.cost_column, cfg.risk_column):
        if name:
            values = pd.to_numeric(frame[name], errors="coerce")
            if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
                raise ValueError(f"{name} must contain finite numeric values")

    return DatasetBundle(
        frame=frame.reset_index(drop=True),
        id_column=cfg.id_column,
        target_column=cfg.target_column,
        feature_columns=list(feature_columns),
        task=cfg.task,
        group_column=cfg.group_column,
        cost_column=cfg.cost_column,
        risk_column=cfg.risk_column,
    )


def split_for_evaluation(
    bundle: DatasetBundle,
    validation_fraction: float,
    seed: int,
    external_validation: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = bundle.frame
    if external_validation is not None:
        validation = _validate_external_validation(bundle, external_validation)
        candidates = frame.sort_values(
            bundle.id_column, key=lambda series: series.astype(str)
        ).reset_index(drop=True)
        candidate_ids = set(candidates[bundle.id_column].astype(str))
        validation_ids = set(validation[bundle.id_column].astype(str))
        if candidate_ids.intersection(validation_ids):
            raise ValueError("external validation IDs must be disjoint from the candidate pool")
        if bundle.group_column:
            candidate_groups = set(candidates[bundle.group_column].astype(str))
            validation_groups = set(validation[bundle.group_column].astype(str))
            if candidate_groups.intersection(validation_groups):
                raise ValueError(
                    "external validation groups must be disjoint from the candidate pool"
                )
        return candidates, validation
    if bundle.group_column:
        groups = frame[bundle.group_column].to_numpy()
        if len(np.unique(groups)) < 2:
            raise ValueError("group-aware outer split requires at least two groups")
        expected_classes = (
            set(frame[bundle.target_column].tolist()) if bundle.task == "classification" else set()
        )
        attempts = 64 if bundle.task == "classification" else 1
        train_idx: np.ndarray | None = None
        validation_idx: np.ndarray | None = None
        for attempt in range(attempts):
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=validation_fraction,
                random_state=seed + attempt,
            )
            candidate_train, candidate_validation = next(splitter.split(frame, groups=groups))
            if bundle.task != "classification" or (
                set(frame.iloc[candidate_train][bundle.target_column].tolist()) == expected_classes
                and set(frame.iloc[candidate_validation][bundle.target_column].tolist())
                == expected_classes
            ):
                train_idx, validation_idx = candidate_train, candidate_validation
                break
        if train_idx is None or validation_idx is None:
            raise ValueError(
                "group-aware classification outer split could not retain every class on both "
                "sides; each class should span multiple groups"
            )
        candidates = frame.iloc[np.sort(train_idx)].reset_index(drop=True)
        validation = frame.iloc[np.sort(validation_idx)].reset_index(drop=True)
    else:
        stratify = frame[bundle.target_column] if bundle.task == "classification" else None
        candidates, validation = train_test_split(
            frame,
            test_size=validation_fraction,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
        candidates = candidates.sort_values(
            bundle.id_column, key=lambda s: s.astype(str)
        ).reset_index(drop=True)
        validation = validation.sort_values(
            bundle.id_column, key=lambda s: s.astype(str)
        ).reset_index(drop=True)
    return candidates, validation


def _validate_external_validation(bundle: DatasetBundle, frame: pd.DataFrame) -> pd.DataFrame:
    validation = frame.copy(deep=False)
    required = [
        bundle.id_column,
        bundle.target_column,
        *bundle.feature_columns,
    ]
    if bundle.group_column:
        required.append(bundle.group_column)
    missing = sorted(set(required).difference(validation.columns))
    if missing:
        raise ValueError(f"external validation is missing required columns: {missing}")
    if validation.empty:
        raise ValueError("external validation must contain at least one row")
    if validation[bundle.id_column].isna().any():
        raise ValueError("external validation IDs must be non-null and unique")
    normalized_ids = validation[bundle.id_column].astype(str)
    if normalized_ids.str.strip().eq("").any() or normalized_ids.duplicated().any():
        raise ValueError(
            "external validation IDs must remain non-empty and unique after string normalization"
        )
    validation[bundle.id_column] = normalized_ids
    if bundle.group_column and validation[bundle.group_column].isna().any():
        raise ValueError("external validation groups must not contain missing values")
    _validate_target(validation[bundle.target_column], bundle.task)
    numeric_features = validation[bundle.feature_columns].select_dtypes(include=[np.number, "bool"])
    for name in numeric_features.columns:
        values = pd.to_numeric(numeric_features[name], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        if np.isinf(values).any():
            raise ValueError(
                f"external validation numeric feature column '{name}' contains infinite values"
            )
    return validation.sort_values(
        bundle.id_column, key=lambda series: series.astype(str)
    ).reset_index(drop=True)


def _validate_target(target: pd.Series, task: str) -> None:
    if target.isna().any():
        raise ValueError("target_column contains missing values")
    if task == "regression":
        numeric_target = pd.to_numeric(target, errors="coerce")
        if (
            numeric_target.isna().any()
            or not np.isfinite(numeric_target.to_numpy(dtype=float)).all()
        ):
            raise ValueError("regression target_column must contain finite numeric values")
    elif target.nunique(dropna=False) < 2:
        raise ValueError("classification target_column must contain at least two classes")


def choose_initial_ids(
    candidates: pd.DataFrame,
    id_column: str,
    size: int,
    seed: int,
    *,
    task: str,
    target_column: str,
) -> list[str]:
    if size >= len(candidates):
        raise ValueError("run.initial_size must leave at least one row in the acquisition pool")
    if task == "classification":
        chosen, _ = train_test_split(
            candidates,
            train_size=size,
            random_state=seed,
            shuffle=True,
            stratify=candidates[target_column],
        )
        return (
            chosen.sort_values(id_column, key=lambda s: s.astype(str))[id_column]
            .astype(str)
            .tolist()
        )
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(len(candidates), size=size, replace=False))
    return candidates.iloc[positions][id_column].astype(str).tolist()


def _synthetic_frame(config: AppConfig) -> pd.DataFrame:
    cfg = config.dataset
    if cfg.task == "regression":
        features, target = make_regression(
            n_samples=cfg.synthetic_samples,
            n_features=cfg.synthetic_features,
            n_informative=cfg.synthetic_informative,
            noise=cfg.noise,
            random_state=config.run.seed,
        )
    else:
        features, target = make_classification(
            n_samples=cfg.synthetic_samples,
            n_features=cfg.synthetic_features,
            n_informative=cfg.synthetic_informative,
            n_redundant=max(0, min(2, cfg.synthetic_features - cfg.synthetic_informative)),
            n_classes=cfg.synthetic_classes,
            random_state=config.run.seed,
        )
    columns = [f"feature_{index:02d}" for index in range(features.shape[1])]
    frame = pd.DataFrame(features, columns=columns)
    frame.insert(0, cfg.id_column, [f"sample-{index:05d}" for index in range(len(frame))])
    frame[cfg.target_column] = target
    return frame


def _tabular_path(config: AppConfig, label: str) -> Path:
    path = config.resolved_dataset_path()
    if path is None or not path.is_file():
        raise FileNotFoundError(f"{label} dataset not found: {path}")
    return path


def _csv_frame(config: AppConfig) -> pd.DataFrame:
    return pd.read_csv(_tabular_path(config, "CSV"))


def _jsonl_frame(config: AppConfig) -> pd.DataFrame:
    return pd.read_json(_tabular_path(config, "JSONL"), lines=True)


def _parquet_frame(config: AppConfig) -> pd.DataFrame:
    try:
        return pd.read_parquet(_tabular_path(config, "Parquet"))
    except ImportError as exc:
        raise RuntimeError(
            "Parquet loading requires an engine: pip install 'agentic-active-autoresearch[parquet]'"
        ) from exc


def _murcko_scaffolds(smiles: pd.Series) -> list[str]:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "automatic scaffold groups require: pip install 'agentic-active-autoresearch[molecule]'"
        ) from exc
    scaffolds: list[str] = []
    for index, value in enumerate(smiles.astype(str)):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"invalid SMILES at row {index}")
        scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
        canonical = Chem.MolToSmiles(scaffold, canonical=True) if scaffold is not None else ""
        scaffolds.append(canonical or f"acyclic::{Chem.MolToSmiles(molecule, canonical=True)}")
    return scaffolds


register_dataset("synthetic", _synthetic_frame)
register_dataset("csv", _csv_frame)
register_dataset("jsonl", _jsonl_frame)
register_dataset("parquet", _parquet_frame)
