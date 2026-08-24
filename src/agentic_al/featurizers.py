"""Built-in feature encoders for generic tables and molecular SMILES."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ModelConfig


class Featurizer(Protocol):
    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray: ...

    def transform(self, frame: pd.DataFrame) -> np.ndarray: ...

    def checkpoint_metadata(self) -> dict[str, Any]: ...


class TabularFeaturizer:
    def __init__(self, feature_columns: list[str]) -> None:
        self.feature_columns = feature_columns
        self.transformer: ColumnTransformer | None = None

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        view = frame[self.feature_columns].copy()
        numeric = view.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        categorical = [column for column in self.feature_columns if column not in numeric]
        if categorical:
            view[categorical] = view[categorical].where(view[categorical].notna(), np.nan)
        transformers: list[tuple[str, Pipeline, list[str]]] = []
        if numeric:
            transformers.append(
                (
                    "numeric",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    numeric,
                )
            )
        if categorical:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            (
                                "encode",
                                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            ),
                        ]
                    ),
                    categorical,
                )
            )
        self.transformer = ColumnTransformer(transformers, remainder="drop")
        return np.asarray(self.transformer.fit_transform(view), dtype=np.float32)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.transformer is None:
            raise RuntimeError("featurizer has not been fitted")
        view = frame[self.feature_columns].copy()
        categorical_entry = next(
            (entry for entry in self.transformer.transformers_ if entry[0] == "categorical"),
            None,
        )
        if categorical_entry is not None:
            categorical = [str(value) for value in categorical_entry[2]]
            view[categorical] = view[categorical].where(view[categorical].notna(), np.nan)
        return np.asarray(self.transformer.transform(view), dtype=np.float32)

    def checkpoint_metadata(self) -> dict[str, Any]:
        if self.transformer is None:
            raise RuntimeError("featurizer has not been fitted")
        numeric_columns: list[str] = []
        categorical_columns: list[str] = []
        numeric_statistics: list[float] = []
        numeric_means: list[float] = []
        numeric_scales: list[float] = []
        categorical_statistics: list[dict[str, Any]] = []
        categorical_categories: list[list[dict[str, Any]]] = []
        if "numeric" in self.transformer.named_transformers_:
            numeric_pipeline = self.transformer.named_transformers_["numeric"]
            numeric_columns = [str(value) for value in self.transformer.transformers_[0][2]]
            numeric_statistics = [
                float(value) for value in numeric_pipeline.named_steps["impute"].statistics_
            ]
            numeric_means = [float(value) for value in numeric_pipeline.named_steps["scale"].mean_]
            numeric_scales = [
                float(value) for value in numeric_pipeline.named_steps["scale"].scale_
            ]
        categorical_entry = next(
            (entry for entry in self.transformer.transformers_ if entry[0] == "categorical"),
            None,
        )
        if categorical_entry is not None:
            categorical_pipeline = self.transformer.named_transformers_["categorical"]
            categorical_columns = [str(value) for value in categorical_entry[2]]
            categorical_statistics = [
                _encode_scalar(value)
                for value in categorical_pipeline.named_steps["impute"].statistics_
            ]
            categorical_categories = [
                [_encode_scalar(value) for value in values]
                for values in categorical_pipeline.named_steps["encode"].categories_
            ]
        output_dim = len(numeric_columns) + sum(len(values) for values in categorical_categories)
        return {
            "kind": "tabular_frozen_v1",
            "feature_columns": list(self.feature_columns),
            "numeric_columns": numeric_columns,
            "numeric_imputer_statistics": numeric_statistics,
            "numeric_means": numeric_means,
            "numeric_scales": numeric_scales,
            "categorical_columns": categorical_columns,
            "categorical_imputer_statistics": categorical_statistics,
            "categorical_categories": categorical_categories,
            "output_dim": output_dim,
        }


class MorganFeaturizer:
    def __init__(self, smiles_column: str, radius: int, bits: int) -> None:
        self.smiles_column = smiles_column
        self.radius = radius
        self.bits = bits

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.transform(frame)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        try:
            from rdkit import Chem, DataStructs
            from rdkit.Chem import rdFingerprintGenerator
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Morgan featurization requires the optional 'molecule' dependencies: "
                "pip install 'agentic-active-autoresearch[molecule]'"
            ) from exc
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=self.radius, fpSize=self.bits)
        matrix = np.zeros((len(frame), self.bits), dtype=np.float32)
        for index, value in enumerate(frame[self.smiles_column].astype(str)):
            molecule = Chem.MolFromSmiles(value)
            if molecule is None:
                raise ValueError(f"invalid SMILES at row {index}")
            fingerprint = generator.GetFingerprint(molecule)
            DataStructs.ConvertToNumpyArray(fingerprint, matrix[index])
        return matrix

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "kind": "morgan_v1",
            "feature_columns": [self.smiles_column],
            "smiles_column": self.smiles_column,
            "radius": self.radius,
            "bits": self.bits,
            "output_dim": self.bits,
        }


class FrozenTabularFeaturizer:
    """Small safe transform reconstructed from JSON rather than a pickled sklearn object."""

    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata
        self.feature_columns = [str(value) for value in metadata["feature_columns"]]
        self.numeric_columns = [str(value) for value in metadata["numeric_columns"]]
        self.numeric_statistics = np.asarray(
            metadata["numeric_imputer_statistics"], dtype=np.float64
        )
        self.numeric_means = np.asarray(metadata["numeric_means"], dtype=np.float64)
        self.numeric_scales = np.asarray(metadata["numeric_scales"], dtype=np.float64)
        self.categorical_columns = [str(value) for value in metadata["categorical_columns"]]
        self.categorical_statistics = [
            _decode_scalar(value) for value in metadata["categorical_imputer_statistics"]
        ]
        self.categorical_categories = [
            [_decode_scalar(value) for value in values]
            for values in metadata["categorical_categories"]
        ]
        if not (
            len(self.numeric_columns)
            == len(self.numeric_statistics)
            == len(self.numeric_means)
            == len(self.numeric_scales)
        ):
            raise ValueError("checkpoint numeric feature contract is inconsistent")
        if len(self.categorical_columns) != len(self.categorical_statistics) or len(
            self.categorical_columns
        ) != len(self.categorical_categories):
            raise ValueError("checkpoint categorical feature contract is inconsistent")
        if np.any(~np.isfinite(self.numeric_statistics)) or np.any(
            ~np.isfinite(self.numeric_means)
        ):
            raise ValueError("checkpoint numeric transform contains non-finite values")
        if np.any(~np.isfinite(self.numeric_scales)) or np.any(self.numeric_scales <= 0.0):
            raise ValueError("checkpoint numeric scales must be finite and positive")

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.transform(frame)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [column for column in self.feature_columns if column not in frame]
        if missing:
            raise ValueError(f"checkpoint feature columns are missing: {missing}")
        blocks: list[np.ndarray] = []
        if self.numeric_columns:
            numeric = frame[self.numeric_columns].to_numpy(dtype=np.float64, copy=True)
            missing_values = np.isnan(numeric)
            if missing_values.any():
                numeric[missing_values] = np.take(
                    self.numeric_statistics, np.nonzero(missing_values)[1]
                )
            blocks.append((numeric - self.numeric_means) / self.numeric_scales)
        for column, statistic, categories in zip(
            self.categorical_columns,
            self.categorical_statistics,
            self.categorical_categories,
            strict=True,
        ):
            values = frame[column].to_numpy(dtype=object, copy=True)
            values[pd.isna(values)] = statistic
            blocks.append(
                np.column_stack([values == category for category in categories]).astype(float)
            )
        if not blocks:
            raise ValueError("checkpoint feature contract contains no output features")
        result = np.column_stack(blocks).astype(np.float32, copy=False)
        if result.shape[1] != int(self.metadata["output_dim"]):
            raise ValueError("checkpoint feature output dimension mismatch")
        return result

    def checkpoint_metadata(self) -> dict[str, Any]:
        return dict(self.metadata)


def featurizer_from_checkpoint(
    metadata: dict[str, Any], *, expected_feature_columns: list[str]
) -> Featurizer:
    """Reconstruct only allowlisted built-in feature transforms from a safe manifest."""

    if [str(value) for value in metadata.get("feature_columns", [])] != [
        str(value) for value in expected_feature_columns
    ]:
        raise ValueError("checkpoint feature columns do not match the current model contract")
    kind = metadata.get("kind")
    if kind == "tabular_frozen_v1":
        return FrozenTabularFeaturizer(metadata)
    if kind == "morgan_v1":
        return MorganFeaturizer(
            smiles_column=str(metadata["smiles_column"]),
            radius=int(metadata["radius"]),
            bits=int(metadata["bits"]),
        )
    raise ValueError(f"unsupported checkpoint featurizer kind: {kind}")


def _encode_scalar(value: Any) -> dict[str, Any]:
    scalar = value.item() if isinstance(value, np.generic) else value
    if isinstance(scalar, bool):
        return {"type": "bool", "value": scalar}
    if isinstance(scalar, int):
        return {"type": "int", "value": scalar}
    if isinstance(scalar, float):
        if not np.isfinite(scalar):
            raise ValueError("categorical checkpoint values must be finite")
        return {"type": "float", "value": scalar}
    return {"type": "str", "value": str(scalar)}


def _decode_scalar(payload: dict[str, Any]) -> Any:
    kind = payload.get("type")
    value = payload.get("value")
    if kind == "bool":
        if not isinstance(value, bool):
            raise ValueError("checkpoint boolean scalar is invalid")
        return value
    if kind == "int" and isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if kind == "float" and isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if kind == "str" and isinstance(value, str):
        return value
    raise ValueError("checkpoint categorical scalar type is unsupported")


def build_featurizer(config: ModelConfig, feature_columns: list[str]) -> Featurizer:
    if config.featurizer == "morgan":
        if config.smiles_column not in feature_columns:
            raise ValueError("model.smiles_column must be listed in dataset.feature_columns")
        return MorganFeaturizer(
            smiles_column=config.smiles_column,
            radius=config.fingerprint_radius,
            bits=config.fingerprint_bits,
        )
    return TabularFeaturizer(feature_columns)
