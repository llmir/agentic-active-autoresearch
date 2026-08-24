"""Safe, typed checkpoint persistence and committed-round lineage discovery."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import atomic_write_json
from .types import CheckpointCatalog, CheckpointReference

_SCHEMA_VERSION = 1
_MAX_CHECKPOINT_BYTES = 512 * 1024**2
_MAX_UNCOMPRESSED_BYTES = 1024**3
_MAX_MANIFEST_BYTES = 16 * 1024**2
_MAX_ARRAY_ELEMENTS = 100_000_000
_MAX_ARRAY_COUNT = 10_000
_ARRAY_NAME_PATTERN = re.compile(r"[A-Za-z0-9_]{1,128}")
_SOURCE_TO_PARADIGM = {
    "global_best": "finetune_global_best",
    "previous_round": "finetune_previous_round",
}


def save_selected_checkpoint(
    model: Any,
    artifact_dir: Path,
    *,
    round_index: int,
    phase: str,
    selection_metric: str,
    selection_mode: str,
    selection_value: float | None,
    candidate_name: str,
    candidate_paradigm: str,
) -> dict[str, Any] | None:
    """Persist a built-in model's safe array state, returning its public manifest."""

    exporter = getattr(model, "checkpoint_payload", None)
    if not callable(exporter):
        return None
    model_manifest, arrays = exporter()
    if not isinstance(model_manifest, dict) or not isinstance(arrays, dict) or not arrays:
        raise ValueError("checkpoint exporter returned an invalid payload")
    clean_arrays: dict[str, np.ndarray] = {}
    for raw_name, value in arrays.items():
        name = str(raw_name)
        if name in clean_arrays:
            raise ValueError("checkpoint exporter returned duplicate normalized array names")
        array = np.asarray(value)
        clean_arrays[name] = array
    array_contract = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in clean_arrays.items()
    }
    _validate_array_contract(array_contract)

    weights_path = artifact_dir / "selected_checkpoint.npz"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{weights_path.name}.", dir=weights_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **clean_arrays)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.stat().st_size > _MAX_CHECKPOINT_BYTES:
            raise ValueError("checkpoint exceeds the safe file-size limit")
        os.replace(temporary, weights_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    weights_sha256 = _file_sha256(weights_path)
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "format": "numpy_npz_allow_pickle_false",
        "weights_file": weights_path.name,
        "weights_sha256": weights_sha256,
        "array_contract": array_contract,
        "origin": {
            "round_index": int(round_index),
            "phase": phase,
            "selection_metric": selection_metric,
            "selection_mode": selection_mode,
            "selection_value": (
                float(selection_value)
                if selection_value is not None and math.isfinite(float(selection_value))
                else None
            ),
            "candidate_name": candidate_name,
            "candidate_paradigm": candidate_paradigm,
        },
        "model": model_manifest,
        "outer_validation_used_for_selection": False,
    }
    manifest_path = artifact_dir / "selected_checkpoint.json"
    atomic_write_json(manifest_path, manifest)
    public_manifest = _public_manifest(manifest)
    public_manifest["manifest_sha256"] = _file_sha256(manifest_path)
    return public_manifest


def discover_checkpoint_catalog(
    run_dir: Path,
    *,
    current_round: int,
    selection_metric: str,
    selection_mode: str,
) -> CheckpointCatalog:
    """Resolve previous and global-best sources from committed active rounds only."""

    references: list[CheckpointReference] = []
    for round_index in range(max(0, current_round)):
        round_dir = run_dir / f"round_{round_index:03d}"
        commit_path = round_dir / "commit.json"
        manifest_path = round_dir / "inner_loop" / "selected_checkpoint.json"
        try:
            commit = json.loads(commit_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            continue
        if commit.get("status") != "complete" or int(commit.get("round", -1)) != round_index:
            continue
        if manifest_path.is_file():
            reference = _load_reference(manifest_path, source="committed_round")
            if commit.get("selected_checkpoint_sha256") != reference.weights_sha256:
                raise ValueError("checkpoint weights do not match the committed digest")
            if commit.get("selected_checkpoint_manifest_sha256") != reference.manifest_sha256:
                raise ValueError("checkpoint manifest does not match the committed digest")
            if reference.origin_round != round_index or reference.origin_phase != "active_round":
                raise ValueError("checkpoint origin does not match its committed active round")
            if not 1 <= reference.lineage_training_fits <= round_index + 1:
                raise ValueError("checkpoint lineage count is inconsistent with its origin round")
            load_checkpoint(reference)
            references.append(reference)

    sources: dict[str, CheckpointReference] = {}
    previous = next(
        (item for item in reversed(references) if item.origin_round == current_round - 1), None
    )
    if previous is not None:
        sources["previous_round"] = _with_source(previous, "previous_round")

    compatible = [
        item
        for item in references
        if item.selection_metric == selection_metric
        and item.selection_mode == selection_mode
        and item.selection_value is not None
        and math.isfinite(float(item.selection_value))
    ]
    if compatible:
        direction = 1.0 if selection_mode == "min" else -1.0
        best = min(
            compatible,
            key=lambda item: (
                direction * _finite_selection_value(item),
                item.origin_round,
            ),
        )
        sources["global_best"] = _with_source(best, "global_best")
    return CheckpointCatalog(sources=sources)


def load_checkpoint(reference: CheckpointReference) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Verify integrity and array contracts before returning non-pickle state."""

    manifest = _read_manifest(reference.manifest_path)
    if _file_sha256(reference.manifest_path) != reference.manifest_sha256:
        raise ValueError("checkpoint manifest SHA-256 mismatch")
    weights_path = reference.weights_path.resolve()
    if weights_path.parent != reference.manifest_path.resolve().parent:
        raise ValueError("checkpoint weights must be adjacent to their manifest")
    if not weights_path.is_file() or weights_path.stat().st_size > _MAX_CHECKPOINT_BYTES:
        raise ValueError("checkpoint weights file is missing or exceeds the size limit")
    actual_hash = _file_sha256(weights_path)
    if actual_hash != reference.weights_sha256 or actual_hash != manifest["weights_sha256"]:
        raise ValueError("checkpoint SHA-256 mismatch")
    contract = manifest.get("array_contract")
    if not isinstance(contract, dict) or not contract:
        raise ValueError("checkpoint array contract is missing")
    validated_contract = _validate_array_contract(contract)
    _validate_npz_container(weights_path, validated_contract)
    arrays: dict[str, np.ndarray] = {}
    total_elements = 0
    with np.load(weights_path, allow_pickle=False) as archive:
        if set(archive.files) != set(contract):
            raise ValueError("checkpoint arrays do not match the manifest contract")
        for name in archive.files:
            array = np.asarray(archive[name])
            if array.dtype.hasobject:
                raise ValueError("checkpoint contains an unsafe object array")
            expected = contract[name]
            if list(array.shape) != expected.get("shape") or str(array.dtype) != expected.get(
                "dtype"
            ):
                raise ValueError(f"checkpoint array contract mismatch for '{name}'")
            total_elements += int(array.size)
            if total_elements > _MAX_ARRAY_ELEMENTS:
                raise ValueError("checkpoint exceeds the safe array-element limit")
            arrays[name] = array.copy()
    return manifest, arrays


def paradigm_checkpoint_source(paradigm: str) -> str | None:
    for source, candidate_paradigm in _SOURCE_TO_PARADIGM.items():
        if paradigm == candidate_paradigm:
            return source
    return None


def _load_reference(manifest_path: Path, *, source: str) -> CheckpointReference:
    manifest = _read_manifest(manifest_path)
    origin = manifest["origin"]
    model = manifest["model"]
    weights_file = manifest["weights_file"]
    if weights_file != "selected_checkpoint.npz":
        raise ValueError("checkpoint weights_file must use the built-in adjacent filename")
    selection_value = origin.get("selection_value")
    return CheckpointReference(
        source=source,
        weights_path=manifest_path.parent / weights_file,
        manifest_path=manifest_path,
        origin_round=int(origin["round_index"]),
        origin_phase=str(origin["phase"]),
        selection_metric=str(origin["selection_metric"]),
        selection_mode=str(origin["selection_mode"]),
        selection_value=(float(selection_value) if selection_value is not None else None),
        weights_sha256=str(manifest["weights_sha256"]),
        manifest_sha256=_file_sha256(manifest_path),
        lineage_training_fits=int(model.get("lineage_training_fits", 1)),
    )


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        if path.stat().st_size > _MAX_MANIFEST_BYTES:
            raise ValueError("checkpoint manifest exceeds the size limit")
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeError) as error:
        raise ValueError("checkpoint manifest is unavailable or invalid") from error
    if not isinstance(manifest, dict):
        raise ValueError("checkpoint manifest root must be an object")
    if manifest.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema version")
    if manifest.get("format") != "numpy_npz_allow_pickle_false":
        raise ValueError("unsupported checkpoint format")
    if not isinstance(manifest.get("origin"), dict) or not isinstance(manifest.get("model"), dict):
        raise ValueError("checkpoint manifest is incomplete")
    if re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("weights_sha256", ""))) is None:
        raise ValueError("checkpoint SHA-256 field is invalid")
    if manifest.get("outer_validation_used_for_selection") is not False:
        raise ValueError("checkpoint lineage must explicitly exclude outer validation selection")
    return manifest


def _validate_array_contract(
    contract: dict[str, Any],
) -> dict[str, tuple[tuple[int, ...], np.dtype[Any], int]]:
    if len(contract) > _MAX_ARRAY_COUNT:
        raise ValueError("checkpoint contains too many arrays")
    validated: dict[str, tuple[tuple[int, ...], np.dtype[Any], int]] = {}
    total_elements = 0
    total_bytes = 0
    for name, raw in contract.items():
        if not isinstance(name, str) or _ARRAY_NAME_PATTERN.fullmatch(name) is None:
            raise ValueError("checkpoint array name is invalid")
        if not isinstance(raw, dict):
            raise ValueError(f"checkpoint array contract is invalid for '{name}'")
        raw_shape = raw.get("shape")
        if (
            not isinstance(raw_shape, list)
            or len(raw_shape) > 8
            or not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in raw_shape
            )
        ):
            raise ValueError(f"checkpoint array shape is invalid for '{name}'")
        shape = tuple(raw_shape)
        try:
            dtype = np.dtype(raw.get("dtype"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"checkpoint array dtype is invalid for '{name}'") from error
        if (
            dtype.hasobject
            or dtype.fields is not None
            or dtype.kind not in "biufc"
            or dtype.itemsize > 16
        ):
            raise ValueError(f"checkpoint array dtype is unsafe for '{name}'")
        elements = math.prod(shape)
        total_elements += elements
        array_bytes = elements * dtype.itemsize
        total_bytes += array_bytes
        if total_elements > _MAX_ARRAY_ELEMENTS or total_bytes > _MAX_UNCOMPRESSED_BYTES:
            raise ValueError("checkpoint exceeds the safe uncompressed array limit")
        validated[name] = (shape, dtype, array_bytes)
    return validated


def _validate_npz_container(
    path: Path, contract: dict[str, tuple[tuple[int, ...], np.dtype[Any], int]]
) -> None:
    expected_files = {f"{name}.npy": size for name, (_, _, size) in contract.items()}
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) != len(expected_files):
                raise ValueError("checkpoint ZIP member count does not match the array contract")
            actual_files = {member.filename for member in members}
            if actual_files != set(expected_files):
                raise ValueError("checkpoint ZIP members do not match the array contract")
            total_uncompressed = 0
            for member in members:
                if Path(member.filename).name != member.filename or member.flag_bits & 0x1:
                    raise ValueError("checkpoint ZIP member path or encryption is forbidden")
                if member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                    raise ValueError("checkpoint ZIP compression method is unsupported")
                expected_bytes = expected_files[member.filename]
                if not expected_bytes <= member.file_size <= expected_bytes + 4096:
                    raise ValueError("checkpoint ZIP member size does not match the array contract")
                total_uncompressed += member.file_size
                if total_uncompressed > _MAX_UNCOMPRESSED_BYTES:
                    raise ValueError("checkpoint ZIP exceeds the uncompressed size limit")
    except zipfile.BadZipFile as error:
        raise ValueError("checkpoint weights are not a valid NPZ container") from error


def _with_source(reference: CheckpointReference, source: str) -> CheckpointReference:
    return CheckpointReference(
        source=source,
        weights_path=reference.weights_path,
        manifest_path=reference.manifest_path,
        origin_round=reference.origin_round,
        origin_phase=reference.origin_phase,
        selection_metric=reference.selection_metric,
        selection_mode=reference.selection_mode,
        selection_value=reference.selection_value,
        weights_sha256=reference.weights_sha256,
        manifest_sha256=reference.manifest_sha256,
        lineage_training_fits=reference.lineage_training_fits,
    )


def checkpoint_commit_evidence(artifact_dir: Path) -> dict[str, str]:
    """Validate a selected checkpoint and return digests for a recovered round commit."""

    reference = _load_reference(
        artifact_dir / "selected_checkpoint.json", source="recovered_committed_round"
    )
    load_checkpoint(reference)
    return {
        "selected_checkpoint_sha256": reference.weights_sha256,
        "selected_checkpoint_manifest_sha256": reference.manifest_sha256,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_selection_value(reference: CheckpointReference) -> float:
    if reference.selection_value is None or not math.isfinite(reference.selection_value):
        raise ValueError("checkpoint selection value is unavailable or non-finite")
    return reference.selection_value


def _public_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest["schema_version"],
        "format": manifest["format"],
        "weights_sha256": manifest["weights_sha256"],
        "origin": manifest["origin"],
        "model": manifest["model"],
        "outer_validation_used_for_selection": False,
    }
