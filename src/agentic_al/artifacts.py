"""Safe, atomic artifact writing and privacy-preserving metadata helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

_SECRET_PATTERN = re.compile(
    r"(?i)("
    r"bearer\s+[a-z0-9._~+/=-]+|"
    r"sk-[a-z0-9_-]{8,}|"
    r"gh[pousr]_[a-z0-9_]{20,}|"
    r"AIza[0-9a-z_-]{20,}|"
    r"(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|client[_-]?secret|"
    r"authorization|cookie|database[_-]?url|dsn)\s*[:=]\s*[^\s,;]+"
    r")"
)
_PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN [^-\r\n]*PRIVATE KEY-----.*?-----END [^-\r\n]*PRIVATE KEY-----",
    flags=re.IGNORECASE | re.DOTALL,
)
_CREDENTIAL_URL_PATTERN = re.compile(r"(?i)https?://[^/\s:@]+:[^@\s/]+@[^\s]+")
_UNIX_USER_PATH = re.compile(r"/(?:Users|home)/[^/\s]+")
_MAC_VOLUME_PATH = re.compile(r"/Volumes/[^/\r\n]+")
_WINDOWS_USER_PATH = re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+")
_SENSITIVE_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "client_secret",
    "connection_string",
    "cookie",
    "credential",
    "credentials",
    "database_url",
    "dsn",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "session_cookie",
    "token",
}


def redact_text(value: str) -> str:
    redacted = _PRIVATE_KEY_PATTERN.sub("[REDACTED]", value)
    redacted = _CREDENTIAL_URL_PATTERN.sub("[REDACTED_URL]", redacted)
    redacted = _SECRET_PATTERN.sub("[REDACTED]", redacted)
    redacted = _UNIX_USER_PATH.sub("${HOME}", redacted)
    redacted = _MAC_VOLUME_PATH.sub("${VOLUME}", redacted)
    return _WINDOWS_USER_PATH.sub("${HOME}", redacted)


def redact_structure(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized = key.strip().lower().replace("-", "_")
            if _sensitive_key(normalized) and not normalized.endswith("_env"):
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = redact_structure(item)
        return redacted
    if isinstance(value, list):
        return [redact_structure(item) for item in value]
    if isinstance(value, tuple):
        return [redact_structure(item) for item in value]
    if isinstance(value, Path):
        return redact_text(str(value))
    if isinstance(value, str):
        return redact_text(value)
    return value


def _sensitive_key(normalized: str) -> bool:
    return any(
        normalized == name or normalized.startswith(f"{name}_") or normalized.endswith(f"_{name}")
        for name in _SENSITIVE_KEYS
    )


def safe_error(error: BaseException) -> str:
    return redact_text(f"{type(error).__name__}: {error}")[:500]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(
            redact_structure(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
    )


def atomic_write_yaml(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        yaml.safe_dump(redact_structure(payload), sort_keys=False, allow_unicode=True),
    )


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def dataframe_hash(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    return hashlib.sha256(hashed).hexdigest()


def software_fingerprint() -> str:
    """Hash the installed package sources used by resume and trial reuse."""

    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_root.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def environment_manifest() -> dict[str, Any]:
    packages = {}
    for package in (
        "agentic-active-autoresearch",
        "numpy",
        "pandas",
        "pydantic",
        "PyYAML",
        "scikit-learn",
        "torch",
    ):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = "not-installed-as-package"
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "packages": packages,
        "hash_seed_configured": "PYTHONHASHSEED" in os.environ,
        "agentic_al_source_hash": software_fingerprint(),
        "argv": [redact_text(argument) for argument in sys.argv],
    }
