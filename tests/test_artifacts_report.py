import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from agentic_al.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_yaml,
    dataframe_hash,
    environment_manifest,
    software_fingerprint,
    stable_hash,
)
from agentic_al.report import render_report


def test_atomic_artifacts_hashes_and_redaction(tmp_path: Path) -> None:
    atomic_write_json(tmp_path / "value.json", {"path": Path("/Users/alice/private")})
    atomic_write_yaml(tmp_path / "value.yaml", {"token": "Bearer secret.value"})
    frame = pd.DataFrame({"x": [1, 2]})
    atomic_write_csv(tmp_path / "value.csv", frame)
    assert json.loads((tmp_path / "value.json").read_text())["path"] == "${HOME}/private"
    assert yaml.safe_load((tmp_path / "value.yaml").read_text())["token"] == "[REDACTED]"
    assert pd.read_csv(tmp_path / "value.csv").equals(frame)
    assert stable_hash({"a": 1}) == stable_hash({"a": 1})
    assert dataframe_hash(frame) != dataframe_hash(pd.DataFrame({"x": [2, 1]}))
    assert "packages" in environment_manifest()
    assert len(software_fingerprint()) == 64
    assert environment_manifest()["agentic_al_source_hash"] == software_fingerprint()


def test_json_artifacts_and_hashes_reject_non_standard_floats(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        atomic_write_json(tmp_path / "nan.json", {"metric": float("nan")})
    with pytest.raises(ValueError, match="Out of range float values"):
        stable_hash({"metric": float("inf")})
    assert not (tmp_path / "nan.json").exists()


def test_report_generation_and_input_validation(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        render_report(tmp_path)
    pd.DataFrame({"round": [0, 1], "mae": [2.0, 1.0], "device": ["cpu", "cpu"]}).to_csv(
        tmp_path / "summary.csv", index=False
    )
    (tmp_path / "final_metrics.json").write_text(
        '{"mae":1.0,"completed_rounds":2,"device":"cpu"}', encoding="utf-8"
    )
    inner_dir = tmp_path / "round_000" / "inner_loop"
    inner_dir.mkdir(parents=True)
    (inner_dir / "inner_loop_summary.json").write_text(
        json.dumps(
            {
                "phase": "active_round",
                "round_index": 0,
                "attempted_trials": 3,
                "best_candidate": "lower_learning_rate",
                "selection_metric": "mae",
                "best_selection_value": 0.8,
                "compute_multiplier_vs_single_fit": 4,
                "device": "mps",
            }
        ),
        encoding="utf-8",
    )
    output = render_report(tmp_path)
    document = output.read_text(encoding="utf-8")
    assert "mae by round" in document
    assert "<svg" in document
    assert "completed rounds" in document
    assert "Inner-loop training candidates" in document
    assert "lower_learning_rate" in document
    assert "outer validation is never used" in document
    assert "adapt acquisition" in document


def test_report_rejects_summary_without_numbers(tmp_path: Path) -> None:
    pd.DataFrame({"device": ["cpu"]}).to_csv(tmp_path / "summary.csv", index=False)
    (tmp_path / "final_metrics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="no numeric metrics"):
        render_report(tmp_path)
