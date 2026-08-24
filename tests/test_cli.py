from pathlib import Path

import pandas as pd
import pytest

from agentic_al import cli
from agentic_al.cli import main
from agentic_al.registry import register_dataset


def test_doctor(capsys) -> None:
    assert main(["doctor"]) == 0
    assert "accelerators" in capsys.readouterr().out


def test_cpu_demo(tmp_path: Path) -> None:
    output = tmp_path / "demo"
    assert main(["demo", "--cpu-baseline", "--output-dir", str(output)]) == 0
    assert (output / "report.html").is_file()


@pytest.mark.parametrize(
    ("yaml_text", "message"),
    [
        ("dataset:\n  kind: not_registered\n", "unknown dataset kind"),
        ("model:\n  name: not_registered\n", "unknown model"),
        (
            "inner_loop:\n  candidate_generator: not_registered\n",
            "unknown training candidate generator",
        ),
        ("acquisition:\n  weights:\n    not_registered: 1\n", "unknown acquisition"),
    ],
)
def test_validate_config_rejects_unregistered_components(
    tmp_path: Path, capsys, yaml_text: str, message: str
) -> None:
    path = tmp_path / "unknown.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    assert main(["validate-config", "--config", str(path)]) == 2
    assert message in capsys.readouterr().err


def test_validate_config_imports_dataset_plugin(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "plugin.yaml"
    path.write_text("dataset:\n  kind: cli_plugin_rows\n", encoding="utf-8")

    def install_plugin(module: str):
        assert module == "unit_plugin"
        register_dataset(
            "cli_plugin_rows",
            lambda config: pd.DataFrame({"sample_id": ["a"], "feature": [1.0], "target": [2.0]}),
            replace=True,
        )

    monkeypatch.setattr(cli.importlib, "import_module", install_plugin)
    assert (
        main(
            [
                "validate-config",
                "--config",
                str(path),
                "--plugin",
                "unit_plugin",
            ]
        )
        == 0
    )
    assert "valid:" in capsys.readouterr().out


def test_validate_config_imports_explicit_plugin_file(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "plugin-file.yaml"
    config_path.write_text("dataset:\n  kind: cli_plugin_file_rows\n", encoding="utf-8")
    plugin_path = tmp_path / "reviewed_plugin.py"
    plugin_path.write_text(
        "from agentic_al import register_dataset\n"
        "import pandas as pd\n"
        "register_dataset(\n"
        "    'cli_plugin_file_rows',\n"
        "    lambda config: pd.DataFrame(\n"
        "        {'sample_id': ['a'], 'feature': [1.0], 'target': [2.0]}\n"
        "    ),\n"
        "    replace=True,\n"
        ")\n",
        encoding="utf-8",
    )
    assert (
        main(
            [
                "validate-config",
                "--config",
                str(config_path),
                "--plugin",
                str(plugin_path),
            ]
        )
        == 0
    )
    assert "valid:" in capsys.readouterr().out


def test_missing_plugin_file_is_reported_without_traceback(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    result = main(
        [
            "validate-config",
            "--config",
            str(config_path),
            "--plugin",
            str(tmp_path / "missing_plugin.py"),
        ]
    )
    assert result == 2
    assert "plugin file not found" in capsys.readouterr().err


def test_missing_plugin_is_reported_without_traceback(tmp_path: Path, capsys) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("{}\n", encoding="utf-8")
    result = main(
        [
            "validate-config",
            "--config",
            str(path),
            "--plugin",
            "module_that_does_not_exist_for_agentic_al_test",
        ]
    )
    assert result == 2
    assert "ModuleNotFoundError" in capsys.readouterr().err
