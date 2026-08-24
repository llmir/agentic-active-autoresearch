"""Command-line interface for local experiments and validation."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from . import __version__
from .artifacts import environment_manifest, safe_error
from .config import AppConfig, load_config
from .engine import run_experiment
from .registry import (
    ACQUISITION_FUNCTIONS,
    DATASET_LOADERS,
    MODEL_FACTORIES,
    TRAINING_CANDIDATE_GENERATORS,
)
from .report import render_report
from .types import RunResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentic-autoresearch",
        description="Auditable Agentic Active AutoResearch",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run an experiment from YAML")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--resume", action="store_true")
    run.add_argument(
        "--plugin",
        action="append",
        default=[],
        help="trusted Python module or .py file to import before resolving components",
    )

    validate = commands.add_parser("validate-config", help="validate YAML without running")
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument(
        "--plugin",
        action="append",
        default=[],
        help="trusted Python module or .py file to import before resolving components",
    )

    demo = commands.add_parser("demo", help="run a synthetic GPU demonstration")
    demo.add_argument("--task", choices=["regression", "classification"], default="regression")
    demo.add_argument("--output-dir", type=Path, default=Path("outputs/demo-gpu"))
    demo.add_argument("--cpu-baseline", action="store_true", help="use RF only as a smoke baseline")

    report = commands.add_parser("report", help="rebuild a self-contained run report")
    report.add_argument("--run-dir", type=Path, required=True)

    commands.add_parser("doctor", help="show dependency and accelerator availability")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate-config":
            _import_plugins(args.plugin)
            config = load_config(args.config)
            _validate_component_references(config)
            print(f"valid: {config.project_name}")
            return 0
        if args.command == "doctor":
            print(json.dumps(_doctor(), indent=2, sort_keys=True))
            return 0
        if args.command == "report":
            print(render_report(args.run_dir))
            return 0
        if args.command == "demo":
            config = _demo_config(args.task, args.output_dir, args.cpu_baseline)
            result = run_experiment(config)
            _print_result(result)
            return 0
        if args.command == "run":
            _import_plugins(args.plugin)
            config = load_config(args.config)
            _validate_component_references(config)
            result = run_experiment(
                config,
                output_dir=args.output_dir,
                resume=True if args.resume else None,
            )
            _print_result(result)
            return 0
    except (
        FileNotFoundError,
        FileExistsError,
        ImportError,
        TypeError,
        ValueError,
        RuntimeError,
        ValidationError,
    ) as error:
        print(f"error: {safe_error(error)}", file=sys.stderr)
        return 2
    return 1


def _demo_config(task: str, output_dir: Path, cpu_baseline: bool) -> AppConfig:
    model = (
        {
            "name": "random_forest",
            "device": "cpu",
            "require_accelerator": False,
            "parameters": {"n_estimators": 96, "min_samples_leaf": 2, "n_jobs": 1},
        }
        if cpu_baseline
        else {
            "name": "torch_mlp",
            "device": "auto",
            "require_accelerator": True,
            "parameters": {
                "hidden_dims": [128, 64],
                "dropout": 0.10,
                "epochs": 30,
                "batch_size": 64,
                "learning_rate": 0.001,
                "weight_decay": 0.00001,
                "ensemble_size": 2,
                "mc_dropout_passes": 10,
            },
        }
    )
    acquisition = {
        "weights": {"uncertainty": 0.5, "diversity": 0.3, "random": 0.15, "target": 0.05},
        "goal": "target_class" if task == "classification" else "minimize",
        "target_class": 1 if task == "classification" else None,
    }
    return AppConfig.model_validate(
        {
            "project_name": f"synthetic-{task}-demo",
            "dataset": {"kind": "synthetic", "task": task, "synthetic_samples": 240},
            "run": {"rounds": 3, "initial_size": 30, "batch_size": 15, "output_dir": output_dir},
            "model": model,
            "acquisition": acquisition,
        }
    )


def _import_plugins(references: Sequence[str]) -> None:
    for reference in references:
        if reference.endswith(".py") or Path(reference).is_file():
            _import_plugin_file(Path(reference))
        else:
            importlib.import_module(reference)


def _import_plugin_file(path: Path) -> None:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ImportError(f"plugin file not found: {resolved}")
    if resolved.suffix != ".py":
        raise ImportError("plugin file must use the .py suffix")
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
    module_name = f"_agentic_al_plugin_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create an import specification for plugin: {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        sys.modules.pop(module_name, None)
        raise ImportError(f"plugin file import failed: {safe_error(error)}") from error


def _validate_component_references(config: AppConfig) -> None:
    if config.dataset.kind not in DATASET_LOADERS:
        raise ValueError(
            f"unknown dataset kind '{config.dataset.kind}'. "
            f"Registered loaders: {sorted(DATASET_LOADERS)}"
        )
    model_name = config.model.name.strip().lower().replace("-", "_")
    if model_name not in MODEL_FACTORIES:
        raise ValueError(
            f"unknown model '{config.model.name}'. Registered models: {sorted(MODEL_FACTORIES)}"
        )
    generator_name = config.inner_loop.candidate_generator
    if generator_name != "auto" and generator_name not in TRAINING_CANDIDATE_GENERATORS:
        raise ValueError(
            f"unknown training candidate generator '{generator_name}'. "
            f"Registered generators: {sorted(TRAINING_CANDIDATE_GENERATORS)}"
        )
    missing = sorted(
        name.strip().lower().replace("-", "_")
        for name in config.acquisition.weights
        if name.strip().lower().replace("-", "_") not in ACQUISITION_FUNCTIONS
    )
    if missing:
        raise ValueError(f"unknown acquisition components: {missing}")


def _doctor() -> dict[str, object]:
    report = environment_manifest()
    try:
        import torch

        report["accelerators"] = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "mps_available": bool(
                getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            ),
        }
    except ImportError:
        report["accelerators"] = {"torch_available": False}
    for optional in ("rdkit", "chemprop"):
        try:
            importlib.import_module(optional)
            report[f"{optional}_available"] = True
        except ImportError:
            report[f"{optional}_available"] = False
    return report


def _print_result(result: RunResult) -> None:
    print(f"run_dir={result.run_dir}")
    print(f"completed_rounds={result.completed_rounds}")
    print(f"report={result.report_path}")
    print(f"final_metrics={json.dumps(result.final_metrics, sort_keys=True)}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
