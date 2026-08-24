"""Self-contained, dependency-light HTML reporting for completed runs."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import atomic_write_text


def render_report(run_dir: str | Path) -> Path:
    root = Path(run_dir)
    summary_path = root / "summary.csv"
    final_path = root / "final_metrics.json"
    if not summary_path.is_file() or not final_path.is_file():
        raise FileNotFoundError("run must contain summary.csv and final_metrics.json")
    summary = pd.read_csv(summary_path)
    final_metrics = json.loads(final_path.read_text(encoding="utf-8"))
    metric_name = _primary_metric(summary)
    chart = _line_chart(summary[metric_name].to_numpy(dtype=float), metric_name)
    metric_cards = "".join(
        f'<div class="metric"><span>{html.escape(str(name).replace("_", " "))}</span>'
        f"<strong>{_format_value(value)}</strong></div>"
        for name, value in final_metrics.items()
        if isinstance(value, int | float)
    )
    table_columns = [
        column
        for column in ("round", "labeled_size_after", "pool_size_after", metric_name, "device")
        if column in summary.columns
    ]
    table = summary[table_columns].to_html(index=False, border=0, classes="data-table")
    inner_loop_panel = _inner_loop_panel(root)
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Agentic Active AutoResearch run report</title>
  <style>
    :root {{ color-scheme: dark; --bg:#07111f; --panel:#0f1d31; --ink:#edf6ff;
      --muted:#9eb2c8; --cyan:#5ee7f2; --violet:#9b87ff; --line:#213654; }}
    * {{ box-sizing:border-box }} body {{ margin:0; font:15px/1.55 Inter,ui-sans-serif,system-ui;
      background:radial-gradient(circle at 15% 0%,#15294d 0,transparent 38%),var(--bg);color:var(--ink) }}
    main {{ width:calc(100% - 32px);max-width:1080px;margin:48px auto 80px }}
    .eyebrow {{ color:var(--cyan);letter-spacing:.14em;text-transform:uppercase;font-weight:700 }}
    h1 {{ font-size:clamp(36px,6vw,72px);line-height:1;margin:.2em 0 }}
    .lede {{ color:var(--muted);max-width:760px;font-size:18px }}
    .grid {{ display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:28px 0 }}
    .metric,.panel {{ background:linear-gradient(145deg,rgba(17,35,60,.96),rgba(10,24,42,.96));
      border:1px solid var(--line);border-radius:18px;box-shadow:0 20px 60px rgba(0,0,0,.25) }}
    .metric {{ padding:18px }} .metric span {{ color:var(--muted);display:block;overflow-wrap:anywhere;
      line-height:1.3;min-height:2.6em }}
    .metric strong {{ font-size:24px;color:var(--cyan) }} .panel {{ padding:24px;margin-top:16px;overflow:auto }}
    svg {{ width:100%;height:auto }} .data-table {{ border-collapse:collapse;width:100% }}
    th,td {{ padding:10px 12px;border-bottom:1px solid var(--line);text-align:right }}
    th:first-child,td:first-child {{ text-align:left }} th {{ color:var(--cyan) }}
    footer {{ color:var(--muted);margin-top:24px }} code {{ color:var(--violet) }}
  </style>
</head>
<body><main>
  <div class="eyebrow">Auditable active learning</div>
  <h1>Agentic Active AutoResearch run report</h1>
  <p class="lede">A local, self-contained view generated only from committed run artifacts.
  Agent proposals are advisory and pass deterministic guardrails before selection.</p>
  <section class="grid">{metric_cards}</section>
  <section class="panel"><h2>{html.escape(metric_name)} by round</h2>{chart}</section>
  <section class="panel"><h2>Round ledger</h2>{table}</section>
  {inner_loop_panel}
  <footer>Reproduce with the redacted <code>config.resolved.yaml</code> and inspect every
  selection under <code>round_*/</code>. Training plans are selected on labeled-only inner
  splits; outer validation is never used to choose a training candidate or adapt acquisition.</footer>
</main></body></html>"""
    output = root / "report.html"
    atomic_write_text(output, document)
    return output


def _inner_loop_panel(root: Path) -> str:
    records: list[dict[str, Any]] = []
    paths = sorted(root.glob("round_*/inner_loop/inner_loop_summary.json"))
    final_path = root / "final_inner_loop" / "inner_loop_summary.json"
    if final_path.is_file():
        paths.append(final_path)
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        records.append(
            {
                "phase": payload.get("phase", "unknown"),
                "round": payload.get("round_index", ""),
                "trials": payload.get("attempted_trials", ""),
                "best candidate": payload.get("best_candidate", ""),
                "selection metric": payload.get("selection_metric", ""),
                "best value": payload.get("best_selection_value", ""),
                "fit multiplier": payload.get("compute_multiplier_vs_single_fit", ""),
                "device": payload.get("device", ""),
            }
        )
    if not records:
        return ""
    table = pd.DataFrame(records).to_html(
        index=False,
        border=0,
        classes="data-table",
        escape=True,
    )
    return (
        '<section class="panel"><h2>Inner-loop training candidates</h2>'
        '<p class="lede">Each row records bounded candidate generation, inner validation, '
        "selection, and the final all-label refit.</p>"
        f"{table}</section>"
    )


def _primary_metric(summary: pd.DataFrame) -> str:
    for candidate in ("mae", "accuracy", "balanced_accuracy", "rmse", "log_loss"):
        if candidate in summary.columns:
            return candidate
    numeric = summary.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric:
        raise ValueError("summary contains no numeric metrics")
    return numeric[-1]


def _line_chart(values: np.ndarray, label: str) -> str:
    width, height, padding = 900, 280, 34
    finite = np.nan_to_num(values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    low, high = float(finite.min()), float(finite.max())
    span = max(high - low, 1e-12)
    x_step = (width - 2 * padding) / max(len(finite) - 1, 1)
    points = []
    for index, value in enumerate(finite):
        x = padding + index * x_step
        y = height - padding - (value - low) / span * (height - 2 * padding)
        points.append(f"{x:.2f},{y:.2f}")
    circles = "".join(
        f'<circle cx="{point.split(",")[0]}" cy="{point.split(",")[1]}" r="5" fill="#5ee7f2"/>'
        for point in points
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(label)} trend">'
        f'<line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#294464"/>'
        f'<polyline fill="none" stroke="#9b87ff" stroke-width="4" points="{" ".join(points)}"/>'
        f"{circles}</svg>"
    )


def _format_value(value: Any) -> str:
    return f"{value:.5g}" if isinstance(value, float) else html.escape(str(value))
