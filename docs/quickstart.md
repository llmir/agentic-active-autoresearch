# Quickstart

## 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## 2. Confirm the accelerator

```bash
agentic-autoresearch doctor
```

Formal configs require either CUDA or Apple MPS. If both are false, use a GPU machine or change
`require_accelerator` only for an explicitly labeled smoke run.

## 3. Run the small GPU loop

```bash
agentic-autoresearch demo --output-dir outputs/quickstart-gpu
```

This creates a synthetic regression pool, holds out a validation set, obtains an initial labeled
set, trains two GPU MLP members, performs MC Dropout, selects three batches, refits on the final
labeled set, and renders `report.html`. Inside each round, the default inner loop trains a control
and sequential bounded candidates, then refits the selected plan. Expect more runtime than one
fixed model fit.

## 4. Inspect evidence

Start with:

1. `environment.json`: dependency and platform record.
2. `manifest.json`: config/data hashes and split sizes.
3. `summary.csv`: comparable round metrics and resolved device.
4. `round_000/inner_loop/summary.csv`: control/candidate metrics and chosen trial.
5. `round_000/inner_loop/step_*/`: candidate context, repairs, plan, result, and reflection.
6. `round_000/inner_loop/selected_checkpoint.json`: safe weight/feature/lineage contract.
7. `round_000/strategy.json`: acquisition proposal, fallback, and guardrail repairs.
8. `round_000/selection.csv`: row-level score provenance.
9. `final_metrics.json` and `report.html`: final refit evaluation.

From `round_001` onward, inspect `inner_loop_config.json` and the trial plans to see the separately
audited global-best and previous-round continuation sources. Round zero cannot use history.

## 5. Move to a reusable config

```bash
agentic-autoresearch run --config configs/gpu_regression.yaml
```

Run a new output directory for every experimental condition. Use `--resume` only to continue the
same hashes; never use it to merge experiments.
