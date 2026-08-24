# Contributing

Thanks for improving Agentic Active AutoResearch. Keep changes small, testable, and reproducible.

1. Create an isolated environment with Python 3.10 or newer.
2. Install `pip install -e '.[dev]'` (add `molecule` or `chemprop` only when needed).
3. Run `ruff check .`, `mypy src/agentic_al`, `pytest`, `bandit -r src/agentic_al`, and
   `pip-audit --local --skip-editable`.
4. Add tests for behavior changes and update the relevant guide or ADR.
5. Never commit datasets, run outputs, checkpoints, `.env` files, API keys, personal paths,
   private endpoint names, or raw external-model responses.

Benchmarks must use identical splits, label budgets, rounds, model-training budgets, and seeds.
If a comparison changes any of these, label it as an ablation rather than a fair head-to-head result.
