# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Agentic Active AutoResearch branding, the `agentic-autoresearch` command, and the
  `agentic-active-autoresearch` distribution/repository slug; the `agentic_al` import and
  `agentic-al` command remain compatibility interfaces.
- GPU-first PyTorch MLP ensemble with CUDA/MPS detection and MC Dropout.
- Optional Chemprop v2 GPU adapter for molecular tasks.
- Regression and classification active-learning loops.
- Provider-neutral, aggregate-only OpenAI-compatible strategy policy.
- Deterministic guardrails, rule-based fallback, resumable artifacts, and HTML reports.
- CSV, synthetic, tabular, and optional Morgan-fingerprint adapters.
- JSONL, optional Parquet, and trusted registered dataset loaders.
- Early rejection of non-finite training, cost, and risk values.
- Scaled float64 diversity distances with portable finite-result checks across NumPy/BLAS builds.
- Sequential training-candidate inner loop with a mandatory control, labeled-only model selection,
  bounded rule/LLM/plugin proposals, plan-hash reuse, reflection artifacts, and all-label refit.
- Same-budget Huber/SmoothL1 training candidates for PyTorch regression.
- Five concrete built-in training paradigms: from-scratch retraining, global-best checkpoint
  continuation, previous-round checkpoint continuation, curriculum training, and robust-loss
  training, with task/backend-aware availability.
- Safe current-run checkpoint continuation using committed-round discovery, SHA-256-verified NPZ
  arrays loaded without pickle, strict size/type/shape/model contracts, and public lineage metadata
  that excludes paths and weights.
- Representativeness and generic group/scaffold-coverage acquisition components.
- Explicit support and legacy-prototype migration matrices.
- Python API path for fully unlabeled external regression pools with an explicit oracle and
  disjoint labeled validation DataFrame.

### Security

- Redact dataset/output locations from public configuration and external-volume paths from errors.
- Redact secret-bearing artifact keys even when their values do not resemble a known token format.
- Restrict the Chemprop subprocess to an explicit environment allowlist and disable optional
  experiment tracking.
- Bind both checkpoint weights and manifest digests to the committed round, and enforce NPZ member,
  dtype, shape, count, compressed-size, and uncompressed-size bounds before use.
- Add a CI dependency-audit job alongside monthly Dependabot checks.
- Add Bandit static security scanning with precise reviewed annotations for the shell-free Chemprop
  subprocess boundary.
- Prevent config-driven credential indirection and arbitrary Chemprop executable selection.

### Fixed

- Make GitHub Actions use Node 24-compatible pinned action releases and keep dependency-tool
  deprecation warnings from being promoted to CI failures by the source-test warning policy.
- Reject IDs that collide after string normalization and missing group labels.
- Require every class on both sides of group-aware outer and inner classification splits.
- Reject non-finite configuration values, malformed model predictions, and non-finite custom
  acquisition scores before selection.
- Resolve Chemprop warmup epochs explicitly so one-epoch smoke runs satisfy the v2 CLI contract.
- Invalidate run resume and inner-trial reuse when built-in source or provider/model runtime identity
  changes, without persisting API keys.
- Remove outer-validation metrics from acquisition-policy feedback and restore previous guarded
  weights/inner metrics exactly on resume.
- Give each available training paradigm a deterministic proposal opportunity before scalar tuning,
  so checkpoint, curriculum, and robust candidates cannot be starved by repeated scratch variants.
- Reject non-standard `NaN` and `Infinity` values in JSON audit artifacts and stable plan hashes.
- Normalize categorical missing values before sklearn fitting and prediction so safe JSON-frozen
  checkpoint transforms stay identical across supported sklearn/Python environments.
