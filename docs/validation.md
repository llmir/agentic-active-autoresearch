# Release-candidate validation

This page records what was actually exercised for the `0.1.0` candidate through 2026-08-21. It is
a software validation record, not a benchmark or a claim of scientific superiority.

## Five-paradigm checkpoint-lineage MPS run

Environment: Homebrew Python 3.12.13, PyTorch 2.12.1, Apple MPS available, CUDA unavailable. Both
regression and classification used an 80-row synthetic pool, seed 31, two active rounds, 20 initial
labels, 6 acquisitions per round, two training epochs, one ensemble member, and two MC-dropout
passes. Regression used a six-trial inner budget; classification used five. CPU fallback was
unset, `require_accelerator=true`, and warnings were promoted to errors.

| Task/phase | Completed trials | Training paradigms observed | Optimizer steps per trial | Device | Outer validation used |
|---|---:|---|---:|---:|---:|
| regression round 0 | 5 | control, robust, curriculum, scratch variants | 4 | MPS | false |
| regression round 1 / final | 6 / 6 | control, global-best, previous-round, robust, curriculum, scratch | 6 | MPS | false |
| classification round 0 | 4 | control, curriculum, scratch variants | 4 | MPS | false |
| classification round 1 / final | 5 / 5 | control, global-best, previous-round, curriculum, scratch | 6 | MPS | false |

All 31 trials completed with `device=mps` and `accelerator_used=true`. Every phase exported a
non-pickle checkpoint. In round 1, global-best and previous-round trials both loaded round 0 and
reported `lineage_training_fits=2`; classification correctly omitted regression-only robust loss.
Within each phase, every completed paradigm used the same optimizer-step count. Round 0 reported
unfilled slots after exhausting distinct history-free plans instead of fabricating checkpoint
candidates. Checkpoint manifests recorded `outer_validation_used_for_selection=false`; global-best
selection used the compatible labeled-only inner metric. These are functional smoke outcomes, not
performance claims. Artifacts remained in temporary local directories outside the release tree.

## Automated QA

- 174 tests passed with warnings as errors in four fresh source-checkout environments:
  Python 3.10.20, 3.11.15, 3.12.13, and 3.13.13.
- The branch-aware Python 3.12 coverage run reached 86.20% overall against an enforced 85% floor.
  The engine measured 91%, the inner loop 82%, the model layer 80%, the checkpoint layer 77%, and
  the checkpoint-safe frozen featurizer layer 85%.
- Ruff format/lint and strict mypy checks passed. The strict loader accepted all eight YAML
  recipes, including the external-oracle regression example and explicit candidate paradigms.
- CSV, JSONL, optional Parquet behavior, registered loaders, grouped/class-preserving splits,
  fully unlabeled external regression, custom oracle/model validation, and non-finite input
  rejection are covered by automated tests.
- The documented `examples/custom_training_candidates.py` plugin and its complete GPU configuration
  were exercised through the CLI. Active and final phases each completed one MPS control plus two
  MPS proposals whose source remained `custom_conservative`; no built-in fallback was substituted.
- Four Image2-generated JPEGs were visually inspected and iteratively corrected for arrow
  direction, spelling, numbering, five-paradigm/checkpoint lineage, logo removal, and privacy boundaries.
  All are below the repository's 1 MB pre-commit limit and all local Markdown links resolve.

## Chemprop functional smoke

The real Chemprop 2.2.3 CLI completed a regression functional smoke in an existing dedicated
Chemprop environment on Apple MPS. The input contained 48 program-generated simple alkane SMILES;
the run used one epoch, zero warmup epochs, a small hidden dimension, one ensemble member, and two
MC-dropout passes. One checkpoint was created, and five checked predictions and uncertainties were
finite. Metadata recorded `backend=chemprop`, `device=mps`, and `accelerator_used=true`.

This verifies the adapter's current train/predict path and exposed the need to set Chemprop warmup
explicitly for one-epoch runs. It is not an independent install test, a chemically meaningful
dataset, a classification test, or a performance result.

## Release and security checks

- Secret-bearing config names, credentialed URLs, private-key text, common home/volume paths, and
  provider responses are rejected or redacted at their boundaries. The external policy receives
  only aggregate state and uses fixed environment-variable names.
- The Chemprop subprocess resolves a fixed executable name, never invokes a shell, receives an
  allowlisted environment, and disables optional experiment tracking.
- A fresh sdist and wheel built successfully and passed `twine check`. The wheel contained 24
  members and the sdist 91; per-file archive scanning found no personal username, legacy external
  volume path, private-key header, or plausible OpenAI key. The wheel contained only package and
  distribution metadata members.
- An independent Python 3.12 environment installed the wheel, resolved 25 runtime packages,
  exercised the new command, compatibility command, and module entry point, reported version 0.1.0
  and MPS availability, and completed a three-round packaged GPU demo with a generated HTML report.
  All 18 completed inner trials used MPS and collectively exercised the control plus scratch,
  global-best continuation, previous-round continuation, curriculum, and robust-loss paradigms.
  Committed checkpoint discovery returned both history sources with no public paths.
- Bandit reported zero findings for `src/` and `examples/`. A `pip-audit` scan of the 25 packages
  installed by the wheel found no known vulnerabilities as of 2026-08-21. This is a point-in-time
  result, so Dependabot and CI repeat the dependency check.
- GitHub Actions are least-privilege and pinned to immutable commits. All eight pre-commit hooks
  passed on the staged tree, including private-key and `detect-secrets` checks. The CFF 1.2 schema
  validation passed, and all 53 checked local Markdown links resolved.

## Explicitly not validated here

- No NVIDIA CUDA hardware was available, so CUDA execution remains a target-environment check.
- Chemprop classification, an independent clean Chemprop install, and a real molecular dataset run
  remain unvalidated for this candidate.
- Fully unlabeled external classification is not built in because it needs an explicit seed-label
  and class-coverage contract. Fully unlabeled external regression is supported through the Python
  API with an explicit oracle and separate labeled validation data.
- No licensed external molecular dataset is redistributed or used to make a performance claim.
- Checkpoint continuation is built in only for `torch_mlp`; Chemprop, RF, and third-party models do
  not claim that capability. Multi-output tasks, distributed pools, and historical benchmark
  conclusions are not claimed. See the [support matrix](support-matrix.md).
- GPU results may vary across PyTorch, drivers, devices, and scientific datasets; compare multiple
  seeds under the matched-budget protocol in [benchmarking](benchmarking.md).
