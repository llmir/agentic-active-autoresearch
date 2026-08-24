# Migration from the research prototype

The open-source package is an isolated refactor; the legacy research directory remains untouched.
This is a capability audit, not a claim that every prototype experiment was copied.

## Preserved and exercised

| Prototype concept | Agentic Active AutoResearch location | Evidence boundary |
|---|---|---|
| outer pool-based active-learning loop | typed engine + idempotent oracle contract | regression/classification end-to-end tests |
| sequential training-plan candidates | mandatory control, generator, inner validation, reflection, selection, all-label refit | unit tests plus MPS integration run |
| strategy agent | offline rule policy or aggregate-only OpenAI-compatible policy | strict JSON, bounded response, fallback tests |
| acquisition principles | random, uncertainty, diversity, target, representativeness, group/scaffold coverage, cost/risk penalties | component and engine tests |
| Chemprop + dry-run runners | GPU PyTorch core, optional Chemprop adapter, explicit RF baseline | PyTorch/RF tested; Chemprop 2.2.3 MPS regression functional smoke completed |
| scaffold separation | generic group split plus optional RDKit scaffold derivation | no-overlap and class-coverage tests |
| guardrail reports | original proposal, repairs, final weights, fallback, per-trial plan/result | artifact-contract tests |

## Redesigned rather than copied

| Prototype subsystem | Release-candidate replacement | Why |
|---|---|---|
| provider-specific agent | environment-only OpenAI-compatible boundary | removes organization endpoints and credentials |
| unrestricted nested config overrides | Pydantic strict schemas and registries | fail early and keep prompts away from code execution |
| round memory | aggregate summary plus completed-trial reflection | prevents raw rows and responses entering prompts |
| checkpoint registry as recovery state | atomic state, observation ledger, plan hashes, committed-round evidence, safe NPZ lineage | allows typed continuation without loading pickle or arbitrary paths |
| Streamlit dashboard | self-contained HTML report | portable, dependency-light, and easy to archive |
| checked-in datasets/outputs | ignored external data and run directories | licensing, privacy, and repository-size safety |

## Capability boundary in 0.1.0

| Legacy capability | Current status | Safe path forward |
|---|---|---|
| global-best and previous-round continuation | migrated for built-in PyTorch MLP | current-run committed NPZ + JSON + SHA-256; labeled-only inner selection; frozen feature/target contract |
| robust loss | migrated for PyTorch regression with fixed compute | Huber/SmoothL1 candidate is selected on the labeled-only inner split |
| curriculum learning | migrated for PyTorch regression/classification | deterministic easy-to-hard expansion with matched optimizer steps |
| sample weighting and ensemble expansion | not built in as candidate paradigms | add typed backend support and matched-compute tests |
| specialized QM9 download/cleaning/unit heuristics | not included | publish a separately licensed data recipe with checksums and provenance |
| large ablation/benchmark orchestration scripts | replaced by a benchmark contract, not a runner | add a seed-matrix runner without embedding historical result claims |
| evidential and hybrid uncertainty variants | not claimed | add calibration tests and backend-specific uncertainty metadata first |
| historical benchmark conclusions | not transferred | rerun on this package with identical data, splits, seeds, and budgets |

The missing items are visible by design. Unsupported candidate paradigms are repaired to scratch
retraining and logged; they are never silently simulated. See the [support matrix](support-matrix.md)
and [benchmark contract](benchmarking.md).
