# ADR 0005: Sequential training-candidate inner loop

- Status: accepted
- Date: 2026-08-18

## Context

The original research prototype generated model-training candidates inside every active-learning
round, trained those candidates sequentially, reflected on completed trials, and selected a plan
before scoring the unlabeled pool. Removing that layer would change the scientific method rather
than merely simplify the implementation. A public implementation also needs to avoid three common
failure modes: choosing a plan on the outer evaluation set, allowing an LLM to enlarge its own
compute budget, and claiming checkpoint continuation that a backend does not actually support.

## Decision

Each active round and the final evaluation use a bounded inner loop:

1. Make a deterministic split using only the currently labeled set.
2. Train a mandatory baseline-equivalent control unless explicitly disabled.
3. Give the candidate generator only aggregate completed-trial metrics, bounded tunable values,
   and remaining trial count.
4. Validate every proposal. Drop unknown or compute-changing parameters, clip numeric values to
   declared bounds, accept only backend-supported paradigm enums, reject unchanged/empty scratch
   candidates, and repair unsupported paradigms to `retrain_from_scratch`.
5. Train the candidate on the inner-training partition and evaluate it on the inner-validation
   partition.
6. Persist the plan, repairs, result, reflection, and plan hash before generating the next step.
7. Select the best completed trial using the configured metric and refit that plan on all currently
   labeled rows before pool prediction.

The outer validation set is used only for round/final reporting. It is never included in candidate
generation or plan selection. Epochs, batch size, architecture/depth/width, ensemble size, MC
passes, workers, and the trial/label budgets are locked. The initial implementation provided a
baseline control, scratch retraining, and same-budget Huber loss for PyTorch regression. ADR-0006
extends this decision with safe global-best/previous-round continuation and curriculum learning.

## Consequences

- The method now faithfully includes candidate generation and sequential reflection.
- Every searched round costs up to `trial_budget + 1` fits: inner trials plus the selected all-label
  refit. Reports expose this multiplier, so baselines can receive a matched budget.
- Small or unsplittable labeled sets fall back to a single auditable control fit instead of leaking
  outer validation information.
- A completed trial can be reused only when its resolved model, seed, metric, and hashed inner data
  contract match.
- A single holdout is intentionally cheaper than inner cross-validation, but noisier. Research
  claims should use multiple seeds and compare with matched-budget random search.
- External policy services see no raw rows, IDs, labels, SMILES, paths, credentials, or checkpoint
  weights. ADR-0006 permits only non-identifying checkpoint lineage descriptors.

## Alternatives considered

- **Remove the inner loop:** rejected because it changes the original project and eliminates
  training-candidate generation.
- **Tune on outer validation:** rejected because it biases reported performance.
- **Let the agent change epochs or architecture:** rejected because comparisons would confound
  decision quality with additional compute.
- **Advertise warm-start before a safe backend existed:** rejected. The built-in PyTorch backend
  now implements the required typed, non-pickle, committed-lineage contract in ADR-0006; other
  backends still do not advertise continuation.
