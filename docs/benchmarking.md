# Benchmarking

## Fair comparison contract

Match candidate/evaluation rows, initial labeled IDs, total label budget, batches, rounds, model
architecture, epochs, ensemble size, uncertainty samples, device class, and hyperparameter-search
budget. The LLM policy cannot modify these quantities in Agentic Active AutoResearch.

For an inner budget of `T`, count all `T` selection fits plus the final all-label refit. Compare an
adaptive generator against a baseline allowed the same number of fits; do not compare it only with
a single fixed fit.

Scratch, continuation, curriculum, and robust candidates use the same current-fit epoch, batch,
architecture, ensemble, and optimizer-step contract. Continuation still inherits historical
training. Report `lineage_training_fits` (and preferably cumulative GPU time) so warm-start methods
are not described as equal total compute merely because the current fit is matched.

## Baselines

At minimum compare:

- random sampling;
- uncertainty sampling;
- diversity sampling;
- fixed hybrid acquisition;
- rule-based adaptive hybrid;
- LLM-advised guarded hybrid;
- fixed training plan with no inner search;
- matched-budget random training-candidate search;
- previous-round and global-best continuation as separate ablations;
- curriculum and robust-loss candidates as separate ablations;
- sequential rule-based and LLM/plugin candidate generators as separate ablations;
- a full-supervision reference when affordable.

For molecular pools, add a strong task-appropriate model, scaffold-disjoint evaluation, and a
MolPAL-style pool-based baseline when licenses and compute permit.

## Metrics

Report predictive accuracy, sample efficiency, uncertainty/error rank correlation, discovery or
hit rate under a predeclared target, diversity/coverage, experiment cost, failure rate, wall-clock,
GPU hours, number of model fits, provider calls/cost, and variability across seeds.

The outer validation/test rows and their metrics must not enter training-candidate generation or
acquisition-policy adaptation. Candidate ranking and policy feedback use only the labeled-set inner
split; final claims should use multiple seeds and, where feasible, nested evaluation to quantify
model-selection variance.

An agentic curve with more model fits is not a fair win unless compute is matched or the extra cost
is reported explicitly. Model-selected candidates remain candidates until the oracle or experiment
confirms them.
