# ADR-0006: Safe checkpoint lineage and matched-budget training paradigms

## Status

Accepted

## Context

The inner loop must compare materially different training strategies, not only scalar
hyperparameters. Requested strategies include training from scratch, fine-tuning the best model
from any previous active-learning round, fine-tuning the immediately previous round, curriculum
learning, and robust-loss training. These comparisons must remain reproducible and must not turn
the outer validation set into a hidden model-selection set.

The design also has to satisfy the following requirements:

- GPU-first training and the configured accelerator requirement remain in force.
- Epochs, batch size, ensemble size, network architecture, data split, and trial count are locked.
- Historical model files are never accepted from the external candidate-generation service.
- A resumed run cannot silently use a partial, corrupt, foreign, or differently configured model.
- Checkpoint persistence must not use Python pickle or expose user paths, credentials, samples, or
  labels to an external service.
- Warm-start comparisons must report inherited training lineage, because equal per-trial budgets do
  not make cumulative historical compute equal.

## Decision

The built-in `torch_mlp` backend supports five explicit paradigms:

1. `retrain_from_scratch`
2. `finetune_global_best`
3. `finetune_previous_round`
4. `curriculum_training`
5. `robust_loss_training`

Only checkpoints produced by this run's completed active rounds are eligible. The global-best
source is selected with the configured labeled-only inner metric and direction. Outer validation
metrics are never read for checkpoint selection. The previous-round source is the most recent
committed active round. Round zero therefore exposes neither fine-tuning paradigm.

Weights are stored as NumPy NPZ arrays with `allow_pickle=False`. A JSON manifest records the exact
array shapes and dtypes, model and featurizer contract, class or target-normalization contract,
origin round, inner selection evidence, lineage fit count, and SHA-256. Each round commit binds both
the weights and manifest digests. Loading verifies those committed digests, the adjacent filename,
schema, size bounds, array keys, shapes, dtypes, task, classes, feature space, architecture, and
ensemble size before any state is applied.

Fine-tuning freezes the originating feature transform and target normalization so that inherited
weights preserve their meaning. The new labeled rows are transformed with that recorded contract.
Unknown categorical values map to an all-zero one-hot block. Incompatible classes, columns, or
architecture fail that candidate and are retained as an audited failed trial.

Curriculum difficulty is deterministic and training-only: regression sorts samples by distance
from the feature centroid; classification ranks distance from each class centroid. The accessible
subset expands from easy to hard over epochs. Early subsets are deterministically resampled to the
full training-set length, so every paradigm executes the same optimizer-step budget per ensemble
member. Robust training uses Huber loss for regression under the same locked compute contract.

External candidate services receive only aggregate checkpoint descriptors: source kind, origin
round and phase, inner metric/value/direction, digest, and lineage fit count. They never receive
paths, weights, row identifiers, raw features, targets, SMILES, or arbitrary checkpoint controls.

## Consequences

### Positive

- The diagrammed inner-loop strategies correspond to executable, testable behavior.
- Checkpoint provenance is local, integrity checked, resume safe, and free of pickle loading.
- Global-best selection cannot leak the outer benchmark into training-plan choice.
- Per-fit compute is matched and cumulative warm-start advantage is visible rather than hidden.
- The typed checkpoint boundary can later support additional safe built-in backends.

### Negative

- Warm-start trials still inherit historical compute and cannot be described as equal total compute.
- Freezing an earlier categorical vocabulary ignores unseen categories until a scratch retrain wins.
- Checkpoint files increase run-artifact storage and SHA-256 verification adds small I/O overhead.
- The first active round cannot evaluate history-dependent paradigms.

### Neutral

- Chemprop and third-party backends continue to use scratch retraining until they implement the same
  safe export/import contract.
- Failed compatibility checks consume a trial slot and remain visible in the audit trail.

## Failure modes and controls

- **Partial round:** ignored unless `commit.json` declares the exact round complete.
- **Corrupt or replaced weights/manifest:** rejected by commit-bound SHA-256 and array-contract
  validation.
- **Path traversal:** manifest weight references must be plain adjacent filenames.
- **Oversized checkpoint:** file bytes and total array elements are bounded before use.
- **Class or feature drift:** the candidate fails closed and scratch remains mandatory fallback.
- **Outer-validation leakage:** manifests must state `outer_validation_used_for_selection=false`, and
  discovery reads inner-loop manifests only.
- **External checkpoint injection:** candidate schemas contain a source paradigm, never a path.
- **Resume drift:** source-code and configuration fingerprints continue to invalidate stale reuse.

## Alternatives Considered

**PyTorch `.pt` or `.pth` serialization**

- Rejected because conventional loading is pickle-based and is unsafe for untrusted artifacts.

**Allow the model-planning service to provide checkpoint paths**

- Rejected because it expands the secret, path-traversal, and arbitrary-file attack surface.

**Select global best with outer validation metrics**

- Rejected because repeated selection on the outer split invalidates its role as an evaluation set.

**Refit a new feature scaler and copy weights without transformation**

- Rejected because the first-layer coordinates would change and the inherited function would no
  longer have a coherent interpretation.

**Give early curriculum epochs fewer updates**

- Rejected for the matched-budget benchmark because it confounds strategy with optimizer steps.

## References

- `docs/inner-loop.md`
- `docs/reproducibility.md`
- `docs/security.md`
- ADR-0004: Local artifacts and secret boundary
- ADR-0005: Sequential training-candidate inner loop
