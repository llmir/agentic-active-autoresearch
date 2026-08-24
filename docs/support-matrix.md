# Support matrix

This matrix separates built-in, exercised behavior from extension points. A plugin boundary is not
the same as a validated built-in backend.

## Tasks and models

| Capability | Built in | Validation status | Notes |
|---|---|---|---|
| single-target regression | PyTorch MLP, RF, Chemprop | MLP/RF end-to-end; Chemprop regression functional smoke | MAE, RMSE, R2, uncertainty/error rank correlation |
| binary classification | PyTorch MLP, RF, Chemprop | MLP/RF end-to-end; Chemprop classification contract only | accuracy, balanced accuracy, log loss |
| multiclass classification | PyTorch MLP, RF | unit and end-to-end model coverage | grouped splits retain every class on both sides |
| molecular regression/classification | Morgan + MLP or Chemprop | Morgan/MLP path covered; Chemprop regression smoke on generated molecules | Chemprop classification is binary only; no real-dataset performance claim |
| multi-output, ranking, survival, structured prediction | not built in | not claimed | requires a custom model plus task metrics/engine extension |

The public model registry supports trusted surrogate backends with `fit`, `predict`, `featurize`,
and `metadata` methods. A custom model can own its feature transformation, but version 0.1.0 does
not expose a separate featurizer registry.

## Data and split behavior

| Source or structure | Status | Contract |
|---|---|---|
| synthetic | built in, tested | regression and classification |
| CSV | built in, tested | strict IDs, targets, features, optional group/cost/risk |
| JSONL | built in, end-to-end tested | same validation contract as CSV |
| Parquet | built in, optional dependency contract tested | requires PyArrow or another pandas engine |
| database, lakehouse, laboratory system | trusted plugin | `register_dataset()` returns a pandas DataFrame |
| fully unlabeled external regression pool | Python API, tested | explicit idempotent oracle plus separate labeled `validation_data` |
| fully unlabeled external classification pool | not built in | needs an explicit seed-label/class-coverage contract |
| group/scaffold holdout | built in, tested | no group overlap; every class retained when feasible |
| Bemis-Murcko scaffold derivation | built in with RDKit | activated by `group_column: scaffold` |
| out-of-core or distributed pools | not built in | current engine materializes a pandas DataFrame |

IDs are normalized to strings and must remain unique after normalization. Group labels must be
non-missing. Numeric targets, costs, risks, and supplied numeric features reject infinities before
training.

## Acquisition and agent behavior

Built-in acquisition components are `random`, `uncertainty`, `diversity`, `target`,
`representativeness`, `group_coverage`, and its molecular alias `scaffold_coverage`. Cost and risk
are separate bounded penalties. Components can be combined by a deterministic policy, a guarded
OpenAI-compatible policy, or a trusted Python policy.

Training-candidate generation is sequential. Built-in PyTorch supports mandatory control,
scratch retraining, global-best checkpoint continuation, previous-round continuation,
matched-step curriculum learning for regression/classification, and matched-step Huber loss for
regression. The two continuation paths become available after a committed active round. Checkpoint
export/import, curriculum, and robust loss have direct unit and CPU end-to-end coverage; MPS
integration evidence is listed in [validation](validation.md). Chemprop, RF, and trusted plugins
expose only paradigms they implement.

## Runtime status

| Runtime | Status for this candidate |
|---|---|
| Apple MPS | regression and classification inner/outer loops exercised |
| NVIDIA CUDA | implemented and fail-closed, but hardware validation pending |
| CPU | explicit RF smoke/baseline; not presented as the formal GPU result |
| Chemprop v2 | 2.2.3 CLI adapter completed a one-epoch MPS regression smoke in an existing Chemprop environment; independent install, classification, CUDA, and real-dataset runs pending |

See [validation](validation.md) for exact local evidence and [benchmarking](benchmarking.md) for
the claims that require multiple seeds, matched compute, and scientific data.
