# Public API

| Symbol | Purpose |
|---|---|
| `load_config(path)` | Parse and validate strict YAML. |
| `run_experiment(config, ...)` | Execute or resume a closed loop; optional `oracle` + `validation_data` enable a fully unlabeled regression pool. |
| `TableOracle` | Reveal labels from a benchmark table. |
| `CallableOracle` | Wrap a trusted experiment callback. |
| `register_model(name, factory)` | Add a surrogate backend. |
| `register_acquisition(name, function)` | Add a score component. |
| `register_dataset(name, loader)` | Add a trusted DataFrame-producing dataset loader. |
| `register_training_candidate_generator(name, factory)` | Add a bounded sequential inner-loop generator. |

`TrainingCandidate` and `TrainingCandidateContext` live in `agentic_al.types`. A generator can
propose plans but cannot bypass local validation, fixed-compute locks, inner validation, or the
final all-label refit.

Checkpoint discovery is intentionally engine-owned rather than a public path argument. Built-in
PyTorch continuation resolves current-run committed lineage automatically; external callers and
candidate generators cannot inject a filename.

The stable data contracts live in `agentic_al.types`. Components should depend on those contracts
rather than engine internals. Version `0.1.x` remains a release-candidate API; breaking changes are
documented in the changelog and ADRs.

`validation_data` is a labeled pandas DataFrame supplied through the Python API, never a config
path. It is required with a fully unlabeled external regression pool and must be ID/group-disjoint
from that pool. External classification seed-label semantics are not part of version 0.1.0.
