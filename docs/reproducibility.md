# Reproducibility

Agentic Active AutoResearch fixes the data split, initial labels, model seeds, DataLoader generator, acquisition
randomness, and score tie-breaking. It records config/data hashes, dependency versions, resolved
device, model backend, and every selected ID.
The resume contract also hashes the installed Agentic Active AutoResearch Python sources. For an external provider,
it hashes the endpoint/model identity and whether a key is configured without storing or hashing
the key itself. Changing code, endpoint, model, or offline/online availability refuses resume.
Each inner trial records its sanitized candidate, locked parameters, resolved plan, selection
metric, deterministic seed, data-contract hash, result, and reflection. A matching completed result
can be reused, but the selected plan is always refitted on all current labels.
Built-in PyTorch checkpoints add a JSON array/feature/architecture contract, non-pickle NPZ
weights, SHA-256, origin round, compatible inner-selection evidence, and cumulative fit lineage.
Resume verifies committed checkpoint evidence before any historical continuation is proposed.
Curriculum resampling keeps the same optimizer-step count as standard training.

GPU arithmetic is not guaranteed bit-identical across CUDA, MPS, driver, PyTorch, or Chemprop
versions. Treat the environment manifest as part of the experiment and compare multiple seeds.

Recommended release protocol:

1. Freeze the environment with `python -m pip freeze` outside the public repository or create a
   reviewed lock file for the target platform.
2. Run at least five seeds with identical budgets.
3. Preserve raw run directories in private artifact storage.
4. Publish aggregate tables plus config/data hashes and a data-access statement.
5. Re-run the release secret/PII scan before sharing artifacts.

Resume is for interruption recovery. It refuses a changed config or dataset and requires the local
observation ledger plus complete inner-loop evidence for committed rounds. A new method, budget,
device policy, candidate generator, tunable range, or dataset deserves a new run directory.
Trusted plugin source is outside the built-in source hash; record and pin its package version or
commit separately.
