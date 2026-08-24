# Security and privacy guide

## Secret handling

Configuration contains environment variable names, not values. Export secrets from a shell,
CI secret store, or operating-system keychain. Never commit `.env`, provider URLs that identify a
private organization, access tokens, cookies, or raw authorization headers.

The built-in provider uses exactly `AGENTIC_AL_API_BASE`, `AGENTIC_AL_API_KEY`, and
`AGENTIC_AL_MODEL`. A YAML file cannot redirect the key field to an unrelated process credential.
API base URLs containing user information, passwords, queries, or fragments are rejected.

The LLM acquisition policy and training-candidate generator:

- requires HTTPS except for localhost;
- refuses HTTP redirects so an authorization header cannot be forwarded to a new endpoint;
- caps response size and timeout;
- does not log headers or raw responses;
- sends aggregate metrics, locally supported paradigm names, and declared numeric bounds only;
- parses strict JSON and never uses `eval`, `exec`, or shell execution;
- applies component/parameter allowlists, numerical clipping, fixed-compute locks, and deterministic
  fallback.

The training service never receives raw rows, row IDs, labels, SMILES, paths, checkpoint weights,
source code, credentials, outer-validation rows, or outer-validation metrics. It may receive a
local checkpoint's non-identifying source type, origin round, inner metric/value, digest, and
lineage count so it can choose an allowlisted continuation paradigm. Public configuration
artifacts replace dataset and output locations with `${DATASET_PATH}` and `${RUN_DIR}`. Candidate
suggestions cannot execute training code; local code converts validated choices into a model
configuration.

## Data handling

`data/` and `outputs/` are ignored because observations, molecules, labels, and run logs can be
confidential. The self-contained report contains aggregate results, but `selection.csv` and
`observations.csv` contain row identifiers and targets. Share them only after domain-specific
review.

## Checkpoints and plugins

PyTorch, Chemprop, joblib, and pickle-compatible checkpoints can execute code when loaded.
Agentic Active AutoResearch never auto-loads third-party checkpoints. Its built-in MLP continuation
accepts only current-run committed artifacts: NumPy arrays with `allow_pickle=False`, an adjacent
JSON contract, commit-bound weights/manifest SHA-256 digests, size bounds, and exact feature/model
compatibility checks. YAML and the LLM cannot supply a checkpoint path. Treat `--plugin` modules,
explicitly named `.py` plugin files, and external model files as executable software. The built-in Chemprop adapter
resolves only the `chemprop` command from PATH; YAML cannot supply an arbitrary executable.

Chemprop receives a minimal allowlist of runtime, accelerator, locale, certificate, and cache
environment variables rather than inheriting the complete parent environment. Optional experiment
tracking is disabled in that subprocess. This reduces accidental credential propagation but does
not make an untrusted executable safe.

## Release scan

Run from the repository root:

```bash
git status --short
git ls-files -z | xargs -0 rg -n -i \
  '(api[_-]?key|secret|token|password|authorization|bearer|/Users/|/Volumes/|C:\\Users\\)'
git ls-files -z | xargs -0 rg -n \
  '(sk-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9_]{20,}|AIza[0-9A-Za-z_-]{20,})'
python -m build
twine check dist/*
bandit -r src/agentic_al
pip-audit --local --skip-editable
```

Review every hit; placeholders and documentation may match by design. Also inspect Git history,
not only the working tree. Rotate a credential immediately if it was ever committed.

The pre-commit configuration also runs `detect-secrets`. A line may use its explicit allowlist
pragma only when the matched text is demonstrably an environment-variable name or a test fixture,
never to silence an unexplained value.

CI repeats static security and dependency audits, and Dependabot monitors Python and GitHub Actions
monthly. An audit is a point-in-time check, not proof that code or dependencies are
vulnerability-free. The two precise Bandit `nosec` annotations document the reviewed, shell-free
Chemprop subprocess boundary; do not add broad rule exclusions.
