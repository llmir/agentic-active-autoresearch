# Security policy

## Supported versions

Security fixes are applied to the latest minor release on the default branch.

## Reporting a vulnerability

Before a public repository exists, report privately to the project maintainer.
After publication, use GitHub's private vulnerability-reporting feature. Do not
open a public issue containing credentials, private datasets, or exploit details.

## Trust boundary

- API credentials are read only from environment variables named in config.
- The built-in provider fixes those names to `AGENTIC_AL_API_BASE`, `AGENTIC_AL_API_KEY`, and
  `AGENTIC_AL_MODEL`; config cannot select an unrelated environment credential.
- Dataset rows are never sent to an LLM policy; only aggregate run statistics and declared numeric
  training bounds are sent.
- LLM output is treated as untrusted data and parsed as strict JSON. Training candidates are
  constrained to bounded tunables while compute/architecture stay locked; acquisition suggestions
  are constrained to registered components and clipped against an anchor. Neither is executed as
  code.
- `--plugin` imports arbitrary Python and must be used only with trusted modules or explicitly
  reviewed `.py` files.
- Chemprop is launched with an argument array rather than a shell, and its subprocess receives a
  minimal runtime/accelerator environment allowlist instead of the parent environment.
- Model checkpoints and pickle-like files are untrusted executable data. Agentic Active
  AutoResearch never auto-loads third-party checkpoints. Built-in MLP continuation accepts only
  current-run committed, SHA-256-verified NPZ arrays loaded with `allow_pickle=False`; configuration
  and LLM output cannot provide paths.

Run the release scan described in [docs/security.md](docs/security.md) before publishing artifacts.
