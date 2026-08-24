# Maintainer release checklist

This checklist covers repository publication and tagged releases. It does not replace the
scientific validation record in [validation.md](validation.md) or the security review in
[security.md](security.md).

## 1. Obtain explicit publication approval

- Ask the project owner to approve the exact repository name, visibility, and destination GitHub
  user or organization before creating a remote or pushing.
- Never infer an account name, author identity, email address, affiliation, or API credential from
  the local machine. Add personal attribution only when its owner explicitly requests it.
- Replace every repository-owner placeholder in `README.md`, `README.zh-CN.md`, `pyproject.toml`,
  and `CITATION.cff`; then scan the public tree for unresolved owner placeholders.
- Keep the Apache-2.0 license and generic contributor attribution unless a maintainer explicitly
  approves a different, legally reviewed value.

## 2. Freeze and inspect the candidate

Run from a clean source checkout with no datasets, outputs, `.env` file, or local checkpoints in
the repository:

```bash
git status --short
git diff --check
git diff --cached --check
ruff check .
ruff format --check .
mypy src/agentic_al
pytest --cov=agentic_al --cov-report=term-missing
bandit -r src/agentic_al examples
pre-commit run --all-files
```

Follow the full secret and personal-path scan in [security.md](security.md). Review matches rather
than suppressing them: only documented environment-variable names and synthetic security fixtures
may remain. Confirm that public artifacts do not contain dataset rows, labels, SMILES, credentials,
private endpoints, personal paths, or raw provider responses.

## 3. Build and inspect distributions

```bash
python -m build
twine check dist/*
python -m zipfile -l dist/*.whl
tar -tzf dist/*.tar.gz
shasum -a 256 dist/*
```

Install the wheel into a fresh environment outside the source tree. Verify all three entry points:
`agentic-autoresearch`, the compatibility alias `agentic-al`, and `python -m agentic_al`. Run
`agentic-autoresearch doctor`, then an accelerator-backed smoke run on the target GPU. Record the
resolved device and any explicitly unvalidated hardware or backend in `validation.md`.

## 4. Publish without rewriting evidence

1. Create the initial commit only after the approval and scans above.
2. Create the GitHub repository under the approved owner with the agreed visibility.
3. Add that exact repository as `origin`; do not force-push or rewrite the reviewed history.
4. Push the default branch and wait for every required GitHub Actions job to pass.
5. Check the rendered English and Chinese READMEs, image links, issue templates, license, security
   policy, and citation metadata on GitHub.
6. Create a version tag or GitHub release only after the default-branch checks pass. Attach the
   inspected wheel/sdist and publish their SHA-256 values when binary artifacts are included.

Do not publish to PyPI, create a DOI, upload a dataset, or announce benchmark superiority as an
automatic consequence of publishing the source repository. Each is a separate approval and
evidence decision.

## 5. Post-publication checks

- Re-run the public clone quickstart in a new directory.
- Confirm private vulnerability reporting and Dependabot are enabled.
- Open a synthetic-data-only issue if a target environment exposes a reproducible defect.
- Rotate and revoke any credential immediately if it ever appears in Git history, an Actions log,
  release artifact, issue, or report.
