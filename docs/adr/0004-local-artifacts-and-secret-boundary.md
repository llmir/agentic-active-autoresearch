# ADR-0004: Keep artifacts local and secrets outside configuration

## Status

Accepted

## Context

Research runs can contain private molecules, labels, paths, endpoints, and credentials. A public
repository must not mix these with source code.

## Decision

Ignore `data/`, `outputs/`, checkpoints, and `.env*` except `.env.example`. Store only environment
variable names in YAML. Redact user-home paths and secret-like strings from metadata. Do not log
authorization headers or raw LLM responses. Require a release-time secret/PII scan.

## Consequences

Users must move run artifacts deliberately when sharing them. Reproducibility depends on hashes
and redacted configs rather than bundling private datasets.
