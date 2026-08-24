# ADR-0001: Use a protocol-first modular monolith

## Status

Accepted

## Context

The research prototype coupled QM9 preparation, Chemprop, a specific provider, experiment
orchestration, and visualization. Open-source users need a runnable local core plus replaceable
domain components without the operational cost of services or a plugin framework daemon.

## Decision

Ship one Python package and process. Define narrow Python protocols/registries for models,
training-candidate generators, acquisition functions, policies, datasets, and oracles. Keep
generated artifacts on disk in a documented schema.

## Consequences

### Positive

- Easy local installation, debugging, and reproducibility.
- Domain-specific components can be added without forking the engine.
- No server, database, or cloud account is required.

### Negative

- One process limits distributed training throughput.
- Python plugins are trusted code and require an explicit `--plugin` import.

## Alternatives considered

- Microservices: rejected because they add deployment and secret-management complexity before scale demands it.
- Keep the prototype layout: rejected because dataset/provider coupling blocks reuse.
