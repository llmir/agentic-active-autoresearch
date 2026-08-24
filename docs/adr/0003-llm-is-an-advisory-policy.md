# ADR-0003: Treat the LLM as an advisory policy

## Status

Accepted

## Context

LLM-driven strategies can adapt training candidates and acquisition mixtures, but raw model output
is non-deterministic, provider-dependent, and untrusted. Allowing it to execute code or change
budgets also makes baseline comparisons invalid.

## Decision

Send aggregate metrics, locally supported paradigm names, and declared numeric bounds only. Accept
one strict JSON contract for bounded training candidates and another for acquisition weights.
Restrict keys to the corresponding registry/parameter allowlist, lock model/label budgets, clip
numeric changes, and fall back to a deterministic policy on failure. Never execute generated code.
The training candidate is still trained, evaluated, selected, and refitted by local authoritative
code.

Acquisition-policy feedback uses aggregate metrics from the labeled-only inner split. Outer
validation metrics are reserved for audit/reporting and are not placed in policy context.

## Consequences

### Positive

- Provider-neutral and safe by construction.
- Agentic decisions remain inspectable and comparable.
- Offline operation is first-class.

### Negative

- The LLM cannot invent arbitrary training procedures or alter compute inside the core loop.
- More ambitious candidate paradigms require explicitly budgeted, tested backend protocols.
