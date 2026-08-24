# ADR-0002: GPU-first formal training with an explicit CPU baseline

## Status

Accepted

## Context

Neural molecular models and repeated uncertainty estimation are computationally expensive.
At the same time, contributors and CI need a fast path that can validate the loop without a GPU.

## Decision

Use PyTorch GPU training as the default model path, resolving `auto` as CUDA, then Apple MPS,
then CPU. Formal configs set `require_accelerator: true`, which turns a CPU resolution into an
error. Retain Random Forest only as a clearly labeled smoke/baseline path. Keep Chemprop v2 as
an optional GPU molecular backend.

## Consequences

### Positive

- Formal experiments use the intended accelerator.
- macOS and CUDA systems share one configuration.
- CI remains practical through an honest CPU baseline.

### Negative

- PyTorch is a core dependency.
- CUDA and MPS can still differ numerically; artifacts record the resolved device.

## Alternatives considered

- CPU-first default: rejected for formal model training.
- CUDA-only: rejected because it excludes Apple Silicon development.
