# GPU setup and troubleshooting

Formal Agentic Active AutoResearch configs set `model.require_accelerator: true`. Device resolution is CUDA first,
then Apple MPS. If neither backend is available, training stops; it does not silently switch to
CPU. The Random Forest config is intentionally a separate smoke/baseline route.

## Inspect the environment

```bash
agentic-autoresearch doctor
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.backends.mps.is_available())"
```

The final run evidence must show `"accelerator_used": true` and `"device": "cuda"` or `"mps"`
in `final_metrics.json`; each round also records the resolved device in `metrics.json`.

## NVIDIA CUDA

Install the PyTorch build that matches the host driver and CUDA platform using the current
[PyTorch installation selector](https://pytorch.org/get-started/locally/). Do not copy a wheel URL
from another machine. Confirm `torch.cuda.is_available()` before starting a formal run.

Agentic Active AutoResearch currently uses one device per local process. Multi-GPU scheduling is an extension point,
not an implied feature of the `0.1.x` release.

## Apple MPS

Use an Apple Silicon Mac and an MPS-enabled PyTorch build. PyTorch documents the backend in its
[MPS notes](https://docs.pytorch.org/docs/stable/notes/mps.html). Agentic Active AutoResearch rejects
`PYTORCH_ENABLE_MPS_FALLBACK=1` for formal runs because that setting permits unsupported operators
to execute on CPU. Remove it before running:

```bash
unset PYTORCH_ENABLE_MPS_FALLBACK
agentic-autoresearch demo --output-dir outputs/mps-check
```

## Chemprop

Install the optional packages in the same environment as Agentic Active AutoResearch and check the CLI contract:

```bash
python -m pip install -e '.[molecule,chemprop]'
command -v chemprop
chemprop train --help
chemprop predict --help
```

The adapter targets Chemprop v2 and launches it without a shell, on one resolved CUDA/MPS device.
It passes an explicit warmup schedule satisfying `0 <= warmup_epochs < epochs`; warmup and epochs
are fixed-compute fields that a training-candidate generator cannot change.
Because Chemprop and Lightning interfaces evolve, validate the pinned version on the target GPU
before a tagged benchmark. `metrics.json` records the discovered Chemprop package version.

## Out-of-memory or unsupported operations

- Reduce `batch_size`, `hidden_dims`, `ensemble_size`, or fingerprint size in that order.
- Keep the comparison budget honest when changing ensemble or MC-dropout counts.
- Start a new output directory after changing config; resume rejects a changed config by design.
- Do not disable `require_accelerator` to make a formal result finish. Use `configs/cpu_smoke.yaml`
  only when the purpose is explicitly a smoke test or classical baseline.
