# b70_SYCL_minimax_h3

A SYCL fast path for **MiniMax H3** (text/image -> video + audio) on an **Intel Arc Pro B70 (32 GB, Xe2)**.

Status: **assessment only** (2026-10-02). No engine code yet. The running PyTorch pipeline is the
reference everything here is measured against.

| document | what it covers |
|---|---|
| [docs/ASSESSMENT.md](docs/ASSESSMENT.md) | What it would take, in which order, what each step should buy, and where to stop |
| [docs/BASELINE.md](docs/BASELINE.md) | Today's measured numbers: stage times, the step-time model, where the time goes |
| [docs/LESSONS-FROM-STRATA.md](docs/LESSONS-FROM-STRATA.md) | What the Strata CUDA->SYCL port taught us that applies here |
| [bench/README.md](bench/README.md) | The comparison harness to build first |

## The short version

- H3's denoiser is a plain 50-block single-stream DiT. One step at production size (52k tokens) is 69 s, and
  **73% of it is attention**.
- There is no native H3 engine to port, as there was with Strata. But the model's own code calls a kernel
  library, **comfy-kitchen** (Apache-2.0), that has a CUDA backend and an AMD HIP backend and **nothing for
  Intel**: int8 linear, int8 ("Sage") attention, block-sparse attention, fused RMSNorm+RoPE, fused GroupNorm
  conv3d. On the B70 all of that falls back to plain PyTorch on GGUF weights.
- So the first engine to build is **a SYCL backend for comfy-kitchen**, not a from-scratch program: the same
  shape of work as the Strata port, with PyTorch still doing everything that is not hot.
- A standalone C++ engine (no PyTorch) is the second step, worth it only for what the backend cannot give:
  no per-job warm-up, no per-job model load, tighter VRAM.
- Honest expectation: **1.4-1.8x per clip**, not the 3x Strata gave, because PyTorch here already runs at
  50-70% of the card's measured peak. The gains come from int8 on XMX and from removing overhead, and both
  must pass a quality gate against the running pipeline.
