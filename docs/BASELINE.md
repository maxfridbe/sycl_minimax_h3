# Baseline: MiniMax H3 on the B70 today (PyTorch XPU)

Measured on the Arc Pro B70 (32 GB, PCIe 3.0 x8 slot, Ryzen 7 1700X, 23 GiB RAM), image `h3-xpu:3`
(torch 2.13 XPU, ComfyUI's H3 model code used as a library, ComfyUI-GGUF weights, comfy-kitchen 0.2.34 on its
`eager` backend). Sources: the film pipeline's job records and the earlier kernel study (`h3cli/kernel/PLAN.md`,
2026-09-17/19).

## The model

| part | what | size on disk |
|---|---|---|
| text encoder | Qwen3-VL-32B truncated to 50 layers, last hidden state (5120-d) | 18.2 GB (Q4_K_M GGUF) |
| denoiser | single-stream DiT: 50 blocks + 2 refiner blocks, hidden 5376, 56 heads x 128, SwiGLU MLP (5376 -> 28672, 14336 -> 5376), RMSNorm, AdaLN (pruned "curve" form), 3-axis RoPE; video + audio + text tokens in one sequence | 21.4 GB (Q8_0, what the films use), 16.6 GB (Q6_K), 11.4 GB (Q4_K_M) |
| video VAE | causal Conv3d encoder + ViT-3D decoder | 5.2 GB (fp16) |
| audio VAE | Conv1d / Snake | 0.6 GB (fp32) |
| latent upscaler | 3D conv | 0.7 GB (bf16) |

Sampling is Euler, cfg 1 (one forward per step), 8 steps with the turbo LoRA.

## One production clip (896x672, Q8_0, 8 steps, ~15 s of video, 362 frames)

From a real job record (seconds):

| stage | s | share |
|---|---|---|
| text encoder load + conditioning | 57 | 7% |
| keyframe encode (video + audio VAE) | 13 | 2% |
| sampling, steady state (7 steps x 63.5 s, and the first step's own 63.5) | 508 | 60% |
| **first-step extra** (step 1 takes 187 s, not 63.5) | **124** | **15%** |
| latent upscale | 11 | 1% |
| audio decode | 8 | 1% |
| video VAE decode | 90 | 11% |
| mux | 6 | 1% |
| process start, loads, other | 28 | 3% |
| **total** | **845** | |

The first-step extra is not a one-off: three consecutive 14-15 s clips paid 122, 124 and 116 s, and a 4.5 s clip
(107 frames, 11.7 s per steady step) paid 115 s - more than its other seven steps together. Two things
hide in it and have not been separated yet: the denoiser's weights are loaded lazily ("DiT loaded 0.2 s" is not
the 21 GB reaching the card - that happens inside step 1, on a 23 GiB host that cannot keep the file cached),
and every clip has its own frame count, so every clip is a new latent shape for `torch.compile` and the kernel
caches. Separating the two is the first measurement in ASSESSMENT.md.

Throughput: ~70-76 s of GPU per second of video, ~4.3 Wh per second of video, the card at its 275 W cap while
sampling.

## The step-time model

Fitted on 21 measured cells (2026-09-19), worst error 3.7%:

    tokens N = latent_frames * (W*H / 1024) + round(frames/24*40) + 10 + 336
    s/step   = 1.829e-8 * N^2  +  3.463e-4 * N
    peak GiB ~= weights + 1.64e-4 * N        (cap 0.95 x 31.89 GiB)

What the two terms are, from the architecture:

| term | work per step | measured rate | card's measured bf16 GEMM peak | headroom |
|---|---|---|---|---|
| N^2 (attention: 50 blocks x 4 * 56 * 128 * N^2 FLOP) | 1.43e6 * N^2 FLOP | **78 TFLOPS** | 155 TFLOPS | 2.0x |
| N (linears: 38.5 GFLOP per token over 50 blocks) | 3.85e10 * N FLOP | **111 TFLOPS** | 155 TFLOPS | 1.4x |

At production size (N = 51,992): attention 49.4 s + linears 18.0 s = 67.4 s (measured 69.4). **Attention is 73%
of a step.** At 640x480 / 10 s (N = 22,649) it is 44%.

## What was already tried (and closed)

- **Fused Q4_K dequant + GEMM in Triton**: correct, 6x slower than dequant-then-GEMM. Dequant + elementwise is
  ~15% of a Q4 step and less with Q8_0.
- **A Triton flash attention**: oneDNN's `micro_sdpa` already runs at ~92 TFLOPS on a 22k-token call; a
  bf16 kernel of our own would at best match it.
- `torch.compile`: 7%.
- MATH / EFFICIENT attention backends: materialize the N x N scores (28.8 GiB at N = 11.7k); they evict VRAM
  into host RAM and hang the box. Never above N = 6k.

So a **bf16-for-bf16** rewrite in SYCL buys little. Everything in ASSESSMENT.md follows from that.

## What the model's own code offers that the B70 does not use

`comfy-kitchen` (Comfy-Org, Apache-2.0) is the kernel library ComfyUI's H3 code calls. Backends in 0.2.34:
`cuda` (61 kernels, compiled), `hip` (52, compiled, AMD), `triton`, `eager` (PyTorch). Selection order is
cuda > triton > eager; on XPU everything lands on `eager`.

Kernels H3 touches:

| call site | kernel | on NVIDIA | on the B70 today |
|---|---|---|---|
| every Linear of an `int8_convrot` checkpoint | `int8_linear` (int8 x int8 GEMM, rotated weights) | cuBLASLt int8 | not used: we load GGUF and run bf16 |
| DiT attention | `int8_attention` ("Sage": int8 Q.K^T) | CUDA | oneDNN bf16 SDPA |
| DiT attention, optional | `sol_attn` (block-sparse top-k) | CUDA | unavailable |
| q/k norm + RoPE | `rms_rope_split_half_` (fused, in place) | CUDA | eager (several passes) |
| AdaLN | `adaln`, `rms_adaln` | CUDA | eager |
| MLP | fused SwiGLU into the next linear (`linear_input_act`) | CUDA | eager |
| video VAE | `group_norm_silu_pad3d`, `fp16_conv3d`, int8 attention in the ViT decoder | CUDA | eager / oneDNN |

Upstream's own guidance for H3: "prefer `int8_convrot`" for the denoiser, the text encoder and the video VAE.
The B70 runs none of that path.
