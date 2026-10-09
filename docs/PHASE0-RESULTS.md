# Phase 0 results (2026-10-02)

Measured on the B70 with the PyTorch comparison harness (not part of this repository). Clips: the bakery prompt, 768x576, 5 s (124 frames, about 16.5k
tokens, ~11 s per steady step), seed 0.

## 1. Matrix multiply on the card: int8 against 16-bit float (oneDNN, `kernels/gemm_bench.cpp`)

Rates in trillions of multiply-adds per second.

| shape | f16 | bf16 | int8 -> int32 | int8 -> f16 | int8 gain |
|---|---|---|---|---|---|
| linear 5376 -> 21504 (attention's input projection) | 178 | 181 | 317 | 335 | 1.8-1.9x |
| linear 5376 -> 28672 (MLP in) | 178 | 181 | 317 | 340 | 1.8-1.9x |
| linear 14336 -> 5376 (MLP out) | 180 | 183 | 354 | 357 | 2.0x |
| linear 7168 -> 5376 (attention's output projection) | 178 | 181 | 323 | 315 | 1.8x |
| attention scores: 56 heads x [T x 128] . [128 x T], T = 2048 / 4096 | 92 / 95 | - | 35 / 33 | 81 / 85 | **none** |
| attention weights times values: 56 x [T x T] . [T x 128], T = 2048 / 4096 | 131 / 141 | - | - | - | - |
| softmax over 56 x [T x T] scores | 229 G scores/s (f16, T = 2048), 156 G scores/s (f32, T = 4096) | | | | |

- **The linear layers pass the gate** (>= 1.5x): int8 is 1.8-2.0x. PyTorch today reaches ~111 of the 180 available.
- **Attention scores do not get faster with int8.** A score is only 128 multiply-adds and one output value: the
  work is bounded by writing the T x T table, and an int32 table is twice the bytes of an f16 one.
- **Attention is three passes of similar cost.** Per score: scores 2.8 ps, softmax 4.4-6.4 ps, weights-times-values
  1.9 ps - about 100 G scores/s if done as three separate primitives. PyTorch's fused attention (oneDNN
  `micro_sdpa`) already does **153 G scores/s** on a production step. A fused kernel of our own, with both matmuls
  at the card's full 180 and the softmax in registers, tops out around 180-190 G scores/s: **1.2x at best.**
  The gate (>= 1.3x) fails. Dense attention is near its floor on this card; the September study's verdict stands.

## 2. The first-step extra, split (3-step clips)

| run | weight load | sampling (3 steps) |
|---|---|---|
| default (today): lazy load + `torch.compile` | inside step 1 | 147.5 s |
| weights loaded before step 1 | 50.4 s | 79.6 s |
| loaded first, `torch.compile` off | 55.3 s | 43.9 s (10.3 / 11.6 / 11.1 / 10.9 s) |

- **~50-55 s is the denoiser's 20 GB reaching the card** at ~0.4 GB/s (one tensor at a time from a file the
  23 GiB host cannot keep cached). The same card took 23 GiB in 19 s with the Strata fill pipeline.
- **~36 s (more on a new shape) is `torch.compile`**, with its cache warm, and at this size the compiled steps
  are no faster than the uncompiled ones. At production size its 7% roughly pays its own cost back; it never
  comes out ahead on an 8-step clip.
- Upstream's int8 file loads in **37 s** (same path, safetensors instead of GGUF).

Neither needs a SYCL kernel to fix.

## 3. Quality reference: GGUF Q8_0 against upstream's int8 denoiser (8 steps, PyTorch path for both)

| | Q8_0 (today) | upstream int8 (`int8_convrot`), kitchen `triton` + `eager` backends |
|---|---|---|
| weight load | 55.3 s | 36.7 s |
| steady step | 11.0 s | 11.9 s |
| clip total (conditioning cached) | 202 s | 186 s |

Step-by-step agreement of the two denoisers' estimates (1.0 = identical direction):

| step | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| video | 0.995 | 0.984 | 0.981 | 0.975 | 0.976 | 0.975 | 0.972 |
| audio | 0.994 | 0.998 | 0.996 | 0.993 | 0.884 | 0.876 | 0.910 |

Finished frames: 24.0 dB PSNR between the two (for scale: the *same* Q8_0 weights with and without
`torch.compile` are 31 dB apart after 3 steps). To the eye: the same scene, framing and light; details differ
(where the loaf sits, the sign, the lamp). The audio diverges more than the picture from step 6 on - to be
judged by ear.

So the int8 checkpoint is usable on the B70 **today**, at Q8_0's speed, without any kernel work: its int8
matmuls just are not reaching the matrix engine (11.9 s/step where the multiply rates above say ~9.5 s).

## 4. What this changes in the assessment

For a production step (N = 47k: attention 40.7 s, linears 16.3 s, 57 s):

| lever | saves per step | needs |
|---|---|---|
| int8 linears on the matrix engine (1.85x) | ~7 s (12%) | one kernel: `int8_linear` for comfy-kitchen, built on oneDNN's int8 matmul |
| a fused attention kernel of our own | <= ~6 s, uncertain | the hard kernel; oneDNN is already within 1.2x of the ceiling |
| block-sparse attention (fewer scores, not faster ones) | the only lever on the 71% | research; picture quality unknown |

Per clip (845 s): the first-step extra (124 s) and the per-clip process overhead are worth more than every
dense kernel together, and need no kernels:

| change | saves per clip | effort |
|---|---|---|
| drop `torch.compile` | 35-70 s | a flag |
| load the denoiser with a pipelined, pinned loader (or keep it resident between clips) | 35-55 s | small |
| int8 linears on the matrix engine | ~55 s (8 steps x 7 s) | one kernel + parity tests |
| a resident worker (no process start, text encoder and denoiser staged once per series) | 30-60 s | medium |

Realistic total: **845 s -> ~620-650 s per clip (1.3-1.35x)**, most of it without SYCL. A from-scratch SYCL engine
would add little beyond that unless sparse attention works.
