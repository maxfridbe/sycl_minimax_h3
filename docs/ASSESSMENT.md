# What it would take to give MiniMax H3 a SYCL engine on the B70

2026-10-02. Based on the measured baseline (BASELINE.md) and on what the Strata port taught us
(LESSONS-FROM-STRATA.md).

## 1. How this differs from Strata

| | Strata (LLM) | MiniMax H3 (video) |
|---|---|---|
| existing optimized engine | a complete CUDA engine, ported with dpct | none; the model runs through PyTorch |
| what made the port faster than the alternative | algorithms the alternative lacked: speculative decoding, a sparse expert cache | nothing comparable: every token goes through every block |
| where PyTorch / llama.cpp stood | far from the hardware's limit | attention at 78 TFLOPS, linears at 111, against a measured bf16 peak of 155 |
| work shape | sparse, latency-bound, many small kernels | dense, throughput-bound, two kernels are 90% of a step |

So "rewrite it in SYCL" is not by itself a speed-up here. A bf16 kernel of ours against oneDNN's bf16 kernel
is a draw at best (measured in September). The speed has to come from three places PyTorch-on-XPU leaves on
the table:

1. **int8 on XMX.** The B70's matrix engine is built to do int8 at twice the element rate of bf16 (to be
   measured in Phase 0). Upstream's fast path
   for H3 is exactly that (int8 weights, int8 attention scores) and has no Intel implementation.
2. **Per-clip overhead.** 124 s of every clip's 845 s is first-step extra (lazy weight load + compile for a new
   latent shape), 57 s is the text encoder, ~28 s is process start. A resident engine with ahead-of-time
   kernels pays none of the first, and the Strata load pipeline moves 23 GiB to the card in 19 s.
3. **Attention that does less work.** Block-sparse attention exists upstream (CUDA only). It is the only lever
   that changes the N^2 term itself, and it is a quality question, not only a porting one.

## 2. What there is to port: comfy-kitchen

The model's own code calls `comfy-kitchen` (Comfy-Org, Apache-2.0, github.com/Comfy-Org/comfy-kitchen). It has
a CUDA backend (~1.3 MB of kernel source: `ops/` 25 files, `sage_attention/` 18 files) and a hand-written AMD
HIP backend (~0.7 MB, with its own matrix-core GEMM headers). The registry picks cuda > triton > eager; a
`sycl` backend slots in the same way. That is the Strata situation again, with one difference that helps:
**PyTorch stays** for everything that is not hot (tokenizer, sampler loop, VAE glue, file formats), so the
port is a handful of kernels, not a program.

The kernels H3 needs, in the order of what they are worth:

| kernel | share of a production step today | what a SYCL version must do | difficulty |
|---|---|---|---|
| `int8_attention` (Sage) | ~70% | int8 Q.K^T on XMX, fp16 P.V, online softmax, tiled; never an N x N buffer | **hard** - the project's main risk |
| `int8_linear` (+ ConvRot weights) | ~25% | int8 x int8 GEMM on XMX with per-row scales; oneDNN's int8 matmul is the first candidate, a `joint_matrix` kernel the second | medium |
| `rms_rope_split_half_`, `adaln`, `rms_adaln`, fused SwiGLU | ~2-5% (elementwise passes) | straight ports, one pass over the tokens each | easy |
| `group_norm_silu_pad3d`, `fp16_conv3d`, int8 attention in the ViT decoder | video decode, 11% of a clip | ports + oneDNN conv3d | medium |
| `sol_attn` (block-sparse top-k) | changes the N^2 term | port after the dense path works; Strata's QSA block-select kernels are the same idea | hard, and quality-gated |

It also means switching the denoiser from GGUF Q8_0 to upstream's `int8_convrot` checkpoint (the format the
int8 kernels read), which changes the output slightly. That is one of the quality gates below.

## 3. What a perfect result would be (ceilings, not promises)

A large step, N = 51,992 tokens (896x672, 12 s), today 69.4 s (production's 47k-token step scales the same way):

| | attention | linears | step | vs today |
|---|---|---|---|---|
| today (PyTorch, bf16, GGUF Q8_0) | 49.4 s | 18.0 s | 69.4 s | 1.0x |
| bf16 at the card's measured GEMM peak (155 TFLOPS) | 25.0 s | 12.9 s | 37.9 s | 1.8x |
| int8 Q.K^T + fp16 P.V, int8 linears, each at its XMX peak | ~18.8 s | ~6.5 s | ~25 s | 2.7x |

Real kernels land at 60-75% of their ceiling (oneDNN's own attention is at 50% of the bf16 one). A realistic
target for the dense path is **a 40-45 s step: 1.5-1.7x on sampling.**

Per clip (845 s today):

| | today | kitchen SYCL backend | + resident engine |
|---|---|---|---|
| sampling, steady | 508 | ~320 | ~320 |
| first-step extra | 124 | 124 (partly less: kernels are AOT, weights still lazy) | ~20 |
| text encoder | 57 | ~40 (int8 linears) | ~15 (resident or cached, Strata-style kernels) |
| video decode | 90 | ~60 | ~60 |
| everything else | 66 | 66 | ~45 |
| **total** | **845** | **~610 (1.4x)** | **~460 (1.8x)** |

Block-sparse attention could take the sampling line further, but only if the pictures hold up; it is not
counted here.

## 4. The plan, with a gate after every phase

Each phase ends in a measurement against the running PyTorch pipeline. A failed gate stops the project there
with what was learned written down - the September kernel study did exactly that and saved weeks.

**Phase 0 - measure first (2-3 days, no engine code).**
- Split the 124 s first-step extra into lazy weight load, `torch.compile`, and SYCL kernel JIT.
- Per-op profile of one production step (N = 52k, Q8_0); the September profile was 640x480 / Q4.
- Run the `int8_convrot` checkpoint through the `eager` backend once, for a quality reference.
- Three standalone micro-benchmarks on the card, at real shapes:
  1. int8 GEMM on XMX (oneDNN matmul, then a `joint_matrix` kernel) against bf16 GEMM;
  2. tiled attention with int8 scores against oneDNN `micro_sdpa`;
  3. the same with fp16 scores, to know what our own kernel costs before the int8 gain.
- **Gate:** int8 GEMM >= 1.5x bf16 GEMM, and int8-score attention >= 1.3x oneDNN at N = 52k. If neither, stop:
  the remaining wins (overhead only, ~15%) do not need an engine, they need a resident PyTorch process.

**Phase 1 - comfy-kitchen SYCL backend, DiT only (2-3 weeks).**
- `backends/sycl` beside `cuda` and `hip`: registry, DLPack bindings (USM pointers from XPU tensors, no
  copies), CMake with the AOT flags the Strata port settled on.
- `int8_linear`, `rms_rope_split_half_`, `adaln` / `rms_adaln`, fused SwiGLU, then `int8_attention`.
- Each kernel gets a parity test against the `eager` backend before it is wired in (bit-exact where the math is
  integer, a stated tolerance where it is not).
- **Gate:** per-step latents within tolerance of the eager int8 path, a blind A/B of finished clips against
  the current Q8_0 films, and a step time under 50 s at N = 52k.

**Phase 2 - the rest of the clip (1-2 weeks).**
- Video VAE kernels; text encoder on the int8 path.
- Remove the first-step extra inside PyTorch as far as it goes: pre-load the weights with the Strata fill
  pipeline, drop `torch.compile` once the hot ops are AOT kernels.
- **Gate:** a whole clip under 620 s.

**Phase 3 - a resident engine (3-5 weeks, optional).**
- A C++/SYCL process that keeps the denoiser on the card between clips and runs the sampler itself: the
  DiT forward is ~10 op types, all of them existing as kitchen kernels by then. The text encoder and the VAEs
  either stay in a PyTorch sidecar or move in later.
- The WFE's GPU scheduling (renders evict the LLM) stays as it is; the engine is one more thing it starts.
- **Gate:** a whole clip under 500 s, and a second clip of a series starting within seconds.

**Phase 4 - block-sparse attention (open-ended, research).**
- Port `sol_attn`; measure quality at 8 steps. The model we run has no gate-compress weights, so only the
  top-k variants apply.

## 5. Risks, most serious first

1. **XMX through SYCL `joint_matrix` underperformed in Strata.** Our XMX prompt-attention kernel was correct
   and ran at 0.4-0.5x of the plain kernel it was meant to replace. oneDNN reaches the matrix engine with its
   own JIT, not through SYCL. Phase 0 exists to find out which route gets int8 to its peak; if only oneDNN
   does, the "kernels" become compositions of oneDNN primitives and the attention tiling is ours.
2. **int8 quality.** Sage attention and ConvRot int8 are upstream's recommended path, so the risk is modest,
   but the films were tuned on GGUF Q8_0. The A/B is part of every gate.
3. **VRAM.** int8_convrot weights are about the size of Q8_0 (the bf16 file is 37 GB and does not fit). The
   `xe` driver has no out-of-memory: an overshoot evicts VRAM into host RAM and hangs the box. Every
   benchmark runs under the allocator cap, and nothing may materialize N x N.
4. **Upstream moves.** comfy-kitchen and ComfyUI's H3 code change often. A backend inside their structure
   (like the Strata port in `sycl/`) keeps merges mechanical; a private fork of the model code would not.
5. **The Level Zero v2 adapter** mishandles a host wait on an event another queue also waits on (it hung
   Strata's prompt path twice). Kernels here should take one queue and no cross-queue events.
6. **It may simply not be worth it.** If Phase 0 shows int8 on XMX at less than 1.3x, the honest outcome is
   a resident PyTorch worker (removing ~150 s of per-clip overhead, 1.2x) and no engine.

## 6. What the running pipeline gives us for comparisons

- `h3.py` caches latents (`.latents.pt`) and decoded frames, and can re-decode or re-mux from them: the same
  latents can be decoded by old and new VAE paths, and the same noise and conditioning can be fed to old and
  new denoisers.
- The conditioning cache means the text encoder can be compared tensor for tensor.
- Every job writes stage timings and energy; the step-time model predicts any shape to within 4%, so a
  measured step is immediately "x% under the model".
- Parity ladder, cheapest first: one kernel against `eager`; one DiT block; one full step (latent cosine and
  PSNR per step); a finished clip (frame PSNR/SSIM, audio level); a blind A/B of two films.

## 7. Effort, in one line each

- Phase 0: days, and it decides everything else.
- Phases 1-2 (the backend): about a month of the kind of work the Strata port was, for ~1.4x per clip.
- Phase 3 (resident engine): another month, for ~1.8x per clip and near-instant clip-to-clip starts.
- Phase 4: unknown; the only path beyond 2x.
