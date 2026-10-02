# The port: what moves where

The rule: **the arithmetic on the GPU is SYCL (C++), everything else is Rust, and the web front end is TypeScript/TSX
on snabbdom, as it was.** No Python in the engine. The Python under `reference/` drives the *existing* PyTorch
pipeline so that every ported piece can be compared against it; it goes away when the port has parity.

`done` = runs on the card and is checked against a reference. `next` = being worked on. Everything else is open.

## The pipeline, piece by piece

| piece of a clip | today (reference) | port | state |
|---|---|---|---|
| device, memory, capped allocator | PyTorch | `kernels/` `h3s_open`, `h3s_alloc` + `h3-core::device` | done |
| checkpoint reading (.safetensors) | safetensors (Python) | `h3-core::safetensors` | done |
| weight load to the GPU, read ahead in parallel | ComfyUI lazy load | `h3-core::load` | done |
| int8 linear (rotate, quantize, int8 multiply, rescale) | comfy-kitchen, PyTorch ops | `kernels/` `h3s_int8_linear` + `h3-core::ops` | done |
| per-head RMS norm + position rotation (on q and k) | comfy-kitchen, 22.7 ms | `h3s_rms_rope`, 6.2 ms at 16.5k tokens | done |
| attention (scores, softmax, weighted sum) | PyTorch -> oneDNN's fused kernel, 102 ms | `h3s_attention`: oneDNN 3.12's fused kernel made fail-safe, 96 ms; our own bounded form behind it (below) | done |
| gated activation between the MLP's two linears | PyTorch | `h3s_swiglu` (its own pass; folding it into the quantizer is open) | done |
| norm, scale+shift, gated residual add | PyTorch | `h3s_rms_norm_mod`, `h3s_gate_add` | done |
| the 50 denoiser blocks, with their per-step tables and position rotations | ComfyUI `ldm/minimax/model.py` | `h3-core::dit` | done (the 2 text refiner blocks: open) |
| time embedding, patch embedding, final layer, token layout | ComfyUI | `h3-core::dit` | open |
| sampler (Euler, 8 steps) and PyTorch-compatible noise | ComfyUI | `h3-core::sampler` | open |
| LoRA applied to int8 weights | ComfyUI | `h3-core::lora` | open |
| latent upscaler (3-D convolutions) | ComfyUI | SYCL conv kernels + Rust graph | open |
| video decoder (3-D convolutions, group norm) | ComfyUI | SYCL conv kernels + Rust graph | open |
| audio decoder | ComfyUI | SYCL kernels + Rust graph | open |
| keyframe / reference image encoders | ComfyUI | SYCL kernels + Rust graph | open |
| text encoder (Qwen3-VL 32B, hidden states) | ComfyUI-GGUF | reuse the int8 linear + attention kernels | open |
| tokenizer, prompt template | Python | Rust | open |
| mp4 writing | ffmpeg subprocess | ffmpeg subprocess from Rust | open |
| job queue, HTTP API, LLM mode switching | `server.py` | `engine/h3-wfe` (Rust) | open |
| web front end | TSX + snabbdom | `wfe/` unchanged | done (needs the Rust server) |

## Order

By what a clip's time is made of (docs/BASELINE.md): the denoiser is ~75% of a production clip and attention is ~70%
of the denoiser at production size. So:

1. **The denoiser, complete, in Rust + SYCL**, checked step by step against tensors dumped from the reference
   (`reference/h3x.py`, `H3X_DUMP_STEPS`). This is where the time is, and it needs no part of PyTorch once the
   conditioning and the starting noise are read from files.
2. **Attention** inside it: oneDNN's fused attention first (what PyTorch reaches), then our own kernel, then the
   block-sparse form the model family was trained to tolerate - each with a clip to look at.
3. **The decoders and the upscaler** (convolutions): ~45 s of a clip today.
4. **The text encoder**: ~54 s per new prompt today, cached per prompt.
5. **The server**: the Rust replacement of `server.py`, keeping the API the front end already speaks, and keeping the
   model resident between clips (no reload per job).

## Where the block stack stands (2026-10-02)

`h3 check-block` runs all 50 blocks from Rust on a dump of the reference (2,159 tokens): every stage of block 0 agrees
to cosine >= 0.99985, the stream after 25 blocks to 0.99998, after all 50 to 0.9975. The reference itself computes in
bfloat16 throughout; this engine keeps intermediate arithmetic in float32 or IEEE half and rounds once.

`h3 bench-blocks`, per block:

| stage | reference (PyTorch), 16.5k tokens | this engine, 16.5k | this engine, 47k (a 15 s clip) |
|---|---:|---:|---:|
| the four int8 linears (+ gated activation) | 70 ms (with our kernel plugged in; 145 ms without) | 67 ms | 194 ms |
| per-head norm + rotation | 22.7 ms | 6.2 ms | 20.8 ms |
| norms, scale/shift, gated adds | ~6 ms | 10.5 ms | 28.1 ms |
| attention | 102 ms | 96 ms | 759 ms |
| **the block** | 222 ms as it runs in production (186 ms with our linear kernel plugged in) | **177 ms** | **998 ms** |
| **50 blocks = one denoiser step** | 11.1 s as it runs in production (9.3 s with our linear kernel plugged in) | **8.95 s** | **49.9 s** (reference 58-63 s) |

At 47k tokens the whole step takes 27.8 of the engine's 30 GiB cap: 18 GiB of weights, the rest work buffers. It
fits since the int8 linear works in row chunks of 8192 (its scratch was 3.4 GiB, now 0.6). The MLP's inner buffer
(2.7 GiB) is the next one to shrink the same way, if room is needed.

### Attention: how it got there

1. oneDNN's separate primitives (multiply, softmax, multiply) in row chunks: correct, memory bounded, **1208 ms** at
   16.5k tokens - oneDNN's stand-alone softmax took 1.0 s of it on rows that long.
2. Our own softmax passes between oneDNN's two multiplies (a polynomial exponential, the row sums delivered by the
   matrix engine through a column of ones beside the values): **559 ms**. What remained was memory traffic over the
   score table - written once, read twice, rewritten once.
3. oneDNN's *fused* attention kernel, which never writes the score table: **96 ms**. It is reachable only through
   oneDNN's graph interface, only from oneDNN 3.12 on, and only for one exact form of the pattern (16-bit q, k, v;
   the two tensors in between declared float32; the scale a float32 host scalar; softmax mode `inf_as_zero`) - found
   with `kernels/sdpa_probe.cpp` and by reading oneDNN's source. Any other form silently runs as separate steps with
   the **whole** score table in device memory: 30 GiB at 16.5k tokens, which on this driver (no out-of-memory error)
   took the machine down once during this work. So the image builds oneDNN 3.12 from source with one small patch
   (`container/onednn-sdpa-no-fallback.patch`): on request, a pattern the fused kernel does not take is an error,
   not a fallback. Then form 2 above, whose memory is bounded by construction, takes over.

Next on attention: it is still ~75% of a production step. The fused kernel runs at ~165 G scores/s; beyond that it
takes computing fewer scores (the block-sparse form the model family was trained to tolerate), with clips to judge.

## Checks that every piece must pass

- against the CPU reference in `h3-core::reference` (exact arithmetic, small sizes);
- against the PyTorch pipeline's dumped tensors (real sizes);
- a clip pair for anything that can change the picture or the sound, looked at and listened to, not only measured.
