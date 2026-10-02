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
| attention (scores, softmax, weighted sum) | PyTorch -> oneDNN's fused kernel, 102 ms | `h3s_attention`: correct and memory-bounded, but 1.2 s at 16.5k tokens (below) | next |
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
to cosine >= 0.99985, the stream after 25 blocks to 0.99997, after all 50 to 0.997. The reference itself computes in
bfloat16 throughout; this engine keeps intermediate arithmetic in float32 or IEEE half and rounds once.

`h3 bench-blocks` at 16.5k tokens (a 5 s clip), per block:

| stage | reference (PyTorch) | this engine |
|---|---:|---:|
| the four int8 linears (+ gated activation) | 70 ms (with our kernel plugged in; 145 ms without) | 64 ms |
| per-head norm + rotation | 22.7 ms | 6.2 ms |
| norms, scale/shift, gated adds | ~6 ms | 8.1 ms |
| everything except attention | ~99 ms | 78 ms |
| attention | 102 ms | **1208 ms** |

Attention is the open problem, and the whole clip's biggest cost. Built from oneDNN's separate primitives it is
correct and its memory is bounded, but oneDNN's stand-alone softmax takes 1.0 s of the 1.2 s on rows 16.5k long.
oneDNN's *fused* attention (what PyTorch reaches) cannot be used blind: when it does not recognize the pattern it
silently builds the whole S x S score table in device memory - 30 GiB at 16.5k tokens - and on this driver that takes
the machine down (it did, once). So the next step is our own kernel: scores, softmax and the weighted sum over tiles
that never leave the GPU's registers, with memory bounded by construction.

## Checks that every piece must pass

- against the CPU reference in `h3-core::reference` (exact arithmetic, small sizes);
- against the PyTorch pipeline's dumped tensors (real sizes);
- a clip pair for anything that can change the picture or the sound, looked at and listened to, not only measured.
