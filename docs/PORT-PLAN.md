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
| RMS norm + rotary position (per head, on q and k) | comfy-kitchen | SYCL kernel | next |
| attention (scores, softmax, weighted sum) | PyTorch / oneDNN | SYCL: oneDNN first, then a fused kernel | next |
| gated activation between the MLP's two linears | PyTorch | folded into the int8 linear's quantizer | next |
| norm, scale+shift, gated residual add | PyTorch | SYCL kernels | next |
| denoiser block, 50 of them + 2 refiner blocks | ComfyUI `ldm/minimax/model.py` | `h3-core::dit` | next |
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

## Checks that every piece must pass

- against the CPU reference in `h3-core::reference` (exact arithmetic, small sizes);
- against the PyTorch pipeline's dumped tensors (real sizes);
- a clip pair for anything that can change the picture or the sound, looked at and listened to, not only measured.
