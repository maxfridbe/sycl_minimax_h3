# Speedups to do

Ordered by what each is expected to buy for the testing it needs. The port's own open items (video decoder
batching, reference pictures, the student text encoder, Xe3) are in [docs/PORT-PLAN.md](docs/PORT-PLAN.md), "To do".
Several of these come from the Strata SYCL port on the same card (docs/LESSONS-FROM-STRATA.md has its earlier
lessons); the numbers below are from the Ryzen 9 9950X box (61 GiB RAM, B70 and B65 at PCIe Gen5 x8), 2026-10-05.

## 1. The text encoder from pinned RAM between clips

Each clip streams the 32B text encoder (Q4_K GGUF, about 19 GB) from the SSD: 16.2 s of the 155 s production clip.
Strata keeps its model's experts in pinned host memory and the GPU reads them over PCIe (26.5 GB/s measured on this
box, host to device). Kept in pinned RAM by the worker after the first clip, the encoder's layers would reach the
GPU in under a second; what remains is dequant + the 356-token GEMMs (well under a second).

- [x] Done (2026-10-05): the worker keeps the encoder's matrices in pinned host memory (14.4 GiB) between clips, and
      every layer goes up by DMA on a copy queue while the layer before computes (two device buffers). Without the
      room (16.5 GB + 8 GiB free; a second worker on the other GPU usually lacks it) two pinned layer slots, same
      overlap. `H3_TE_PIN=0`: no kept copy. On the B65: the first clip's encode 9.1 s (was 13.6 cold, 10.3 with
      the file in the page cache), every later one **1.6 s**; latents byte-identical to the old build (two
      prompts). The disk was not the limit on this box: the pageable, synchronous per-layer copies were.

## 2. Denoiser weights streamed from pinned RAM, for the cells that do not fit

The canvas table refuses the 15 s clips at most canvases and 10 s at 1344x768: the 18 GiB of block weights plus the
work buffers pass the 30 GiB cap. Strata decodes with a quarter of its experts in pinned RAM, read over PCIe every
step, at 65-78 tok/s. Here, half the block weights (9 GiB) streamed per step cost ~0.35 s at Gen5 against a 32-56 s
step at 47k tokens (1%), and free ~9 GiB for activations.

- [ ] A resident/streamed split per block, chosen from the token count (stream only what the step needs freed),
      double-buffered so block n+1's weights copy while block n runs.
- [ ] The studio's fit check and canvas table learn the split (a cell that fits only with streaming says so, with
      its measured cost).
- [ ] Check: bit-identical steps against all-resident at a size that fits both ways; the step time added.

Expected: 15 s clips at 1024x576-class canvases and longer ones become possible; ~1-3% per step where used.

## 2b. The video decoder (the largest piece of a clip after sampling)

Profiled 2026-10-05 on the B65, 124 frames at 768x576 (the fp16 decoder, as production uses it): 17.6 s, of it the
four linears 54% (already at the card's fp16 rate, ~88-90 TFLOP/s), attention 26% (oneDNN's fused kernel at ~33
TFLOP/s on 1,797-token tiles), the gated activation 10%, norms + rotation 10%.

- [x] Attention for the batch of tiles in one call, reading q/k/v and writing the result in place through strides
      (`h3s_attention_batch`): the per-tile copies gone. 17.6 -> 16.6 s, output identical (PSNR inf).
      `H3_VAE_ATTN_PER_TILE=1`: the old calls.
- [x] The gated activation 8 features per work-item through 16-byte loads (`swiglu16`): 1.57 -> 0.51 s, identical
      results. Decode 16.6 -> **15.6 s (-11% in all)**; the denoiser's step -2% (9.91 vs 10.09 s at 16k tokens).
      `H3S_SWIGLU_SCALAR=1`: the old kernel.
- [ ] **Opt-in, output changes:** the int8 decoder checkpoint (models/kitchen) is only 5% faster whole (its
      rotations cost what the int8 GEMMs save, except in w1: 1.67x there), at PSNR 44.8 dB / SSIM 0.989 against
      fp16. int8 for w1 alone would save ~1.7 s of 15.6 - a clip pair to judge first.
- [x] The norms the same way (16-byte loads, the same arithmetic and summing order): the per-block norms 0.84 ->
      0.39 s, the q/k norm + rotation 0.66 -> 0.32 s. Decode 15.6 -> **15.0 s (17.6 at the start: -15%)**, still
      identical; a whole clip's latents (text encoder, denoiser) byte-identical too. `H3S_NORM_SCALAR=1`: the old
      kernels.
- [ ] Attention itself: oneDNN's kernel at ~33 TFLOP/s for 64-feature heads on 1.8k tokens (SageAttention was 2x
      slower at this length). A kernel of our own would have to beat oneDNN's; open.

## 2c. An ESRGAN upscaler in pixels (investigate)

Today a clip is sampled at 768x576, its latents upscaled 1.5x by the latent upscaler (3D convolutions, 4.8 s), and
the video decoder runs at 1152x864 - 2.25x the pixels of the sampled size, so ~2.25x the decoder's time (47.5 s at
1152x864 on the B70 before the 2026-10-05 kernels; ~20 s at 768x576). The other way: decode at the sampled size,
then upscale the frames in pixels with an ESRGAN-type network (RRDBNet / Real-ESRGAN, x2 or x1.5) as a SYCL kernel
set (oneDNN convolutions, the residual-in-residual dense blocks, pixel shuffle). No ESRGAN model is on the box yet
(ComfyUI's upscale_models is empty).

- [x] **What it would save, measured** (2026-10-05, B65, one probe clip's latents, 124 frames): today's path - the
      latent upscale 5.6 s, then the decoder at 1152x864 36.3 s - against the decoder at 768x576 15.0 s: **~27 s** a
      clip on the B65 (~17 s on the B70 by the cards' ratio).
- [x] **A clip pair, the network run on the CPU (PyTorch 2.14) for quality only**: Real-ESRGAN `realesr-animevideov3`
      (16 convs, 1.0 s a frame on the CPU) and `realesr-general-x4v3` (32 convs, 1.8 s), x4 then an area resize to
      1152x864, against today's path; a bicubic control. Mean frame-to-frame change (a flicker proxy): today 1.644,
      animevideov3 1.627, general-x4v3 1.697, bicubic 1.504 (softer). PSNR against today's path: 34.4 / 33.1 / 35.3 dB.
      By eye (centre crops): both as sharp as today's path or crisper on edges; today's path keeps a little more
      natural fine texture (the decoder synthesizes it at full size); bicubic visibly softer. **Waiting on the
      user's judgment** (`out/esr-sbs-*.mp4`, `out/esr-grid-crop.png` on the box). Weights in models/esrgan
      (official Real-ESRGAN releases v0.2.1 / v0.2.5.0, sha256 recorded there).
- [x] **Built, and the studio's default** (2026-10-05: after a detail test at 1024x768 -> 1536x1152 the user picked
      general-x4v3 as the default; "latent" stays selectable, and a box without the weights falls back to it):
      `h3s_conv2d` (oneDNN, frames channels-last, zero padding), `h3s_prelu`, `h3s_pixel_shuffle_add`,
      `h3s_resize_area` (adaptive-average area resize to the latent path's exact output size, clamped, planar);
      `h3-core::esrgan::PixelUpscaler` (SRVGGNetCompact, 4 frames a pass, half); jobs take `pixel_upscaler`, `h3d
      decode --pixel-upscaler`, the studio a per-clip `upscaler`: "latent" | "esrgan-anime" | "esrgan-general" (a
      select beside the factor). Weights: reference/esrgan_to_safetensors.py into models/esrgan.
      B65, 124 frames to 1152x864: the network **3.0 s** (animevideov3) / 4.7 s (general-x4v3) after a 14.9 s decode,
      against 5.6 + 36.3 s through the latent upscaler: **17.9 / 19.6 s against 41.9 s**. Frames against the CPU
      PyTorch run: 42.0 / 41.0 dB (half precision, and the CPU run's input was the mp4).
- [ ] The canvas table times only the sampling; the decode-side choice (latent vs pixels) could show there too.

## 3. Smaller

- [ ] The denoiser's weight load (11.6 s per worker start) through a pipelined staging ring like Strata's expert
      cache fill (1.42 GB/s there): only matters when a worker starts (H3_IDLE).
- [ ] Check docs/LESSONS-FROM-STRATA.md covers two later Strata lessons: a queue copy with no device end (pinned host
      <-> pageable host) hangs the B70's copy engine ("Engine reset: bcs", then "device wedged" - a reboot); and
      every device-side spin must be bounded.

## Not expected to pay (from Strata's measurements)

- Strata's attention findings (staging in local memory, 16-lane sub-groups, joint_matrix attention all lost there)
  are about a sparse, gathered attention; H3's dense attention already runs near the card's limit (SageAttention).
- Graph / launch-gap work: Strata's decode has ~2,500 small nodes per round; H3's kernels are large at real token
  counts.
