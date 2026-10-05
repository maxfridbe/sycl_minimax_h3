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

- [ ] Keep the encoder's layers in a pinned host buffer in the worker (allocated once, filled on the first encode),
      streamed layer by layer as now but from RAM; `H3_TE_PIN=0` keeps today's path.
- [ ] Only when the RAM is there: the box also runs a chat model (8-15 GB pinned); skip the pinning below a free-RAM
      floor, as Strata's lend mirror does (it pins only beyond 8 GiB free).
- [ ] Check: the same conditioning bytes (cosine 1.0 against the streamed path), the clip's TE time, two jobs in one
      daemon (cross-job state, the 2026-10-03 lesson).

Expected: ~16 s -> ~1-2 s per clip after the first (about 9% of a production clip, more of a short one).

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
