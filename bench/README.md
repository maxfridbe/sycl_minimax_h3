# Comparison harness (to build first - Phase 0)

Nothing here yet. The harness comes before any engine code, because its first three numbers decide whether
there is an engine at all (ASSESSMENT.md, Phase 0).

Planned, in order:

1. `split_first_step.py` - one production clip with the denoiser's weights forced onto the card before step 1,
   then with `torch.compile` off, then with a warm SYCL kernel cache: how the 124 s first-step extra divides.
2. `profile_step.py` - per-op profile of one production step (N = 52k, Q8_0), self time on the device.
3. `dump_step.py` - saves one step's inputs and outputs per block (noise, conditioning, latents, one block's
   q/k/v) at a small shape, as the reference tensors every kernel parity test reads.
4. `gemm_int8_bench` (C++/SYCL + oneDNN) - int8 x int8 and bf16 GEMM at the DiT's shapes (5376 x 21504,
   5376 x 28672, 14336 x 5376, 7168 x 5376; M = 22k and 52k tokens).
5. `attn_bench` (C++/SYCL + oneDNN) - 56 heads x 128, N = 22k and 52k: oneDNN SDPA, a tiled fp16 kernel, a tiled
   int8-score kernel. Allocator-capped; nothing may allocate N x N.
6. `clip_ab.py` - two finished clips from the same noise and conditioning: per-step latent cosine / PSNR, frame
   PSNR / SSIM, audio level, stage timings and Wh side by side.

Rules carried over from the Strata rig: the card holds one job at a time (the front end's scheduler owns it);
a benchmark waits for the card, runs under a watchdog, stops gracefully, and gives the card back to what had it.
