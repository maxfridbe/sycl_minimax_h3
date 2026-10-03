# The reference pipeline's harness

The only Python in this repository. It drives the **existing** PyTorch pipeline (ComfyUI's H3 model code in the
`h3-xpu` image) so that every ported piece can be measured and compared against it. It is not part of the engine and
is deleted when the port has parity.

| | |
|---|---|
| `h3x.py` | the production command line, with switches for experiments (below) |
| `run_h3x.sh <tag> [-e VAR=..]... -- <gen args>` | one clip through `h3x.py`, log to `out/<tag>.log` |
| `compare.py A B --out P --labels a b` | two clips: per-step similarity of the denoised latents, frame PSNR, a frame grid, a side-by-side mp4 |
| `kitchen/h3sycl.py`, `kitchen/kitchen_sycl.py` | `libh3sycl` loaded into the PyTorch process and registered as a backend of the model's kernel dispatch (comfy-kitchen), so the old pipeline runs with our kernels |
| `tests/test_int8_linear.py` | the int8 linear on PyTorch tensors: against comfy-kitchen's and against exact float32 |
| `card.sh`, `with-card.sh` | take and give back a GPU that another program normally holds |
| `exp1.sh` ... `exp7.sh` | the experiments behind the numbers in `docs/` and the README; `exp7.sh` makes the block dump `h3 check-block` reads |
| `h2d_bench.py` | host-to-device copy rates |
| `prompts/` | the prompts the comparison clips use |

Switches of `h3x.py` (environment):

| | |
|---|---|
| `H3X_PRELOAD=1` | put the denoiser's weights on the GPU before step 1, timed |
| `H3X_PREFETCH=<threads>` | read the checkpoint ahead in parallel while the rest starts |
| `H3X_DUMP_STEPS=<prefix>` | save the denoised estimate after every step (for `compare.py`) |
| `H3X_DUMP_RUN=<file>` | save a whole sampling run - conditioning, noise, sigmas, every step's estimate, the final latents - for the Rust engine's `denoise` check |
| `H3X_DUMP_BLOCK=<file>` | save block 0's input, every intermediate of it, and later blocks' outputs (for `h3 check-block`) |
| `H3X_INT8_NATIVE=1` | keep int8 weights int8 (ComfyUI turns them back into floats on an Intel GPU) |
| `H3X_SYCL=1` | the above, with `libh3sycl` in front of comfy-kitchen's own backends |
| `H3X_SYCL_SYNC=1`, `H3X_PROFILE=1` | time our kernels / the pieces of a denoiser step (synchronized, so slower) |

The scripts expect this directory's files in one working directory on the GPU box (mounted at `/work` in the
`h3-xpu` image), with `kitchen/*.py` and the built `libh3sycl.so` + `libdnnl.so.3` in its `kernels/` subdirectory.
