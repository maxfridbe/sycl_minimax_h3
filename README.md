# b70_SYCL_minimax_h3

**MiniMax H3** (text / image -> video with sound) as a native engine for the **Intel Arc Pro B70** (32 GB, Xe2).

## The goal

Today H3 runs on the B70 through PyTorch and ComfyUI's model code: it works, but a 15-second clip takes about 14
minutes, every job reloads 20 GB of weights and recompiles, and none of the fast kernels the model's authors wrote
exist for Intel (they are CUDA and AMD only).

This project replaces that stack, piece by piece, until nothing of it is left:

- **SYCL (C++)** for everything that runs on the GPU - `kernels/`, one library, `libh3sycl.so`, with a plain C
  interface (`kernels/h3sycl.h`). It uses the card's matrix engine through oneDNN where oneDNN is at the card's
  limit, and its own kernels where it is not.
- **Rust** for everything else - `engine/`: device memory, checkpoint loading, the model graph, the sampler, the
  command line, and the server.
- **TypeScript / TSX on snabbdom** for the web front end - `wfe/`, unchanged from the one in use.

Each piece is checked against the running PyTorch pipeline (`reference/`) before it replaces it: numerically, and
with a pair of clips to look at and listen to. `reference/` is the only Python here; it drives the old pipeline for
those comparisons and is deleted when the port has parity. The plan and its state: [docs/PORT-PLAN.md](docs/PORT-PLAN.md).

## Timings on a B70

Measured on one Arc Pro B70 (Ryzen 7 1700X, 23 GiB RAM), 2026-10-02. "Reference" is the PyTorch pipeline as it runs
in production: Q8 GGUF weights, `torch.compile`, 768x576, 8 steps, realism LoRA, 1.5x latent upscale.

### A whole clip ("Goodnight Borg", 5 s, production recipe, text encoding cached)

| | reference | int8 weights + SYCL linears | |
|---|---:|---:|---:|
| weights to the GPU + first-step warm-up | 119.6 s | 44.0 s | 2.7x |
| one denoiser step (16.5k tokens) | 11.1 s | 9.3 s | 1.19x |
| 8 steps, with the above | 217.2 s | 127.3 s | 1.71x |
| everything after (upscale, decode, mux) and process start-up | 69 s | 75 s | not ported yet |
| **the clip** | **286.2 s** | **202.1 s** | **1.42x** |

The SYCL column still runs inside the PyTorch process (the kernel library is plugged into the model's own kernel
dispatch); only the denoiser's linear layers are ours so far. A 15-second production clip (47k tokens) is about
845 s on the reference, 58-63 s per step, 70% of it attention - which is the next piece.

### The pieces that are ported

| | reference | this engine | |
|---|---:|---:|---:|
| a denoiser block's four linear layers, 16k tokens | 84 ms (bf16) / 145 ms (int8, PyTorch ops) | 61 ms | 1.4x / 2.4x |
| the same, error against exact arithmetic | 0.69% with the rotation kept in 16 bits, as comfy-kitchen does it | 0.17% | |
| 19.5 GiB of weights, disk to GPU | 15-44 s | 10.7-12.8 s (Rust, 8 readers) | 1.2-3.5x |
| start-up before the first step | load + ~36 s compile, every job | load only | |

### The denoiser in the Rust engine (no PyTorch)

The denoiser's 50 blocks run from the Rust engine alone and track the reference block by block
(docs/PORT-PLAN.md). One step = 50 blocks:

| | reference | Rust + SYCL engine | |
|---|---:|---:|---:|
| 16.5k tokens (a 5 s clip) | 11.1 s | 8.95 s | 1.24x |
| 47k tokens (a 15 s clip) | 58-63 s | 49.9 s | 1.16-1.27x |
| of which attention, per block, 16.5k tokens | 102 ms | 96 ms | |
| of which everything else, per block, 16.5k tokens | ~99 ms | 81 ms | |

### A whole denoise in the Rust engine

Text conditioning and starting noise in, finished latents out, with no PyTorch: the text refiner, the patch
embeddings, the 50 blocks, the final layer and the Euler sampler (`h3d denoise`, checked against a reference run dump).
The bakery prompt at 384x288, 2 s, 8 steps (2,159 tokens), with the same int8 weights:

| | reference | Rust + SYCL engine | |
|---|---:|---:|---:|
| 8 sampler steps | 15.7 s | 6.8 s (0.78 s a step after the first) | 2.3x |
| text refiner (once per clip) | included above | 3.9 s, loaded and freed | |
| finished latents against the reference's | | cosine 0.985 video, 0.997 audio | |

Decoded by the reference's decoders, it is the same scene with small differences in detail.
The decoders, the upscaler and the text encoder are still to port, so the whole-clip timings above come from the
PyTorch process with our linear kernel plugged in.

What the card can do, measured with bare oneDNN (docs/PHASE0-RESULTS.md): int8 matrix multiply 317-357 T-ops/s
against 178-183 for 16-bit floats, so int8 linears have a ceiling near 2x; attention built from separate steps is bound by
writing its score table; oneDNN's fused kernel avoids the table (165 G scores/s at production size), and beyond that
it takes computing fewer scores.

## Layout

    kernels/      SYCL C++: libh3sycl (h3sycl.h is the whole interface), and a oneDNN benchmark
    engine/       Rust workspace
      h3-sys/       bindings to libh3sycl, loaded at run time
      h3-core/      device memory, checkpoints, loading, kernels with checked shapes, the denoiser's blocks,
                    CPU reference arithmetic
      h3-http/      the small HTTP/JSON layer both programs below share (TCP or Unix socket)
      h3d/          the engine side, in the container: the daemon, the per-GPU engine process, the jobs, checks
      sycl-h3/      the command line, on the host (a static binary): starts the services, talks over the socket
    wfe/          the web front end (TSX + snabbdom, vendored compiler, no node_modules)
    container/    the build-and-run image (podman)
    reference/    the PyTorch pipeline's harness, for comparisons only
    docs/

## Build and run

Everything builds inside one podman container; the host needs podman (with `crun`, so a rootless container can
reach the GPU) and nothing else.

    ./setup.sh                      # build the container image (SYCL compiler, oneDNN 3.12 from source, Rust, node)
    ./build.sh                      # kernels + engine + front end -> dist/
    ./build.sh test                 # the Rust tests and lints
    ./teardown.sh [--all]           # stop the services; --all also removes the build output and the image

### Using it: `sycl-h3`

`sycl-h3` is the command line, on the host: a static binary in `dist/` (copy it onto your PATH if you like; it finds
`dist/` beside itself, or through `H3_DIST`). It starts two services, each in its own container - either can run
without the other - and talks to the engine over a Unix socket, like `docker` and `dockerd`.

    ./sycl-h3 gpus                  # the GPUs, numbered as --gpu takes them
    ./sycl-h3 start                 # the engine daemon: every GPU (or --gpu 0 --gpu 1 ...); --shared-gpu N for the
                                    # GPU(s) another program normally holds (see below)
    ./sycl-h3 serve                 # the web front end, http://127.0.0.1:8095/ (--bind 0.0.0.0 --port 9000 ...)
    ./sycl-h3 status                # live, one row per GPU, like docker stats (--no-stream: once)
    ./sycl-h3 jobs add bench-blocks --tokens 47173 -f     # queue a job and follow its log (--gpu N to pin it)
    ./sycl-h3 jobs add check-block --dump /out/blockdump.safetensors
    ./sycl-h3 jobs ps [-a] | stop <id>... | rem <id>... | details <id>
    ./sycl-h3 unload [--gpu N]      # give a GPU back now; the next job loads again
    ./sycl-h3 stop [--web | --all]  # the engine (default), the web front end, or both - gracefully
    ./sycl-h3 logs [--web]

How it fits together:

    sycl-h3 (host) --start/stop (podman)--> [sycl-h3]      h3d daemon --pipes--> h3d worker --gpu 0, --gpu 1 ...
         |                                     ^ Unix socket, JSON over HTTP
         +--status / jobs / unload ------------+
         +--serve (podman)------------------> [sycl-h3-web]  web front end, TCP -> the same socket

- `h3d daemon` keeps the job queue and never opens a GPU. Each GPU it serves gets its own engine process,
  `h3d worker --gpu N`, started when a job needs that GPU; the model stays loaded on it between jobs. The worker
  ends after a while without jobs (`H3_IDLE`, 600 s by default), on `unload`, or on `stop`: the GPU's memory comes
  back with the process, and a crash in an engine leaves the daemon up. Daemon and workers talk over the worker's
  stdin/stdout, one JSON object per line (job, cancel, exit; log lines, progress, memory, results); tensors and video
  never cross the pipe. Jobs run on any free GPU, or on the one they name. A cancel lands at the next block
  boundary, never inside a kernel (1.1 s at production size).
- The web service serves the same TSX/snabbdom front end as before and passes the engine's API through to the
  daemon's socket. The front end's calls for the clip queue, projects, scenes and films are not ported to Rust yet:
  it passes them on to the server they were written for (`H3_LEGACY_API`), so it works whole while those move over.
- Sharing a GPU with another program: `H3_GPU_LOCK` names a lock file the daemon waits on and holds while an engine
  on a shared GPU is loaded; `H3_LLM_SWITCHER` names a front end's model switcher whose model is stopped before
  loading and restored after the last such engine has ended (`--shared-gpu` / `H3_SHARED_GPUS` say which GPUs these
  are about; all served ones by default). Either way an engine waits until its card really has the memory free.

Settings go in `sycl-h3.conf` beside the repository or `~/.config/sycl-h3.conf` (`sycl-h3 help` lists them).

For measuring and debugging the engine itself, `./run.sh` runs one-shot checks in the container (`H3_MODELS` set):

    ./run.sh device | info <ckpt> | load <ckpt>
    ./run.sh check-linear /models/<ckpt>.safetensors    # GPU against the CPU reference, and timed
    ./run.sh check-block /models/<ckpt>.safetensors /out/blockdump.safetensors   # 50 blocks against a reference dump
    ./run.sh bench-blocks /models/<ckpt>.safetensors --tokens 16500              # a denoiser step, stage by stage

### Versions and releases

The version is `yy.mmdd.###` - year, month and day, then that day's sequence number - and lives in the `VERSION`
file (`h3 version` prints it; a test keeps Cargo's copy in step). `.github/workflows/build.yml` builds the image,
runs `./build.sh` and `./build.sh test` on every push and pull request and keeps the result as a workflow artifact;
on `main` it also publishes the build image to the GitHub container registry, and on a tag `v<VERSION>` it publishes
a release with `h3-engine-<VERSION>-linux-x86_64.tar.gz` (sycl-h3, h3d, the kernel library with its oneDNN, the built
front end).

One model per GPU: Intel's `xe` driver has no out-of-memory error, an over-committed card stalls the whole machine.
The engine counts its own allocations and refuses to pass 94% of the card; do not start it beside another program
that holds the card, and stop it with `./teardown.sh`, never with a kill.

## Documents

| | |
|---|---|
| [docs/PORT-PLAN.md](docs/PORT-PLAN.md) | every piece of a clip: where it runs today, where it goes, its state |
| [docs/PHASE0-RESULTS.md](docs/PHASE0-RESULTS.md) | what the card can do: int8 against float, attention, the first-step cost |
| [docs/BASELINE.md](docs/BASELINE.md) | the reference pipeline's numbers and where a step's time goes |
| [docs/LESSONS-FROM-STRATA.md](docs/LESSONS-FROM-STRATA.md) | what an earlier SYCL port on this card taught |
| [docs/ASSESSMENT.md](docs/ASSESSMENT.md) | the first assessment (before the measurements; kept for the record) |
| [reference/README.md](reference/README.md) | the comparison harness |
