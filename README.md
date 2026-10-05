# b70_SYCL_minimax_h3

**MiniMax H3** (text / image -> video with sound) as a native engine for the **Intel Arc Pro B70** (32 GB, Xe2).

## The goal

H3 ran on the B70 through PyTorch and ComfyUI's model code: it works, but a 15-second clip takes about 14 minutes,
every job reloads 20 GB of weights and recompiles, and none of the fast kernels the model's authors wrote exist for
Intel (they are CUDA and AMD only).

This project replaces that stack, and every piece of a clip now runs in it - text encoder to .mp4, keyframes, audio
anchors, voice references, masked regeneration, LoRAs, the studio and the film tools. A 15-second clip takes under
9 minutes, and a five-clip film renders in half the time the old stack takes (below). It is built from:

- **SYCL (C++)** for everything that runs on the GPU - `kernels/`: `libh3sycl.so`, with a plain C interface
  (`kernels/h3sycl.h`). It uses the card's matrix engine through oneDNN where oneDNN is at the card's limit, and its
  own kernels where it is not; attention goes through `libh3sage.so`, SageAttention as Intel's ARK kernel on
  sycl-tla (loaded at run time; without it, oneDNN's fused attention).
- **Rust** for everything else - `engine/`: device memory, checkpoint loading, the model graph, the sampler, the
  command line, and the server.
- **TypeScript / TSX on snabbdom** for the web front end - `wfe/`, unchanged from the one in use.

Each piece is checked against the running PyTorch pipeline (`reference/`) before it replaces it: numerically, and
with a pair of clips to look at and listen to. `reference/` is the only Python here; it drives the old pipeline for
those comparisons and is deleted when the port has parity. The plan and its state: [docs/PORT-PLAN.md](docs/PORT-PLAN.md).

## Timings on a B70

Measured on one Arc Pro B70 (Ryzen 7 1700X, 23 GiB RAM), 2026-10-03, with the default attention (SageAttention). "Reference" is the PyTorch pipeline
(ComfyUI's MiniMax H3 code) with the same int8 weights and our linear kernel plugged in - already faster than the
production path it replaced (Q8 GGUF, `torch.compile`: 286 s for the clip below).

### A whole clip, prompt to .mp4, in the engine

"Goodnight Borg" (`reference/prompts/borg_goodnight_5s.txt`, 356 tokens), the production recipe: 768x576 sampled,
5 s (124 frames), 8 steps, the realism LoRA at 0.5, a 1.5x latent upscale to 1152x864, sound. `h3d generate`:

| | reference | this engine | |
|---|---:|---:|---:|
| text encoder (Qwen3-VL 32B, 50 layers) | ~54 s (cached for this run) | 16.5 s, streamed from disk | 3.3x |
| denoiser weights to the GPU | 20.1 s | 12.1 s | 1.7x |
| 8 denoiser steps (16.8k tokens) | 83.3 s | 63.4 s; 58.0 s once the attention kernel is loaded (a ~5 s one-off per engine process) | 1.31x / 1.44x |
| latent upscaler | 7.5 s | 4.5 s | 1.7x |
| video decoder (124 frames, 1152x864) | 37.1 s | 45.4 s | 0.8x |
| audio decoder | 7.4 s | 2.6 s | 2.8x |
| **the clip** | **202 s** with the text encoding cached, ~256 s without | **156 s** wall, text encoder included | **1.29x / ~1.64x** |

The video decoder is the one piece still slower: it runs at about 54 TFLOPS of half-precision work per tile batch,
roughly a third of what the card does on a bare matrix product - the next thing to tune.

### A film: five chained clips through the studio

"Goodnight Borg" part 1, clips 1-5 (54.9 s of clips: 15.1, 14.4, 13.7, 7.3 and 4.5 s), each anchored on the first
clip's last frame at both ends and on the previous clip's last second of sound; 768x576 -> 1.5x, 8 steps, the
realism LoRA. Queued in the studio, joined by `sycl-h3 join` (anchor overlaps, -18 LUFS, sync checked); measured
while Sage also ran the video decoder, which now stays on oneDNN (~1 s faster per second of video):

| | the old stack (PyTorch, Q8 GGUF) | this engine |
|---|---:|---:|
| the five clips | 3343 s | 1699 s |
| per second of video | 61 s | 31 s |
| the joined film | 50.875 s, sync OK | 50.875 s, sync OK - the same shots |

### Each piece against the reference

Every piece was checked against the reference on the same inputs (`reference/` dumps them):

| piece | agreement | reference | this engine |
|---|---|---:|---:|
| tokenizer | identical ids (2 prompts, 19 awkward strings) | | |
| text encoder, 137 tokens | worst token cosine 0.999997 | ~54 s | 16.3 s |
| starting noise (PyTorch's generator) | 83% bit-exact, the rest within 1 ulp | | |
| one denoiser step, 2,159 tokens | first step cosine 0.9988 | 2.0 s | 0.66 s |
| one denoiser step, 16.5k tokens (5 s) | 50 blocks cosine 0.9975 (oneDNN attention) | 11.1 s (production) / 9.3 s | 6.6 s |
| one denoiser step, 47k tokens (15 s) | | 58-63 s | 32.7 s |
| 8 steps, 2 s at 384x288 | latents cosine 0.95-0.985 (8 steps amplify rounding) | 15.7 s | 6.5 s |
| latent upscaler (2x) | cosine 0.99993 | 5.2 s | 3.0 s |
| video decoder, 56 frames 384x288 | PSNR 72.5 dB | 12.0 s | 4.1 s |
| audio decoder | rel err 3e-5 | 8.3 s | 2.2 s |
| video encoder (keyframes): a picture / 22 frames, 384x288 | cosine 0.99999 / 0.99994 | | 3.0 s / 2.6 s |
| audio encoder (audio keyframes), 1 s | rel err 1e-6 | | 0.3 s |

What the card can do, measured with bare oneDNN (docs/PHASE0-RESULTS.md): int8 matrix multiply 317-357 T-ops/s
against 178-183 for 16-bit floats, so int8 linears have a ceiling near 2x; attention built from separate steps is bound by
writing its score table; oneDNN's fused kernel avoids the table (165 G scores/s at production size). The default
attention is SageAttention (Intel's ARK kernel, q and k in int8) for sequences of 8192 tokens and more - the
denoiser: 1.6x oneDNN's at 47k tokens, 1.47x at 16.5k, no visible change in a clip pair. Shorter ones (the video
decoder's tiles, the text refiner) stay on oneDNN, where Sage's quantize pass costs more than int8 saves (the
decoder ran 2x slower with it). `H3S_ATTN=onednn` turns Sage off; `H3S_SAGE_MIN_S` moves the threshold. Beyond
that it takes computing fewer scores.

### The canvas table on each GPU

What the studio's canvas table shows, per GPU, measured by `sycl-h3 plan measure` (2026-10-05, Ryzen 9 9950X,
61 GiB RAM, both cards at PCIe Gen5 x8): the denoiser's 8 steps of the production recipe for each canvas and clip
length. The text encoder, the weights' load, the upscaler and the decoders come on top (97 s of the 155 s clip
above). Each GPU's step time is its `bench-blocks` curve (2,048-47,104 tokens) times a factor from one real
clip on that GPU (a clip's step is the 50 blocks plus its embeddings, final layer and sampler). The B70's 768x576,
5 s cell (0:58) matches the clip measured above (58.0 s); the B65 takes about 1.55x the B70's time throughout.
"Does not fit" is the studio's memory estimate against the 30.3 GiB cap.

**Arc Pro B70** (GPU 0, shared with the chat model; a clip's step = 1.12 x the 50 blocks):

| canvas | 5 s clip | 10 s clip | 15 s clip |
|---|---:|---:|---:|
| 640x480 (4:3) | 0:36 (4.5 s/step) | 1:30 (11.2 s/step) | 2:44 (20.4 s/step) |
| 768x576 (4:3) | 0:58 (7.2 s/step) | 2:35 (19.3 s/step) | 4:48 (36.0 s/step) |
| 576x1024 (9:16) | 1:27 (10.8 s/step) | 4:02 (30.2 s/step) | does not fit (29.8 GiB) |
| 768x768 (1:1) | 1:27 (10.8 s/step) | 4:02 (30.2 s/step) | does not fit (29.8 GiB) |
| 1024x576 (16:9) | 1:27 (10.8 s/step) | 4:02 (30.2 s/step) | does not fit (29.8 GiB) |
| 896x672 (4:3) | 1:29 (11.2 s/step) | 4:10 (31.2 s/step) | does not fit (30.0 GiB) |
| 768x1024 (3:4) | 2:14 (16.8 s/step) | 6:10 (46.2 s/step) | does not fit (33.2 GiB) |
| 1024x768 (4:3) | 2:14 (16.8 s/step) | 6:10 (46.2 s/step) | does not fit (33.2 GiB) |
| 768x1344 (9:16) | 3:24 (25.5 s/step) | does not fit (31.6 GiB) | does not fit (37.4 GiB) |
| 1344x768 (16:9) | 3:24 (25.5 s/step) | does not fit (31.6 GiB) | does not fit (37.4 GiB) |

**Arc Pro B65** (GPU 1; a clip's step = 1.10 x the 50 blocks):

| canvas | 5 s clip | 10 s clip | 15 s clip |
|---|---:|---:|---:|
| 640x480 (4:3) | 0:57 (7.1 s/step) | 2:18 (17.3 s/step) | 4:11 (31.4 s/step) |
| 768x576 (4:3) | 1:30 (11.2 s/step) | 3:57 (29.7 s/step) | 7:24 (55.5 s/step) |
| 576x1024 (9:16) | 2:14 (16.7 s/step) | 6:12 (46.5 s/step) | does not fit (29.8 GiB) |
| 768x768 (1:1) | 2:14 (16.7 s/step) | 6:12 (46.5 s/step) | does not fit (29.8 GiB) |
| 1024x576 (16:9) | 2:14 (16.7 s/step) | 6:12 (46.5 s/step) | does not fit (29.8 GiB) |
| 896x672 (4:3) | 2:18 (17.2 s/step) | 6:24 (48.0 s/step) | does not fit (30.0 GiB) |
| 768x1024 (3:4) | 3:26 (25.8 s/step) | 9:34 (71.7 s/step) | does not fit (33.2 GiB) |
| 1024x768 (4:3) | 3:26 (25.8 s/step) | 9:34 (71.7 s/step) | does not fit (33.2 GiB) |
| 768x1344 (9:16) | 5:13 (39.1 s/step) | does not fit (31.6 GiB) | does not fit (37.4 GiB) |
| 1344x768 (16:9) | 5:13 (39.1 s/step) | does not fit (31.6 GiB) | does not fit (37.4 GiB) |

## Layout

    kernels/      SYCL C++: libh3sycl (h3sycl.h is the whole interface), libh3sage (sage.cpp: SageAttention),
                  and two oneDNN probes (gemm_bench, sdpa_probe)
    engine/       Rust workspace
      h3-sys/       bindings to libh3sycl, loaded at run time
      h3-core/      device memory, checkpoints (safetensors, GGUF), kernels with checked shapes, and the model:
                    tokenizer, text encoder, denoiser, sampler, LoRA, upscaler, video and audio decoders
      h3-http/      the small HTTP/JSON layer both programs below share (TCP or Unix socket)
      h3d/          the engine side, in the container: the daemon, the per-GPU engine process, the jobs, checks
      sycl-h3/      the command line, on the host (a static binary): starts the services, talks over the socket;
                    the studio (the front end's API and clip queue) and the film tools
    tokenizer/    the Qwen2 tokenizer's vocabulary and merges (Apache-2.0)
    wfe/          the web front end (TSX + snabbdom, vendored compiler, no node_modules)
    container/    the build-and-run image (podman)
    tools/        examples for the studio's configuration (characters for sycl-h3 speech, language-model modes)
    reference/    the PyTorch pipeline's harness, for comparisons only
    docs/

## Build and run

Everything builds inside one podman container; the host needs podman (with `crun`, so a rootless container can
reach the GPU) and nothing else.

    ./setup.sh                      # build the container image (SYCL compiler, oneDNN 3.12 from source, sycl-tla and
                                    # ARK's headers at pinned commits, Rust, node)
    ./build.sh                      # kernels + SageAttention library + engine + front end -> dist/
                                    # (./build.sh kernels|sage|engine|wfe for one part)
    ./build.sh test                 # the Rust tests and lints
    ./teardown.sh [--all]           # stop the services; --all also removes the build output and the image

### Using it: `sycl-h3`

`sycl-h3` is the command line, on the host: a static binary in `dist/` (copy it onto your PATH if you like; it finds
`dist/` beside itself, or through `H3_DIST`). It starts two services - either runs without the other - and talks to
the engine over a Unix socket, like `docker` and `dockerd`.

    ./sycl-h3 gpus                  # the GPUs, numbered as --gpu takes them
    ./sycl-h3 start                 # the engine daemon: every GPU (or --gpu 0 --gpu 1 ...); --shared-gpu N for the
                                    # GPU(s) another program normally holds (see below)
    ./sycl-h3 serve                 # the studio: the web front end and its clip queue, http://127.0.0.1:8095/
                                    # (--bind 0.0.0.0 --port 9000 ...)
    ./sycl-h3 status                # live, one row per GPU, like docker stats (--no-stream: once)
    ./sycl-h3 jobs add generate --prompt "a cat on a piano" --width 768 --height 576 --seconds 5 \
        --upscale 1.5 --lora /models/loras/<lora>.safetensors:0.5 --out /out/cat.mp4 -f    # a whole clip, directly
    ./sycl-h3 jobs ps [-a] | stop <id>... | rem <id>... | details <id>
    ./sycl-h3 unload [--gpu N]      # give a GPU back now; the next job loads again
    ./sycl-h3 stop [--web | --all]  # the engine (default), the studio, or both - gracefully
    ./sycl-h3 logs [--web]

    ./sycl-h3 speech speech.txt --character data --audio-anchor prev --guide-frames 22   # a speech as chained clips
    ./sycl-h3 scene film.scene.json # queue a scene file (the front end's export)
    ./sycl-h3 join "Speech"         # the series' finished clips as one film, frame-exact
    ./sycl-h3 speechpct out/h3_*.mp4
    ./sycl-h3 jobs add generate ... --upscale 1.5 --pixel_upscaler /models/esrgan/realesr-animevideov3.safetensors
                                    # enlarge the decoded frames with an ESRGAN-type network instead of the latents
                                    # (the studio: "upscaler" per clip; the latent upscaler stays the default)
    ./sycl-h3 plan measure          # what a step costs on each GPU (bench-blocks over 2k-47k tokens and a short
                                    # clip per GPU), for the studio's canvas table: one column per GPU

How it fits together:

    sycl-h3 (host) --start/stop (podman)--> [sycl-h3]      h3d daemon --pipes--> h3d worker --gpu 0, --gpu 1 ...
         |                                     ^ Unix socket, JSON over HTTP
         +--status / jobs / unload ------------+
         +--serve (a host process)----------> studio        web front end + clip queue, TCP -> the same socket

- `h3d daemon` keeps the job queue and never opens a GPU. Each GPU it serves gets its own engine process,
  `h3d worker --gpu N`, started when a job needs that GPU; the model stays loaded on it between jobs. The worker
  ends after a while without jobs (`H3_IDLE`, 600 s by default), on `unload`, or on `stop`: the GPU's memory comes
  back with the process, and a crash in an engine leaves the daemon up. Daemon and workers talk over the worker's
  stdin/stdout, one JSON object per line (job, cancel, exit; log lines, progress, memory, results); tensors and video
  never cross the pipe. Jobs run on any free GPU, or on the one they name. A cancel lands at the next block
  boundary, never inside a kernel (1.1 s at production size).
- The studio serves the same TSX/snabbdom front end as before and answers its API the way the old Python server
  did (docs/LEGACY-API.md): the clip queue with holds per project and batch, anchors (`prev`, `first`, per camera)
  resolved against the series, retries, per-clip records with the speech fraction, thumbnails, the timeline, scene
  files, films. Each clip becomes the engine's `generate` job. It runs on the host because it also switches the
  box's language models (`H3_LLM_MODES`), which are the host's own programs.
- What a clip can be anchored to (the request fields, and `sycl-h3 jobs add generate` options): a picture or the
  previous clip's last frame at the start and/or the end (exposure-matched), the previous clip's last latent, a
  **motion guide** (the previous clip's last 22 frames as one moving keyframe - position and velocity), an **audio
  keyframe** (the previous clip's last second of sound, so the room tone and the voice carry across the cut), a
  **voice reference** (`<Audio 1>` in the prompt), and a **masked run** (regenerate seconds a-b of a clip, or extend
  it, keeping the rest exactly). Every clip leaves the next one's anchors beside it (`.last.png`,
  `.lastaud.safetensors`, `.lastlat.safetensors`, `.latents.safetensors`).
- Sharing a GPU with another program: `H3_GPU_LOCK` names a lock file the daemon waits on and holds while an engine
  on a shared GPU is loaded; `H3_LLM_SWITCHER` names a model switcher (the studio's `/rpc/llm.mode`) whose model is
  stopped before loading and restored after the last such engine has ended (`--shared-gpu` / `H3_SHARED_GPUS` say
  which GPUs these are about; all served ones by default). Either way an engine waits until its card really has the
  memory free. With a GPU of its own beside a shared one, a job that names no GPU goes to the unshared one while it is
  free, and an engine on it does not keep the model off the shared card.

Settings go in `sycl-h3.conf` beside the repository or `~/.config/sycl-h3.conf` (`sycl-h3 help` lists them).

For measuring and debugging the engine itself, `./run.sh` runs one-shot checks in the container (`H3_MODELS` set):

    ./run.sh device | info <ckpt> | load <ckpt>
    ./run.sh check-linear /models/<ckpt>.safetensors    # GPU against the CPU reference, and timed
    ./run.sh check-block /models/<ckpt>.safetensors /out/blockdump.safetensors   # 50 blocks against a reference dump
    ./run.sh bench-blocks /models/<ckpt>.safetensors --tokens 16500              # a denoiser step, stage by stage
    ./run.sh generate /models/<ckpt>.safetensors --prompt-file /out/p.txt --out /out/clip.mp4  # a clip, one-shot
    ./run.sh encode | denoise | decode ...       # the pieces alone, each with --check against a reference dump

### Versions and releases

The version is `yy.mmdd.###` - year, month and day, then that day's sequence number - and lives in the `VERSION`
file (`sycl-h3 version` prints it; a test keeps Cargo's copy in step). `.github/workflows/build.yml` builds the image,
runs `./build.sh` and `./build.sh test` on every push and pull request and keeps the result as a workflow artifact;
on `main` it also publishes the build image to the GitHub container registry, and on a tag `v<VERSION>` it publishes
a release with `h3-engine-<VERSION>-linux-x86_64.tar.gz` (sycl-h3, h3d, the kernel libraries with their oneDNN, the
built front end).

One model per GPU: Intel's `xe` driver has no out-of-memory error, an over-committed card stalls the whole machine.
The engine counts its own allocations and refuses to pass 94% of the card; do not start it beside another program
that holds the card (or share it through `H3_GPU_LOCK`, above), and stop it with `sycl-h3 stop` or `./teardown.sh`,
never with a kill.

## Documents

| | |
|---|---|
| [docs/PORT-PLAN.md](docs/PORT-PLAN.md) | every piece of a clip: where it runs today, where it goes, its state |
| [docs/PHASE0-RESULTS.md](docs/PHASE0-RESULTS.md) | what the card can do: int8 against float, attention, the first-step cost |
| [docs/BASELINE.md](docs/BASELINE.md) | the reference pipeline's numbers and where a step's time goes |
| [docs/LESSONS-FROM-STRATA.md](docs/LESSONS-FROM-STRATA.md) | what an earlier SYCL port on this card taught |
| [docs/ASSESSMENT.md](docs/ASSESSMENT.md) | the first assessment (before the measurements; kept for the record) |
| [reference/README.md](reference/README.md) | the comparison harness |
