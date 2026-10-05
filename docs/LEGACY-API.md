# The studio API (what the web front end and the tools call)

The studio (`sycl-h3 serve`, `engine/sycl-h3/src/studio/`) answers the routes of the Python server the web front end
was written for, the same way, so the front end, scene files and queue scripts keep working. The clips are made by
the engine daemon's `generate` jobs instead of a PyTorch container. This page is the contract; `/api/types.ts` (served
by the studio) has the request and result types.

## Routes

GET: `/` (the front end), `/ui/<module>`, `/api/status` (= RPC `status`), `/api/summary`, `/api/gpu`, `/api/list`,
`/api/films`, `/api/timeline?project=`, `/api/plan`, `/api/engines`, `/api/canvases`, `/api/templates`,
`/api/clip?label=`, `/api/job/<h3_YYYYmmdd_HHMMSS>`, `/api/scene?project=` (a scene file), `/api/types.ts`,
`/thumb/<clip>.webp` (240 px, 5 s in), `/out/<file>.mp4` (byte ranges), `/engine/*` (the daemon's API, passed through).

POST: `/rpc/<method>` with a JSON object body; the answer is `{"ok": true, "result": ...}` or
`{"ok": false, "error": {"code", "message"}}` (codes `invalid_request`, `invalid_param`, `not_found`, `busy`,
`not_running`, `unknown_method`, `internal`). Also `/api/queue/pause {on, project?, batch?}`, `/api/queue/clear`,
`/api/scene/import {scene, project?, paused?}`, and the oldest `/api/gen` (clamps instead of rejecting).

Methods: `status`, `wait {version?, timeout_s?}`, `generate`, `cancel`, `queue.list`, `queue.clear {project?|batch?}`,
`queue.pause` / `queue.resume {project?|batch?}`, `projects.list`, `timeline {project}`, `scene.export {project}`,
`scene.import {scene, project?, paused?}`, `films.list`, `jobs.list {limit?}`, `jobs.get {name}`, `plan`,
`engines.list`, `canvases.list`, `templates.list`, `gpu`, `summary`, `llm.mode {mode?}`.

## A clip request (`generate`)

`prompt` (required), `seconds` 1-15.1 (10), `steps` 1-40 (10), `seed`, `width` / `height` (multiples of 32,
256-1344, at most 768x1344 pixels; 768x576), `label` ("Name NN/MM: ..." - NN is the edit order, the prefix the
series), `project`, `batch`, `camera`, `upscale` (the factor), `upscaler` ("latent", the default; "esrgan-anime" / "esrgan-general": decode at the sampled size, then the frames enlarged by an ESRGAN-type network), `loras` (["PATH:STRENGTH", ...]; they
stack), `queue` (true: queue when busy instead of answering 409).

Anchors - an anchor is `prev` (the series' previous clip), `prev_cam` (same camera), `first` (the series' first clip),
`first_cam`, or a file in the clips directory:

| field | the engine's option | notes |
|---|---|---|
| `first_frame` + `chain_mode` | `first_frame` / `first_latent` | `png`: the clip's lossless last frame; `video`: its mp4's last frame; `latent`: its last latent frame (see the caveat below); `none` |
| `last_frame` | `last_frame` | always a picture (the old server passed a latent in latent mode) |
| `exposure_ref` | `first_frame_ref` | keyframe pictures matched to that clip's exposure |
| `first_audio`, `first_audio_s` | `first_audio`, `first_audio_s` | the clip's last seconds of sound at frame 0; uses `<clip>.lastaud.safetensors` (the old server always used the mp4) |
| `guide_clip` | `guide_clip` | new: `"<anchor>[:frames[:at]]"`, the clip's last frames (22 by default) as one moving keyframe - a motion guide |
| `ref_audios` | `ref_audio` | voices to speak in, `<Audio 1>` ... in the prompt |
| `source`, `regen`, `regen_box` | same | new: regenerate seconds `regen` ("a-b") of a clip (inside a pixel box), keeping the rest; a source shorter than the clip is extended |
| `cond_noise_aug`, `shift_video`, `shift_audio` | same | keyframe trust; the flow shifts (12, 3) |

`ref_images` is refused: reference pictures need the text encoder's vision tower, which the 32B checkpoint in use
does not have. Caveat carried over from the tools: a clip's last latent frame summarises its last four pixel frames,
which is not what encoding one picture gives, so `latent` chaining anchors a slightly different frame than `png`.

## Files

Each finished clip `out/h3_YYYYmmdd_HHMMSS.mp4` comes with its record `.json` (`{job, times, frames, wall,
energy_wh}`, `job.speech_pct` / `speech_s` / `words_per_s` from the clip's sound), and the next clip's anchors:
`.last.png`, `.lastlat.safetensors`, `.lastaud.safetensors` (2 s, the model's own level), `.latents.safetensors`.

The studio keeps `queue.json`, `paused.json` (`{all, projects, batches}`), `job.json` and `llm-mode` in its own
directory (`H3_STUDIO_DIR`).

## Scheduling

First in, first out, skipping held items (held: everything, a project, or a batch). A failed clip is retried once at
the front of the queue; a second failure in a row holds the queue (it is not cleared). A clip whose log has not
moved for `H3_STALL_S` seconds (1200) is stopped. A finished clip is never queued again.

## Language models (`llm.mode`)

The models the box serves when it is not rendering are described in a JSON file, `H3_LLM_MODES`
(`tools/llm-modes.example.json`): name, title, URL and health path (up = one of `up_codes`, or the body names
`model` and `"loaded"`), and the shell commands that start and stop it. `llm.mode {mode}` selects one (`none` stops
them all); with the engine daemon's `H3_LLM_SWITCHER` pointed at `http://127.0.0.1:8095/rpc/llm.mode` the daemon
switches to `none` before it loads and back after it unloads. A model never starts while the engine holds the card.
After 30 min with the GPU idle and the selected model down, the studio starts it.

## The tools

`sycl-h3 speech <text> --character NAME [--characters FILE]` (characters: `tools/characters.example.json` shows the
fields), `sycl-h3 scene <scene.json>`, `sycl-h3 join <label prefix>`, `sycl-h3 speechpct <clip>...` - see `--help`.
