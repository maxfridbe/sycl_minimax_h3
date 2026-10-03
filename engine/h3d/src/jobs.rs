//! What the engine can be asked to do, written once for both ways of asking: the one-shot commands (`h3d bench-blocks
//! ...`, which load, run and exit) and the daemon (`h3d daemon`, which keeps the model loaded between jobs).
//!
//! A job reports through a log callback and looks at a cancel flag between blocks - never inside one: a GPU process
//! stopped in the middle of a kernel can leave the xe driver stuck.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use h3_core::device::{Device, Tensor};
use h3_core::dtype::{f32_to_bf16, DType};
use h3_core::rng::Rng;
use h3_core::safetensors::Checkpoint;
use h3_core::denoiser::{self, Denoiser, Outer, Schedule, Shape, TextRefiner};
use h3_core::{dit, reference, Error, Result};
use serde_json::{json, Value};

pub fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

/// The loaded engine: the GPU, the checkpoint, the denoiser's blocks and the small parts around them on the GPU.
pub struct Engine {
    pub dev: Arc<Device>,
    pub ck: Checkpoint,
    pub model: dit::Blocks,
    pub outer: Outer,
    pub threads: usize,
}

impl Engine {
    /// Opens the GPU and loads the first `count` blocks (all of them when `None`).
    pub fn load(path: &Path, count: Option<usize>, threads: usize, log: &mut dyn FnMut(String)) -> Result<Engine> {
        Engine::load_on(Device::open()?, path, count, threads, log)
    }

    /// The same, on a GPU already opened.
    pub fn load_on(dev: Arc<Device>, path: &Path, count: Option<usize>, threads: usize, log: &mut dyn FnMut(String)) -> Result<Engine> {
        let ck = Checkpoint::open(path)?;
        log(format!("device : {}", dev.name()));
        let model = dit::Blocks::load(&dev, &ck, count, threads)?;
        log(format!("blocks : {} loaded, {:.2} GiB in {:.1} s", model.blocks.len(), gib(model.load_bytes), model.load_seconds));
        let outer = Outer::load(&dev, &ck)?;
        denoiser::check_fits(&model.cfg, &outer)?;
        Ok(Engine { dev, ck, model, outer, threads })
    }

    fn inv_freq(&self) -> Result<Vec<f32>> {
        h3_core::dtype::bytes_to_f32(&self.ck.read("rope.inv_freq")?, self.ck.get("rope.inv_freq")?.dtype)
    }
}

/// How a running job talks back.
pub struct Ctl<'a> {
    pub log: &'a mut dyn FnMut(String),
    pub cancel: &'a AtomicBool,
    /// how far it is: (done, total), e.g. blocks
    pub progress: Option<&'a mut dyn FnMut(usize, usize)>,
}

impl Ctl<'_> {
    fn step(&mut self, done: usize, total: usize) {
        if let Some(p) = self.progress.as_mut() {
            p(done, total)
        }
    }

    fn say(&mut self, s: String) {
        (self.log)(s)
    }
    /// Between blocks: stop here if the job was cancelled. The GPU queue is asynchronous - without a wait, a job
    /// would have queued every block (48 s of work at 47k tokens) long before it looked at the flag - so a block
    /// boundary means: what was queued has run.
    fn check_at(&self, dev: &Device) -> Result<()> {
        dev.wait()?;
        self.check()
    }

    fn check(&self) -> Result<()> {
        if self.cancel.load(Ordering::Relaxed) {
            Err(Error("cancelled".into()))
        } else {
            Ok(())
        }
    }
}

/// Random activations as bfloat16: the bytes for the device and the values they stand for.
pub fn activations(rng: &mut Rng, n: usize) -> (Vec<u8>, Vec<f32>) {
    let mut bytes = Vec::with_capacity(n * 2);
    let mut vals = Vec::with_capacity(n);
    for _ in 0..n {
        let b = f32_to_bf16(rng.normal() * 0.7);
        bytes.extend_from_slice(&b.to_le_bytes());
        vals.push(h3_core::dtype::bf16_to_f32(b));
    }
    (bytes, vals)
}

/// Columns [from, from + width) of every `stride`-wide row.
fn columns(v: &[f32], stride: usize, from: usize, width: usize) -> Vec<f32> {
    v.chunks_exact(stride).flat_map(|r| r[from..from + width].iter().copied()).collect()
}

/// The denoiser's blocks against a dump of the reference pipeline (reference/h3x.py, H3X_DUMP_BLOCK): the host-side
/// tables, every stage of block 0, then the stream after later blocks. Fails when the result drifts.
pub fn check_block(e: &Engine, dump_path: &Path, ctl: &mut Ctl) -> Result<Value> {
    let dump = Checkpoint::open(dump_path)?;
    let mut cfg = e.model.cfg;
    for (key, eps) in [("norm_eps", &mut cfg.norm_eps), ("qk_eps", &mut cfg.qk_eps)] {
        if let Some(v) = dump.metadata.get(key) {
            *eps = v.parse().map_err(|_| Error(format!("the dump's {key} is not a number: {v}")))?;
        }
    }
    if cfg.norm_eps != e.model.cfg.norm_eps || cfg.qk_eps != e.model.cfg.qk_eps {
        return Err(Error("the dump was made with other norm constants than the loaded model".into()));
    }
    let f32s = |name: &str| -> Result<Vec<f32>> { h3_core::dtype::bytes_to_f32(&dump.read(name)?, dump.get(name)?.dtype) };
    fn report(ctl: &mut Ctl, what: &str, got: &[f32], want: &[f32]) -> (f64, f64) {
        let (rel, cos) = reference::compare(got, want);
        ctl.say(format!("  {what:34} rel err {rel:.2e}  cosine {cos:.6}"));
        (rel, cos)
    }

    let rows: Vec<i32> = dump.read("mod_rows")?.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let positions: Vec<f64> = dump.read("position_ids")?.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    let inv_freq = e.inv_freq()?;
    let t_emb = f32s("t_emb")?;
    let tokens = rows.len();
    ctl.say(format!("dump   : {tokens} tokens, {} timestep rows", t_emb.len() / cfg.t_dim));

    ctl.say("host side, against the reference:".into());
    let tables = e.model.blocks[0].tables(&cfg, &t_emb);
    let mut worst = 0f64;
    for (t, name) in tables.iter().zip(["shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp"]) {
        worst = worst.max(reference::compare(t, &f32s(&format!("b0.{name}"))?).0);
    }
    ctl.say(format!("  {:34} rel err {worst:.2e}  (worst of six)", "block 0's tables"));
    // the dump's rotation matrices are [S, pairs, 2, 2] = (cos, -sin, sin, cos)
    let cs = dit::rotations(&positions, &inv_freq);
    let want_cs: Vec<f32> = f32s("rope_table")?.chunks_exact(4).flat_map(|m| [m[0], m[2]]).collect();
    report(ctl, "position rotations (cos, sin)", &cs, &want_cs);

    let dev = &e.dev;
    let step = dit::Step::new(dev, &e.model, &rows, &positions, &inv_freq, &t_emb)?;
    let scratch = dit::Scratch::new(dev, &cfg, tokens, DType::BF16)?;
    let x_in = dump.read("x_in")?;
    let x = Tensor::from_bytes(dev, DType::BF16, &[tokens, cfg.hidden], &x_in)?;

    ctl.say("block 0, stage by stage (each stage runs on this engine's own previous stage):".into());
    let w = cfg.heads * cfg.head_dim;
    let mut worst_cos = 1f64;
    {
        let mut tap = |name: &str, t: &Tensor| -> Result<()> {
            let got = t.to_f32()?;
            let c = match name {
                "qkv_rotated" => {
                    let (_, cq) = report(ctl, "q after norm + rotation", &columns(&got, 3 * w, 0, w), &f32s("b0.q_rope")?);
                    let (_, ck) = report(ctl, "k after norm + rotation", &columns(&got, 3 * w, w, w), &f32s("b0.k_rope")?);
                    cq.min(ck)
                }
                _ => report(ctl, name, &got, &f32s(&format!("b0.{name}"))?).1,
            };
            worst_cos = worst_cos.min(c);
            Ok(())
        };
        e.model.block(0, &x, &step, &scratch, Some(&mut tap))?;
    }

    ctl.say("the stream after later blocks:".into());
    let last = e.model.blocks.len() - 1;
    let mut after = Vec::new();
    for i in 1..=last {
        ctl.check_at(dev)?;
        ctl.step(i, 2 * (last + 1));
        e.model.block(i, &x, &step, &scratch, None)?;
        let key = if dump.metadata.get("last_block").is_some_and(|l| *l == i.to_string()) { "out.last".to_string() } else { format!("out.{i}") };
        if dump.entries.contains_key(&key) {
            let c = report(ctl, &format!("after block {i}"), &x.to_f32()?, &f32s(&key)?).1;
            after.push(json!({"block": i, "cosine": c}));
            worst_cos = worst_cos.min(c);
        }
    }

    // speed: the whole stack again, one wait at the end
    x.buf.write(0, &x_in)?;
    dev.wait()?;
    let t0 = Instant::now();
    for i in 0..=last {
        ctl.check_at(dev)?;
        ctl.step(last + 1 + i, 2 * (last + 1));
        e.model.block(i, &x, &step, &scratch, None)?;
    }
    dev.wait()?;
    let s = t0.elapsed().as_secs_f64();
    ctl.say(format!("speed  : {} blocks on {tokens} tokens in {:.2} s ({:.1} ms per block)", last + 1, s, s * 1e3 / (last + 1) as f64));
    if worst_cos > 0.99 {
        Ok(json!({"tokens": tokens, "worst_cosine": worst_cos, "after": after, "seconds": s}))
    } else {
        Err(Error(format!("the engine's result drifts from the reference (worst cosine {worst_cos:.4})")))
    }
}

/// The block stack on made-up tokens at a chosen sequence length: what one denoiser step costs, and where.
/// `blocks`: run only the first that many (all the loaded ones when `None`).
pub fn bench_blocks(e: &Engine, tokens: usize, blocks: Option<usize>, ctl: &mut Ctl) -> Result<Value> {
    let cfg = e.model.cfg;
    let dev = &e.dev;
    let mut rng = Rng::new(0);
    let rows: Vec<i32> = (0..tokens).map(|i| if i < tokens / 20 { 1 } else { 0 }).collect();
    let positions: Vec<f64> = (0..tokens * 3).map(|_| (rng.uniform() * 64.0) as f64).collect();
    let t_emb: Vec<f32> = (0..cfg.t_dim).map(|_| rng.normal()).collect();
    let step = dit::Step::new(dev, &e.model, &rows, &positions, &e.inv_freq()?, &t_emb)?;
    let scratch = dit::Scratch::new(dev, &cfg, tokens, DType::BF16)?;
    // made-up input: a million random values repeated (drawing all 250 million at 47k tokens took the slow host
    // half a minute, during which a cancel could not get through)
    let (pattern, _) = activations(&mut rng, 1 << 20);
    let xb: Vec<u8> = pattern.iter().copied().cycle().take(tokens * cfg.hidden * 2).collect();
    ctl.check()?;
    let x = Tensor::from_bytes(dev, DType::BF16, &[tokens, cfg.hidden], &xb)?;
    let n = blocks.unwrap_or(e.model.blocks.len()).min(e.model.blocks.len());

    // warm-up: the first call of each kernel shape compiles it
    e.model.block(0, &x, &step, &scratch, None)?;
    dev.wait()?;
    x.buf.write(0, &xb)?;
    let t0 = Instant::now();
    for i in 0..n {
        ctl.check_at(dev)?;
        ctl.step(i, 2 * n);
        e.model.block(i, &x, &step, &scratch, None)?;
    }
    dev.wait()?;
    let total = t0.elapsed().as_secs_f64();
    ctl.say(format!(
        "speed  : {n} blocks on {tokens} tokens in {total:.2} s ({:.1} ms per block), {:.1} of {:.1} GiB in use",
        total * 1e3 / n as f64,
        gib(dev.mem_used()),
        gib(dev.mem_cap())
    ));

    // where it goes: the same again, waiting for the device after every stage
    x.buf.write(0, &xb)?;
    let mut stages: Vec<(String, f64)> = Vec::new();
    let mut mark = Instant::now();
    {
        let mut tap = |name: &str, t: &Tensor| -> Result<()> {
            t.buf.device().wait()?;
            let dt = mark.elapsed().as_secs_f64();
            match stages.iter_mut().find(|(n, _)| n == name) {
                Some(s) => s.1 += dt,
                None => stages.push((name.to_string(), dt)),
            }
            mark = Instant::now();
            Ok(())
        };
        for i in 0..n {
            ctl.check_at(dev)?;
            if let Some(p) = ctl.progress.as_mut() {
                p(n + i, 2 * n);
            }
            e.model.block(i, &x, &step, &scratch, Some(&mut tap))?;
        }
    }
    let sum: f64 = stages.iter().map(|s| s.1).sum();
    let label = |n: &str| match n {
        "h1" | "h2" => "norm + scale/shift",
        "qkv" => "linear: q, k, v",
        "qkv_rotated" => "per-head norm + rotation (q, k)",
        "att" => "attention",
        "attn_out" => "linear: attention out",
        "x1" | "x2" => "gated add",
        "fc1" => "linear: MLP in",
        "mlp" => "gated activation + linear: MLP out",
        _ => "other",
    };
    let mut merged: Vec<(&str, f64)> = Vec::new();
    for (n, s) in &stages {
        match merged.iter_mut().find(|(l, _)| *l == label(n)) {
            Some(m) => m.1 += s,
            None => merged.push((label(n), *s)),
        }
    }
    merged.sort_by(|a, b| b.1.total_cmp(&a.1));
    ctl.say("stages (a wait after each, so the sum is a little above the run without):".into());
    let mut out = serde_json::Map::new();
    for (l, s) in merged {
        ctl.say(format!("  {l:36} {:7.1} ms per block  {:4.1}%", s * 1e3 / n as f64, s / sum * 100.0));
        out.insert(l.to_string(), json!(s * 1e3 / n as f64));
    }
    Ok(json!({"tokens": tokens, "blocks": n, "seconds": total, "ms_per_block": total * 1e3 / n as f64,
              "gib_in_use": gib(dev.mem_used()), "stages_ms_per_block": out}))
}

/// A whole sampling run against a run dump of the reference pipeline (reference/h3x.py, H3X_DUMP_RUN): the same
/// text conditioning, starting noise and schedule; every step's denoised estimate compared with the reference's,
/// then the finished latents. `out`: where to write the latents (for the reference's decoders).
pub fn denoise(e: &Engine, dump_path: &Path, out: Option<&Path>, ctl: &mut Ctl) -> Result<Value> {
    let t_all = Instant::now();
    let dump = Checkpoint::open(dump_path)?;
    let f32s = |name: &str| -> Result<Vec<f32>> { h3_core::dtype::bytes_to_f32(&dump.read(name)?, dump.get(name)?.dtype) };
    let meta = |k: &str, d: f32| -> f32 { dump.metadata.get(k).and_then(|v| v.parse().ok()).unwrap_or(d) };
    let schedule = Schedule { shift_video: meta("shift", 12.0), shift_audio: meta("audio_shift", 3.0), audio_scale: meta("audio_scale", 4.0) };
    if e.model.blocks.len() != e.model.cfg.blocks || dump.metadata.get("keyframes").is_some_and(|k| k == "True") {
        return Err(Error("denoise needs every block loaded and a run without keyframes".into()));
    }
    let vs = &dump.get("noise.video")?.shape; // [1, C, T, H, W]
    let as_ = &dump.get("noise.audio")?.shape; // [1, C, 2, T]
    let shape = Shape { t: vs[2], h: vs[3], w: vs[4], audio_t: as_[3] };
    let cs = &dump.get("context")?.shape; // [1, L, text_dim]
    let (l, text_dim) = (cs[1], cs[2]);
    let tags: Option<Vec<i32>> = match dump.entries.contains_key("token_tags") {
        true => Some(dump.read("token_tags")?.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
        false => None,
    };
    let sigmas = f32s("sigmas")?;
    let own = denoiser::sigmas(sigmas.len() - 1, schedule.shift_video);
    let worst_sigma = own.iter().zip(&sigmas).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    ctl.say(format!("dump   : {l} text tokens, video latent {}x{}x{}, audio {} frames, {} steps (schedule differs by {worst_sigma:.1e})",
                    shape.t, shape.h, shape.w, shape.audio_t, sigmas.len() - 1));

    // the text: projection + refiner, loaded for this and dropped
    let t0 = Instant::now();
    let text = {
        let refiner = TextRefiner::load(&e.dev, &e.ck, e.threads)?;
        let ctx = Tensor::from_bytes(&e.dev, DType::BF16, &[l, text_dim], &denoiser::bf16_bytes(&f32s("context")?))?;
        let t = refiner.run(&e.model.cfg, &ctx)?;
        e.dev.wait()?;
        t
    };
    ctl.say(format!("text   : refined in {:.1} s (refiner loaded and freed)", t0.elapsed().as_secs_f64()));
    ctl.check()?;

    // the starting noise: drawn here as PyTorch draws it (same seed), checked against the dump's
    let seed: u64 = dump.metadata.get("seed").and_then(|v| v.parse().ok()).unwrap_or(0);
    let (want_v, want_a) = (f32s("noise.video")?, f32s("noise.audio")?);
    let (noise_v, noise_a) = h3_core::noise::clip_noise(seed, want_v.len(), want_a.len());
    let worst_noise = noise_v.iter().zip(&want_v).chain(noise_a.iter().zip(&want_a)).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    ctl.say(format!("noise  : seed {seed}, drawn by the engine; differs from the reference's by {worst_noise:.1e} at most"));
    if worst_noise > 1e-4 {
        return Err(Error("the engine's starting noise is not the reference's".into()));
    }

    let mut d = Denoiser::new(&e.model, &e.outer, text, tags, shape, schedule)?;
    let steps = sigmas.len() - 1;
    let nblocks = e.model.blocks.len();
    ctl.say(format!("tokens : {}", d.tokens()));
    let mut per_step = Vec::new();
    let mut worst = 1f64;
    let mut t_step = Instant::now();
    let (v, a) = {
        let dev = e.dev.clone();
        let cancel = ctl.cancel;
        let mut progress = ctl.progress.take();
        let mut lines: Vec<String> = Vec::new();
        let r = denoiser::sample(
            &mut d,
            &noise_v,
            &noise_a,
            &sigmas,
            &mut |i, dv, da| {
                let secs = t_step.elapsed().as_secs_f64();
                t_step = Instant::now();
                let key = format!("x0.{:02}", i + 1);
                let (mut cv, mut ca) = (f64::NAN, f64::NAN);
                if dump.entries.contains_key(&format!("{key}.video")) {
                    cv = reference::compare(dv, &f32s(&format!("{key}.video"))?).1;
                    ca = reference::compare(da, &f32s(&format!("{key}.audio"))?).1;
                    worst = worst.min(cv).min(ca);
                }
                lines.push(format!("  step {}/{steps}  sigma {:.4}  {secs:6.2} s   cosine to the reference: video {cv:.5}  audio {ca:.5}", i + 1, sigmas[i]));
                per_step.push(json!({"step": i + 1, "seconds": secs, "cosine_video": cv, "cosine_audio": ca}));
                Ok(())
            },
            &mut |i, b| {
                if b == 0 || b == nblocks / 2 {
                    dev.wait()?;
                    if cancel.load(Ordering::Relaxed) {
                        return Err(Error("cancelled".into()));
                    }
                }
                if let Some(p) = progress.as_mut() {
                    p(i * nblocks + b, steps * nblocks);
                }
                Ok(())
            },
        );
        ctl.progress = progress;
        for s in lines {
            ctl.say(s);
        }
        r?
    };
    let cv = reference::compare(&v, &f32s("samples.video")?);
    let ca = reference::compare(&a, &f32s("samples.audio")?);
    ctl.say(format!("latents: video rel err {:.2e} cosine {:.5}; audio rel err {:.2e} cosine {:.5}", cv.0, cv.1, ca.0, ca.1));
    if let Some(p) = out {
        let mut t = BTreeMap::new();
        t.insert("samples.video".to_string(), (dump.get("samples.video")?.shape.clone(), v));
        t.insert("samples.audio".to_string(), (dump.get("samples.audio")?.shape.clone(), a));
        h3_core::safetensors::write_f32(p, &t, &dump.metadata)?;
        ctl.say(format!("written: {}", p.display()));
    }
    let secs = t_all.elapsed().as_secs_f64();
    ctl.say(format!("total  : {secs:.1} s"));
    Ok(json!({"tokens": d.tokens(), "steps": per_step, "cosine_video": cv.1, "cosine_audio": ca.1, "worst_step_cosine": worst, "seconds": secs}))
}

/// The decoders' checkpoints: video, and audio (optional: a silent clip without it).
pub struct Vaes<'a> {
    pub video: &'a Path,
    pub audio: Option<&'a Path>,
    /// the latent upscaler and its factor (the video latents are upscaled before the video decoder)
    pub upscale: Option<(&'a Path, f32)>,
}

/// Latents -> a clip: the video decoder (and the audio decoder) on the latents in `latents` (a `.safetensors` with
/// `samples.video` / `samples.audio`, as `denoise` writes it, or `latents.*`), written to `out` (.mp4). `check`: a
/// decode dump of the reference (h3x.py H3X_DUMP_DECODE) to compare the frames and the sound with.
pub fn decode(dev: &Arc<Device>, threads: usize, latents: &Path, vaes: &Vaes, out: Option<&Path>, check: Option<&Path>, ctl: &mut Ctl) -> Result<Value> {
    let lat = Checkpoint::open(latents)?;
    let key = ["samples.video", "latents.video"].into_iter().find(|k| lat.entries.contains_key(*k)).ok_or("the latents file has no samples.video")?;
    let shape = lat.get(key)?.shape.clone(); // [1, 24, T, H, W]
    let video = h3_core::dtype::bytes_to_f32(&lat.read(key)?, lat.get(key)?.dtype)?;
    let audio = match ["samples.audio", "latents.audio"].into_iter().find(|k| lat.entries.contains_key(*k)) {
        Some(k) => Some((h3_core::dtype::bytes_to_f32(&lat.read(k)?, lat.get(k)?.dtype)?, lat.get(k)?.shape[3])), // [1, 32, 2, T]
        None => None,
    };
    decode_latents(dev, threads, Latents { video, t: shape[2], h: shape[3], w: shape[4], audio }, vaes, out, check, ctl)
}

/// A clip's finished latents: video [24, t, h, w], and audio [32, 2, T] with its T.
pub struct Latents {
    pub video: Vec<f32>,
    pub t: usize,
    pub h: usize,
    pub w: usize,
    pub audio: Option<(Vec<f32>, usize)>,
}

/// Latents in memory -> frames and sound (and an .mp4); see `decode`.
pub fn decode_latents(dev: &Arc<Device>, threads: usize, lat: Latents, vaes: &Vaes, out: Option<&Path>, check: Option<&Path>, ctl: &mut Ctl) -> Result<Value> {
    let vae = vaes.video;
    let t_all = Instant::now();
    let (t, mut h, mut w) = (lat.t, lat.h, lat.w);
    let mut z = lat.video;
    let mut report = json!({});
    if let Some((up, s)) = vaes.upscale {
        let t0 = Instant::now();
        let u = h3_core::upscale::Upscaler::load(dev, &Checkpoint::open(up)?)?;
        let cancel = ctl.cancel;
        let (z2, ho, wo) = u.upscale(&z, t, h, w, s, &mut || {
            if cancel.load(Ordering::Relaxed) {
                return Err(Error("cancelled".into()));
            }
            Ok(())
        })?;
        let secs = t0.elapsed().as_secs_f64();
        ctl.say(format!("upscale: latents {w}x{h} -> {wo}x{ho} (x{s}) in {secs:.1} s"));
        report["upscale_seconds"] = json!(secs);
        if let Some(c) = check {
            let d = Checkpoint::open(c)?;
            let want = h3_core::dtype::bytes_to_f32(&d.read("latents.video")?, d.get("latents.video")?.dtype)?;
            if want.len() == z2.len() {
                let (rel, cos) = reference::compare(&z2, &want);
                ctl.say(format!("check  : against the reference's upscaled latents: rel err {rel:.2e}, cosine {cos:.6}"));
                report["upscale_cosine"] = json!(cos);
            } else {
                ctl.say(format!("check  : the reference's latents have {} values ({:?}), ours {}", want.len(), d.get("latents.video")?.shape, z2.len()));
            }
        }
        drop(u);
        (z, h, w) = (z2, ho, wo);
    }
    let ck = Checkpoint::open(vae)?;
    let dec = h3_core::vae::VideoDecoder::load(dev, &ck, threads)?;
    ctl.say(format!("vae    : {} loaded, {:.2} GiB in {:.1} s", vae.display(), gib(dec.load_bytes), dec.load_seconds));
    ctl.say(format!("latents: video {t}x{h}x{w}"));
    let t0 = Instant::now();
    let mut tiles = 0usize;
    let cancel = ctl.cancel;
    let (px, frames) = dec.decode(&z, t, h, w, &mut || {
        tiles += 1;
        dev.wait()?;
        if cancel.load(Ordering::Relaxed) {
            return Err(Error("cancelled".into()));
        }
        Ok(())
    })?;
    let secs = t0.elapsed().as_secs_f64();
    let (fh, fw) = (h * 16, w * 16);
    ctl.say(format!("decoded: {frames} frames of {fw}x{fh} in {secs:.1} s ({tiles} tiles)"));
    for (k, v) in [("frames", json!(frames)), ("width", json!(fw)), ("height", json!(fh)), ("seconds", json!(secs)), ("tiles", json!(tiles))] {
        report[k] = v;
    }
    if let Some(c) = check {
        let d = Checkpoint::open(c)?;
        let want = h3_core::dtype::bytes_to_f32(&d.read("images")?, d.get("images")?.dtype)?; // [.., F, H, W, 3]
        if want.len() != px.len() {
            return Err(Error(format!("the reference decoded {} values ({:?}), this engine {}", want.len(), d.get("images")?.shape, px.len())));
        }
        let plane = fh * fw;
        // ours planar [3, F, H, W] -> the reference's [F, H, W, 3]
        let mut ours = vec![0f32; px.len()];
        for c in 0..3 {
            for fi in 0..frames {
                for i in 0..plane {
                    ours[(fi * plane + i) * 3 + c] = px[(c * frames + fi) * plane + i];
                }
            }
        }
        let (rel, cos) = reference::compare(&ours, &want);
        let mse = ours.iter().zip(&want).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>() / ours.len() as f64;
        let psnr = 10.0 * (1.0 / mse.max(1e-20)).log10();
        ctl.say(format!("check  : against the reference's frames: rel err {rel:.2e}, cosine {cos:.6}, PSNR {psnr:.1} dB"));
        report["psnr"] = json!(psnr);
        report["cosine"] = json!(cos);
    }
    // the sound
    let mut sound = None;
    if let Some(ap) = vaes.audio {
        let (za, at) = lat.audio.as_ref().ok_or("the latents have no audio")?;
        let (za, at) = (za, *at);
        let t0 = Instant::now();
        let ad = h3_core::audio::AudioDecoder::load(dev, &Checkpoint::open(ap)?)?;
        let ch = ad.decode(za, at, &mut || ctl.check())?;
        let secs = t0.elapsed().as_secs_f64();
        ctl.say(format!("audio  : {at} latent frames -> {} samples x 2 in {secs:.1} s", ch[0].len()));
        report["audio_seconds"] = json!(secs);
        if let Some(c) = check {
            let d = Checkpoint::open(c)?;
            if d.entries.contains_key("waveform") {
                let want = h3_core::dtype::bytes_to_f32(&d.read("waveform")?, d.get("waveform")?.dtype)?; // [1, L, 2]
                let ours: Vec<f32> = (0..ch[0].len()).flat_map(|i| [ch[0][i], ch[1][i]]).collect();
                if want.len() == ours.len() {
                    let (rel, cos) = reference::compare(&ours, &want);
                    ctl.say(format!("check  : against the reference's sound: rel err {rel:.2e}, cosine {cos:.6}"));
                    report["audio_cosine"] = json!(cos);
                } else {
                    ctl.say(format!("check  : the reference's sound has {} values, ours {}", want.len(), ours.len()));
                }
            }
        }
        sound = Some(crate::media::Audio { channels: ch, sample_rate: h3_core::audio::SAMPLE_RATE });
    }
    if let Some(o) = out {
        crate::media::write_mp4(o, &crate::media::Frames { px: &px, count: frames, height: fh, width: fw, fps: 24 }, sound.as_ref(), ctl.log)?;
        ctl.say(format!("written: {}", o.display()));
    }
    ctl.say(format!("total  : {:.1} s", t_all.elapsed().as_secs_f64()));
    Ok(report)
}

/// The text encoder's files: its GGUF weights and the tokenizer's directory.
pub struct TeFiles<'a> {
    pub te: &'a Path,
    pub tokenizer: &'a Path,
}

/// A prompt -> the denoiser's text conditioning: the tokenizer and the text encoder, streamed layer by layer.
/// `check`: a run dump whose `context` is the reference's conditioning for the same prompt. `out`: a `.safetensors`
/// with `context` [1, L, 5120] and `token_tags`.
pub fn encode(dev: &Arc<Device>, threads: usize, prompt: &str, files: &TeFiles, out: Option<&Path>, check: Option<&Path>, ctl: &mut Ctl) -> Result<Value> {
    let t_all = Instant::now();
    let (ctx, ids, hidden, secs) = encode_prompt(dev, threads, prompt, files, ctl)?;
    let mut report = json!({"tokens": ids.len(), "seconds": secs});
    if let Some(c) = check {
        let d = Checkpoint::open(c)?;
        let want = h3_core::dtype::bytes_to_f32(&d.read("context")?, d.get("context")?.dtype)?;
        if want.len() == ctx.len() {
            let (rel, cos) = reference::compare(&ctx, &want);
            ctl.say(format!("check  : against the reference's conditioning: rel err {rel:.2e}, cosine {cos:.6}"));
            // per token, the worst
            let worst = ctx.chunks(hidden).zip(want.chunks(hidden)).map(|(a, b)| reference::compare(a, b).1).fold(1f64, f64::min);
            ctl.say(format!("check  : the worst token's cosine {worst:.6}"));
            report["cosine"] = json!(cos);
            report["worst_token_cosine"] = json!(worst);
        } else {
            ctl.say(format!("check  : the reference's conditioning has {} values ({:?}), ours {}", want.len(), d.get("context")?.shape, ctx.len()));
        }
    }
    if let Some(o) = out {
        let mut t = BTreeMap::new();
        t.insert("context".to_string(), (vec![1, ids.len(), hidden], ctx));
        t.insert("token_tags".to_string(), (vec![ids.len()], vec![1.0; ids.len()]));
        h3_core::safetensors::write_f32(o, &t, &BTreeMap::from([("prompt".to_string(), prompt.to_string())]))?;
        ctl.say(format!("written: {}", o.display()));
    }
    ctl.say(format!("total  : {:.1} s", t_all.elapsed().as_secs_f64()));
    Ok(report)
}

/// The tokenizer and the streamed text encoder: (conditioning [L, hidden], token ids, hidden, seconds).
pub fn encode_prompt(dev: &Arc<Device>, threads: usize, prompt: &str, files: &TeFiles, ctl: &mut Ctl) -> Result<(Vec<f32>, Vec<u32>, usize, f64)> {
    let (te, tokenizer) = (files.te, files.tokenizer);
    let tok = h3_core::tokenizer::Tokenizer::load(tokenizer)?;
    let mut ids = tok.encode(prompt)?;
    if ids.is_empty() {
        ids.push(151643); // the reference encodes an empty prompt as one pad token
    }
    ctl.say(format!("prompt : {} tokens", ids.len()));
    let enc = h3_core::te::TextEncoder::open(te)?;
    ctl.say(format!("te     : {} layers, hidden {}, {} query / {} key-value heads (streamed from {})", enc.layers, enc.hidden, enc.heads, enc.kv_heads, te.display()));
    let t0 = Instant::now();
    let cancel = ctl.cancel;
    let layers = enc.layers;
    let mut progress = ctl.progress.take();
    let r = enc.encode(dev, &ids, threads, &mut |i| {
        if cancel.load(Ordering::Relaxed) {
            return Err(Error("cancelled".into()));
        }
        if let Some(p) = progress.as_mut() {
            p(i, layers);
        }
        Ok(())
    });
    ctl.progress = progress;
    let ctx = r?;
    let secs = t0.elapsed().as_secs_f64();
    ctl.say(format!("encoded: {} x {} in {secs:.1} s", ids.len(), enc.hidden));
    Ok((ctx, ids, enc.hidden, secs))
}

/// What a `generate` job makes: the prompt, the canvas, the length, the sampling, the files it reads.
pub struct ClipSpec<'a> {
    pub prompt: String,
    pub width: usize,
    pub height: usize,
    pub seconds: f64,
    pub seed: u64,
    pub steps: usize,
    pub te: TeFiles<'a>,
    pub vaes: Vaes<'a>,
}

/// The model's frame grid: frames at 24 fps snapped up to 17k + 5; (frames, video latent frames, audio latent frames).
pub fn temporal_shape(seconds: f64) -> (usize, usize, usize) {
    let mut n = ((seconds * 24.0).round() as usize).max(5);
    while n % 17 != 5 {
        n += 1;
    }
    let lt = if n <= 5 { 2 } else { (n - 5) / 17 * 5 + 2 };
    (n, lt, (n as f64 / 24.0 * 40.0).round() as usize)
}

/// A whole clip in the engine: the prompt through the text encoder, the sampler on the loaded denoiser, the
/// (upscaler and) decoders, an .mp4.
pub fn generate(e: &Engine, c: &ClipSpec, out: &Path, ctl: &mut Ctl) -> Result<Value> {
    let t_all = Instant::now();
    if !c.width.is_multiple_of(32) || !c.height.is_multiple_of(32) {
        return Err(Error(format!("the canvas must be a multiple of 32 on both sides, got {}x{}", c.width, c.height)));
    }
    let (frames, lt, at) = temporal_shape(c.seconds);
    let shape = Shape { t: lt, h: c.height / 16, w: c.width / 16, audio_t: at };
    ctl.say(format!("clip   : {}x{}, {frames} frames ({:.2} s), {} steps, seed {}", c.width, c.height, frames as f64 / 24.0, c.steps, c.seed));
    let (ctx, ids, hidden, te_secs) = encode_prompt(&e.dev, e.threads, &c.prompt, &c.te, ctl)?;
    ctl.check()?;
    let t0 = Instant::now();
    let text = {
        let refiner = TextRefiner::load(&e.dev, &e.ck, e.threads)?;
        let t = Tensor::from_bytes(&e.dev, DType::BF16, &[ids.len(), hidden], &denoiser::bf16_bytes(&ctx))?;
        let r = refiner.run(&e.model.cfg, &t)?;
        e.dev.wait()?;
        r
    };
    ctl.say(format!("text   : refined in {:.1} s", t0.elapsed().as_secs_f64()));
    let schedule = Schedule::default();
    let mut d = Denoiser::new(&e.model, &e.outer, text, None, shape, schedule)?;
    let n_v = 24 * shape.t * shape.h * shape.w;
    let (noise_v, noise_a) = h3_core::noise::clip_noise(c.seed, n_v, 32 * 2 * shape.audio_t);
    let sigmas = denoiser::sigmas(c.steps, schedule.shift_video);
    ctl.say(format!("tokens : {} in the denoiser", d.tokens()));
    let nblocks = e.model.blocks.len();
    let steps = c.steps;
    let t0 = Instant::now();
    let (v, a) = {
        let dev = e.dev.clone();
        let cancel = ctl.cancel;
        let mut progress = ctl.progress.take();
        let mut times = Vec::new();
        let mut t_step = Instant::now();
        let r = denoiser::sample(&mut d, &noise_v, &noise_a, &sigmas, &mut |_, _, _| {
            times.push(t_step.elapsed().as_secs_f64());
            t_step = Instant::now();
            Ok(())
        }, &mut |i, b| {
            if b == 0 || b == nblocks / 2 {
                dev.wait()?;
                if cancel.load(Ordering::Relaxed) {
                    return Err(Error("cancelled".into()));
                }
            }
            if let Some(p) = progress.as_mut() {
                p(i * nblocks + b, steps * nblocks);
            }
            Ok(())
        });
        ctl.progress = progress;
        let (v, a) = r?;
        ctl.say(format!("sampled: {steps} steps in {:.1} s ({})", t0.elapsed().as_secs_f64(), times.iter().map(|s| format!("{s:.2}")).collect::<Vec<_>>().join(" ")));
        (v, a)
    };
    let sample_secs = t0.elapsed().as_secs_f64();
    drop(d);
    let mut rep = decode_latents(&e.dev, e.threads, Latents { video: v, t: shape.t, h: shape.h, w: shape.w, audio: Some((a, shape.audio_t)) }, &c.vaes, Some(out), None, ctl)?;
    let total = t_all.elapsed().as_secs_f64();
    rep["text_encoder_seconds"] = json!(te_secs);
    rep["sample_seconds"] = json!(sample_secs);
    rep["total_seconds"] = json!(total);
    ctl.say(format!("clip   : {} in {total:.1} s", out.display()));
    Ok(rep)
}

/// One job, as the daemon receives it: `{"kind": "bench-blocks", "tokens": 47173}` and so on.
pub fn run(e: &Engine, spec: &Value, ctl: &mut Ctl) -> Result<Value> {
    let kind = spec.get("kind").and_then(|k| k.as_str()).ok_or("a job needs a \"kind\"")?;
    let num = |k: &str| spec.get(k).and_then(|v| v.as_u64()).map(|v| v as usize);
    match kind {
        "bench-blocks" => bench_blocks(e, num("tokens").unwrap_or(16500), num("blocks"), ctl),
        "check-block" => {
            let dump = spec.get("dump").and_then(|d| d.as_str()).ok_or("check-block needs \"dump\": a path the engine can read")?;
            check_block(e, Path::new(dump), ctl)
        }
        "denoise" => {
            let dump = spec.get("dump").and_then(|d| d.as_str()).ok_or("denoise needs \"dump\": a run dump the engine can read")?;
            let out = spec.get("out").and_then(|d| d.as_str()).map(Path::new);
            denoise(e, Path::new(dump), out, ctl)
        }
        "decode" => {
            let s = |k: &str| spec.get(k).and_then(|d| d.as_str()).map(Path::new);
            let latents = s("latents").ok_or("decode needs \"latents\": a latents file the engine can read")?;
            let video = s("vae").ok_or("decode needs \"vae\": the video decoder's checkpoint")?;
            let upscale = s("upscaler").map(|p| (p, spec.get("upscale").and_then(|v| v.as_f64()).unwrap_or(2.0) as f32));
            decode(&e.dev, e.threads, latents, &Vaes { video, audio: s("audio_vae"), upscale }, s("out"), s("check"), ctl)
        }
        "generate" => {
            let s = |k: &str| spec.get(k).and_then(|d| d.as_str());
            let n = |k: &str| spec.get(k).and_then(|d| d.as_f64());
            let prompt = match (s("prompt"), s("prompt_file")) {
                (Some(p), _) => p.to_string(),
                (None, Some(f)) => std::fs::read_to_string(f).map_err(|e| Error(format!("{f}: {e}")))?.trim().to_string(),
                _ => return Err(Error("generate needs \"prompt\" or \"prompt_file\"".into())),
            };
            let out = s("out").ok_or("generate needs \"out\": where to write the .mp4")?;
            let upscale = n("upscale").filter(|u| *u > 1.0);
            let c = ClipSpec {
                prompt,
                width: n("width").unwrap_or(384.0) as usize,
                height: n("height").unwrap_or(288.0) as usize,
                seconds: n("seconds").unwrap_or(2.0),
                seed: n("seed").unwrap_or(0.0) as u64,
                steps: n("steps").unwrap_or(8.0) as usize,
                te: TeFiles {
                    te: Path::new(s("te").unwrap_or("/models/teacher/qwen3vl_32b_minimax_h3-Q4_K_M.gguf")),
                    tokenizer: Path::new(s("tokenizer").unwrap_or("/app/tokenizer")),
                },
                vaes: Vaes {
                    video: Path::new(s("vae").unwrap_or("/models/Comfy-Org-MiniMax-H3/vae/minimax_h3_video_vae_fp16.safetensors")),
                    audio: Some(Path::new(s("audio_vae").unwrap_or("/models/Comfy-Org-MiniMax-H3/vae/minimax_h3_audio_vae_fp32.safetensors"))),
                    upscale: upscale.map(|u| (Path::new(s("upscaler").unwrap_or("/models/upscaler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors")), u as f32)),
                },
            };
            generate(e, &c, Path::new(out), ctl)
        }
        "encode" => {
            let s = |k: &str| spec.get(k).and_then(|d| d.as_str());
            let prompt = match (s("prompt"), s("prompt_file")) {
                (Some(p), _) => p.to_string(),
                (None, Some(f)) => std::fs::read_to_string(f).map_err(|e| Error(format!("{f}: {e}")))?.trim().to_string(),
                _ => return Err(Error("encode needs \"prompt\" or \"prompt_file\"".into())),
            };
            let te = s("te").ok_or("encode needs \"te\": the text encoder's GGUF file")?;
            let tokenizer = s("tokenizer").unwrap_or("/app/tokenizer");
            encode(&e.dev, e.threads, &prompt, &TeFiles { te: Path::new(te), tokenizer: Path::new(tokenizer) }, s("out").map(Path::new), s("check").map(Path::new), ctl)
        }
        other => Err(Error(format!("unknown job kind {other:?} (known: bench-blocks, check-block, denoise, decode, encode, generate)"))),
    }
}
