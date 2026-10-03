//! What the engine can be asked to do, written once for both ways of asking: the one-shot commands (`h3 bench-blocks
//! ...`, which load, run and exit) and the daemon (`h3 serve`, which keeps the model loaded between jobs).
//!
//! A job reports through a log callback and looks at a cancel flag between blocks - never inside one: a GPU process
//! stopped in the middle of a kernel can leave the xe driver stuck.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use h3_core::device::{Device, Tensor};
use h3_core::dtype::{f32_to_bf16, DType};
use h3_core::rng::Rng;
use h3_core::safetensors::Checkpoint;
use h3_core::{dit, reference, Error, Result};
use serde_json::{json, Value};

pub fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

/// The loaded engine: the GPU, the checkpoint, the denoiser's blocks on the GPU.
pub struct Engine {
    pub dev: Arc<Device>,
    pub ck: Checkpoint,
    pub model: dit::Blocks,
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
        Ok(Engine { dev, ck, model })
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
        other => Err(Error(format!("unknown job kind {other:?} (known: bench-blocks, check-block)"))),
    }
}
