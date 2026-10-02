//! `h3` - MiniMax H3 on an Intel Arc GPU.
//!
//!     h3 version                              yy.mmdd.###
//!     h3 device                               the GPU the kernels found, and its memory cap
//!     h3 info <checkpoint.safetensors>        what is in a checkpoint
//!     h3 load <checkpoint> [--threads 8]      load it onto the GPU, timed
//!     h3 check-linear <checkpoint> [--block 0] [--rows 64] [--bench-rows 16384]
//!                                             a block's int8 linears: the GPU against the CPU reference, and timed
//!     h3 check-block <checkpoint> <dump> [--blocks N]
//!                                             the denoiser.s blocks against a dump of the reference pipeline
//!     h3 bench-blocks <checkpoint> [--tokens 16500] [--blocks N]
//!                                             what a denoiser step costs at a sequence length, stage by stage

use std::collections::BTreeMap;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use h3_core::device::{Device, Tensor};
use h3_core::dtype::{f32_to_bf16, DType};
use h3_core::ops::Int8Linear;
use h3_core::rng::Rng;
use h3_core::safetensors::Checkpoint;
use h3_core::{dit, load, reference, Error, Result};

/// The repository's version, yy.mmdd.### (Cargo carries it without the leading zeros).
fn version() -> String {
    let p: Vec<u32> = env!("CARGO_PKG_VERSION").split('.').map(|x| x.parse().unwrap_or(0)).collect();
    format!("{:02}.{:04}.{:03}", p[0], p[1], p[2])
}

const USAGE: &str = "usage:
  h3 version
  h3 device
  h3 info <checkpoint.safetensors>
  h3 load <checkpoint.safetensors> [--threads 8]
  h3 check-linear <checkpoint.safetensors> [--block 0] [--rows 64] [--bench-rows 16384]
  h3 check-block <checkpoint.safetensors> <dump.safetensors> [--blocks N] [--threads 8]
  h3 bench-blocks <checkpoint.safetensors> [--tokens 16500] [--blocks N]";

/// `--name value` options after the positional arguments.
struct Args {
    positional: Vec<String>,
    options: BTreeMap<String, String>,
}

impl Args {
    fn parse(raw: &[String]) -> Result<Args> {
        let (mut positional, mut options) = (Vec::new(), BTreeMap::new());
        let mut it = raw.iter();
        while let Some(a) = it.next() {
            match a.strip_prefix("--") {
                Some(name) => {
                    let v = it.next().ok_or_else(|| Error(format!("--{name} needs a value")))?;
                    options.insert(name.to_string(), v.clone());
                }
                None => positional.push(a.clone()),
            }
        }
        Ok(Args { positional, options })
    }

    fn path(&self, i: usize) -> Result<&Path> {
        self.positional.get(i).map(|s| Path::new(s.as_str())).ok_or_else(|| Error(USAGE.into()))
    }

    fn number(&self, name: &str, default: usize) -> Result<usize> {
        match self.options.get(name) {
            None => Ok(default),
            Some(v) => v.parse().map_err(|_| Error(format!("--{name}: {v} is not a number"))),
        }
    }
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

fn cmd_device() -> Result<()> {
    let dev = Device::open()?;
    println!("device : {}", dev.name());
    println!("memory : the engine will allocate up to {:.1} GiB", gib(dev.mem_cap()));
    Ok(())
}

fn cmd_info(args: &Args) -> Result<()> {
    let ck = Checkpoint::open(args.path(0)?)?;
    println!("{}: {:.2} GiB, {} tensors", ck.path.display(), gib(ck.file_bytes), ck.entries.len());
    let mut by_type: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    for e in ck.entries.values() {
        let s = by_type.entry(format!("{:?}", e.dtype)).or_default();
        s.0 += 1;
        s.1 += e.bytes as u64;
    }
    for (t, (n, b)) in &by_type {
        println!("  {t:5} {n:5} tensors {:8.2} GiB", gib(*b));
    }
    let blocks = ck.entries.keys().filter_map(|k| k.strip_prefix("blocks.")?.split('.').next()?.parse::<usize>().ok()).max();
    if let Some(last) = blocks {
        println!("  denoiser blocks: {}", last + 1);
        for layer in ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"] {
            let name = format!("blocks.0.{layer}");
            let w = ck.get(&format!("{name}.weight"))?;
            let q = ck.quant(&name)?;
            println!(
                "  blocks.0.{layer:14} {:?} {:?}{}",
                w.dtype,
                w.shape,
                q.map_or(String::new(), |q| format!("  {}{}", q.format, if q.convrot { format!(", rotated in groups of {}", q.group) } else { String::new() }))
            );
        }
    }
    Ok(())
}

fn cmd_load(args: &Args) -> Result<()> {
    let ck = Checkpoint::open(args.path(0)?)?;
    let threads = args.number("threads", 8)?;
    let dev = Device::open()?;
    println!("device : {}", dev.name());
    let l = load::load(&dev, &ck, threads, |name| !name.ends_with(".comfy_quant"))?;
    println!(
        "loaded : {} tensors, {:.2} GiB in {:.1} s ({:.2} GiB/s, {threads} readers); {:.2} of {:.1} GiB on the device",
        l.tensors.len(),
        gib(l.bytes),
        l.seconds,
        gib(l.bytes) / l.seconds,
        gib(dev.mem_used()),
        gib(dev.mem_cap())
    );
    Ok(())
}

/// One int8 layer of the checkpoint, on the device.
fn layer(dev: &std::sync::Arc<Device>, ck: &Checkpoint, name: &str) -> Result<(Int8Linear, Vec<i8>, Vec<f32>)> {
    let we = ck.get(&format!("{name}.weight"))?.clone();
    if we.dtype != DType::I8 {
        return Err(Error(format!("{name}.weight is {:?}, not int8: this is not an int8 checkpoint", we.dtype)));
    }
    let wb = ck.read(&format!("{name}.weight"))?;
    let se = ck.get(&format!("{name}.weight_scale"))?.clone();
    let sb = ck.read(&format!("{name}.weight_scale"))?;
    let scale = h3_core::dtype::bytes_to_f32(&sb, se.dtype)?;
    let q = ck.quant(name)?.ok_or_else(|| Error(format!("{name} has no quantization record")))?;
    let lin = Int8Linear {
        weight: Tensor::from_bytes(dev, DType::I8, &we.shape, &wb)?,
        scale: Tensor::from_bytes(dev, DType::F32, &[scale.len()], &sb)?,
        bias: None,
        group: q.convrot.then_some(q.group),
    };
    let w: Vec<i8> = wb.iter().map(|b| *b as i8).collect();
    Ok((lin, w, scale))
}

/// Random activations as bfloat16: the bytes for the device and the values they stand for.
fn activations(rng: &mut Rng, n: usize) -> (Vec<u8>, Vec<f32>) {
    let mut bytes = Vec::with_capacity(n * 2);
    let mut vals = Vec::with_capacity(n);
    for _ in 0..n {
        let b = f32_to_bf16(rng.normal() * 0.7);
        bytes.extend_from_slice(&b.to_le_bytes());
        vals.push(h3_core::dtype::bf16_to_f32(b));
    }
    (bytes, vals)
}

fn cmd_check_linear(args: &Args) -> Result<()> {
    let ck = Checkpoint::open(args.path(0)?)?;
    let block = args.number("block", 0)?;
    let rows = args.number("rows", 64)?;
    let bench_rows = args.number("bench-rows", 16384)?;
    let dev = Device::open()?;
    println!("device : {}", dev.name());
    let mut rng = Rng::new(0);
    let mut ok = true;
    let mut block_ms = 0.0;
    for l in ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"] {
        let name = format!("blocks.{block}.{l}");
        let (lin, w, scale) = layer(&dev, &ck, &name)?;
        let (k, n) = (lin.inputs(), lin.outputs());
        // parity: a few rows, against the CPU reference
        let (xb, xv) = activations(&mut rng, rows * k);
        let x = Tensor::from_bytes(&dev, DType::BF16, &[rows, k], &xb)?;
        let out = Tensor::new(&dev, DType::BF16, &[rows, n])?;
        lin.forward(&x, &out)?;
        let got = out.to_f32()?;
        let host = reference::Layer { w: &w, n, k, wscale: &scale, bias: None, group: lin.group };
        let want = reference::int8_linear(&xv, rows, &host);
        let (rel, cos) = reference::compare(&got, &want);
        // bf16 output keeps 8 bits: its own rounding is ~2e-3 relative
        let good = rel < 5e-3 && cos > 0.9999;
        ok &= good;
        // speed: the denoiser's row count
        let (xb, _) = activations(&mut rng, bench_rows * k);
        let x = Tensor::from_bytes(&dev, DType::BF16, &[bench_rows, k], &xb)?;
        let out = Tensor::new(&dev, DType::BF16, &[bench_rows, n])?;
        lin.forward(&x, &out)?;
        dev.wait()?;
        let t0 = Instant::now();
        const REPS: usize = 5;
        for _ in 0..REPS {
            lin.forward(&x, &out)?;
        }
        dev.wait()?;
        let ms = t0.elapsed().as_secs_f64() * 1e3 / REPS as f64;
        block_ms += ms;
        println!(
            "{name:24} [{n:5} x {k:5}]{}  vs CPU reference: rel err {rel:.2e} cosine {cos:.6}  {}   {bench_rows} rows: {ms:6.1} ms",
            if lin.group.is_some() { " rotated" } else { "        " },
            if good { "ok" } else { "MISMATCH" }
        );
    }
    println!("the block's four linears at {bench_rows} rows: {block_ms:.1} ms");
    if ok {
        Ok(())
    } else {
        Err(Error("the GPU result does not match the CPU reference".into()))
    }
}

/// One line of a comparison: how far `got` is from the reference's `want`.
fn report(what: &str, got: &[f32], want: &[f32]) -> (f64, f64) {
    let (rel, cos) = reference::compare(got, want);
    println!("  {what:34} rel err {rel:.2e}  cosine {cos:.6}");
    (rel, cos)
}

/// Columns [from, from + width) of every `stride`-wide row.
fn columns(v: &[f32], stride: usize, from: usize, width: usize) -> Vec<f32> {
    v.chunks_exact(stride).flat_map(|r| r[from..from + width].iter().copied()).collect()
}

/// The denoiser's blocks against a dump of the reference pipeline (reference/h3x.py, H3X_DUMP_BLOCK): the host-side
/// tables, every stage of block 0, then the stream after later blocks.
fn cmd_check_block(args: &Args) -> Result<()> {
    let ck = Checkpoint::open(args.path(0)?)?;
    let dump = Checkpoint::open(args.path(1)?)?;
    let dev = Device::open()?;
    println!("device : {}", dev.name());
    let count = args.options.get("blocks").map(|_| args.number("blocks", 1)).transpose()?;
    let mut model = dit::Blocks::load(&dev, &ck, count, args.number("threads", 8)?)?;
    println!("blocks : {} loaded, {:.2} GiB in {:.1} s", model.blocks.len(), gib(model.load_bytes), model.load_seconds);
    for (key, eps) in [("norm_eps", &mut model.cfg.norm_eps), ("qk_eps", &mut model.cfg.qk_eps)] {
        if let Some(v) = dump.metadata.get(key) {
            *eps = v.parse().map_err(|_| Error(format!("the dump's {key} is not a number: {v}")))?;
        }
    }
    let cfg = model.cfg;
    let f32s = |name: &str| -> Result<Vec<f32>> { h3_core::dtype::bytes_to_f32(&dump.read(name)?, dump.get(name)?.dtype) };

    // what the step fixes: table rows, positions, the timestep embedding
    let rows: Vec<i32> = dump.read("mod_rows")?.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let positions: Vec<f64> = dump.read("position_ids")?.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    let inv_freq = h3_core::dtype::bytes_to_f32(&ck.read("rope.inv_freq")?, ck.get("rope.inv_freq")?.dtype)?;
    let t_emb = f32s("t_emb")?;
    let tokens = rows.len();
    println!("dump   : {tokens} tokens, {} timestep rows", t_emb.len() / cfg.t_dim);

    println!("host side, against the reference:");
    let tables = model.blocks[0].tables(&cfg, &t_emb);
    let mut worst = 0f64;
    for (t, name) in tables.iter().zip(["shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp"]) {
        worst = worst.max(reference::compare(t, &f32s(&format!("b0.{name}"))?).0);
    }
    println!("  {:34} rel err {worst:.2e}  (worst of six)", "block 0's tables");
    // the dump's rotation matrices are [S, pairs, 2, 2] = (cos, -sin, sin, cos)
    let cs = dit::rotations(&positions, &inv_freq);
    let want_cs: Vec<f32> = f32s("rope_table")?.chunks_exact(4).flat_map(|m| [m[0], m[2]]).collect();
    report("position rotations (cos, sin)", &cs, &want_cs);

    let step = dit::Step::new(&dev, &model, &rows, &positions, &inv_freq, &t_emb)?;
    let scratch = dit::Scratch::new(&dev, &cfg, tokens, DType::BF16)?;
    let x_in = dump.read("x_in")?;
    let x = Tensor::from_bytes(&dev, DType::BF16, &[tokens, cfg.hidden], &x_in)?;

    println!("block 0, stage by stage (each stage runs on this engine's own previous stage):");
    let w = cfg.heads * cfg.head_dim;
    let mut worst_cos = 1f64;
    let mut tap = |name: &str, t: &Tensor| -> Result<()> {
        let got = t.to_f32()?;
        let c = match name {
            "qkv_rotated" => {
                let (_, cq) = report("q after norm + rotation", &columns(&got, 3 * w, 0, w), &f32s("b0.q_rope")?);
                let (_, ck) = report("k after norm + rotation", &columns(&got, 3 * w, w, w), &f32s("b0.k_rope")?);
                cq.min(ck)
            }
            _ => report(name, &got, &f32s(&format!("b0.{name}"))?).1,
        };
        worst_cos = worst_cos.min(c);
        Ok(())
    };
    model.block(0, &x, &step, &scratch, Some(&mut tap))?;

    println!("the stream after later blocks:");
    let last = model.blocks.len() - 1;
    for i in 1..=last {
        model.block(i, &x, &step, &scratch, None)?;
        let key = if dump.metadata.get("last_block").is_some_and(|l| *l == i.to_string()) { "out.last".to_string() } else { format!("out.{i}") };
        if dump.entries.contains_key(&key) {
            let c = report(&format!("after block {i}"), &x.to_f32()?, &f32s(&key)?).1;
            worst_cos = worst_cos.min(c);
        }
    }

    // speed: the whole stack again, one wait at the end
    x.buf.write(0, &x_in)?;
    dev.wait()?;
    let t0 = Instant::now();
    for i in 0..=last {
        model.block(i, &x, &step, &scratch, None)?;
    }
    dev.wait()?;
    let s = t0.elapsed().as_secs_f64();
    println!("speed  : {} blocks on {tokens} tokens in {:.2} s ({:.1} ms per block)", last + 1, s, s * 1e3 / (last + 1) as f64);
    if worst_cos > 0.99 {
        Ok(())
    } else {
        Err(Error(format!("the engine's result drifts from the reference (worst cosine {worst_cos:.4})")))
    }
}

/// The block stack on made-up tokens at a chosen sequence length: what one denoiser step costs, and where.
fn cmd_bench_blocks(args: &Args) -> Result<()> {
    let ck = Checkpoint::open(args.path(0)?)?;
    let tokens = args.number("tokens", 16500)?;
    let dev = Device::open()?;
    println!("device : {}", dev.name());
    let count = args.options.get("blocks").map(|_| args.number("blocks", 1)).transpose()?;
    let model = dit::Blocks::load(&dev, &ck, count, args.number("threads", 8)?)?;
    let cfg = model.cfg;
    println!("blocks : {} loaded, {:.2} GiB in {:.1} s", model.blocks.len(), gib(model.load_bytes), model.load_seconds);

    let mut rng = Rng::new(0);
    let rows: Vec<i32> = (0..tokens).map(|i| if i < tokens / 20 { 1 } else { 0 }).collect();
    let positions: Vec<f64> = (0..tokens * 3).map(|_| (rng.uniform() * 64.0) as f64).collect();
    let inv_freq = h3_core::dtype::bytes_to_f32(&ck.read("rope.inv_freq")?, ck.get("rope.inv_freq")?.dtype)?;
    let t_emb: Vec<f32> = (0..cfg.t_dim).map(|_| rng.normal()).collect();
    let step = dit::Step::new(&dev, &model, &rows, &positions, &inv_freq, &t_emb)?;
    let scratch = dit::Scratch::new(&dev, &cfg, tokens, DType::BF16)?;
    let (xb, _) = activations(&mut rng, tokens * cfg.hidden);
    let x = Tensor::from_bytes(&dev, DType::BF16, &[tokens, cfg.hidden], &xb)?;
    let n = model.blocks.len();

    // warm-up: the first call of each kernel shape compiles it
    model.block(0, &x, &step, &scratch, None)?;
    dev.wait()?;
    x.buf.write(0, &xb)?;
    let t0 = Instant::now();
    for i in 0..n {
        model.block(i, &x, &step, &scratch, None)?;
    }
    dev.wait()?;
    let total = t0.elapsed().as_secs_f64();
    println!("speed  : {n} blocks on {tokens} tokens in {total:.2} s ({:.1} ms per block), {:.1} of {:.1} GiB in use", total * 1e3 / n as f64, gib(dev.mem_used()), gib(dev.mem_cap()));

    // where it goes: the same again, waiting for the device after every stage
    x.buf.write(0, &xb)?;
    let mut stages: Vec<(String, f64)> = Vec::new();
    let mut mark = Instant::now();
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
        model.block(i, &x, &step, &scratch, Some(&mut tap))?;
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
    println!("stages (a wait after each, so the sum is a little above the run without):");
    for (l, s) in merged {
        println!("  {l:36} {:7.1} ms per block  {:4.1}%", s * 1e3 / n as f64, s / sum * 100.0);
    }
    Ok(())
}

fn run() -> Result<()> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = raw.first() else {
        return Err(Error(USAGE.into()));
    };
    let args = Args::parse(&raw[1..])?;
    match cmd.as_str() {
        "version" | "--version" | "-V" => {
            println!("h3 {}", version());
            Ok(())
        }
        "device" => cmd_device(),
        "info" => cmd_info(&args),
        "load" => cmd_load(&args),
        "check-linear" => cmd_check_linear(&args),
        "check-block" => cmd_check_block(&args),
        "bench-blocks" => cmd_bench_blocks(&args),
        _ => Err(Error(USAGE.into())),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn version_matches_the_version_file() {
        assert_eq!(super::version(), include_str!("../../../VERSION").trim());
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("h3: {e}");
            ExitCode::FAILURE
        }
    }
}
