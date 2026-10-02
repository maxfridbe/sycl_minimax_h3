//! `h3` - MiniMax H3 on an Intel Arc GPU.
//!
//!     h3 device                               the GPU the kernels found, and its memory cap
//!     h3 info <checkpoint.safetensors>        what is in a checkpoint
//!     h3 load <checkpoint> [--threads 8]      load it onto the GPU, timed
//!     h3 check-linear <checkpoint> [--block 0] [--rows 64] [--bench-rows 16384]
//!                                             a block's int8 linears: the GPU against the CPU reference, and timed

use std::collections::BTreeMap;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use h3_core::device::{Device, Tensor};
use h3_core::dtype::{f32_to_bf16, DType};
use h3_core::ops::Int8Linear;
use h3_core::rng::Rng;
use h3_core::safetensors::Checkpoint;
use h3_core::{load, reference, Error, Result};

const USAGE: &str = "usage:
  h3 device
  h3 info <checkpoint.safetensors>
  h3 load <checkpoint.safetensors> [--threads 8]
  h3 check-linear <checkpoint.safetensors> [--block 0] [--rows 64] [--bench-rows 16384]";

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

fn run() -> Result<()> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = raw.first() else {
        return Err(Error(USAGE.into()));
    };
    let args = Args::parse(&raw[1..])?;
    match cmd.as_str() {
        "device" => cmd_device(),
        "info" => cmd_info(&args),
        "load" => cmd_load(&args),
        "check-linear" => cmd_check_linear(&args),
        _ => Err(Error(USAGE.into())),
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
