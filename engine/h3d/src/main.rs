//! `h3d` - the MiniMax H3 engine, inside its container. Not the command a person types: `sycl-h3` (on the host)
//! starts `h3d daemon` in a container and talks to it over a Unix socket; the daemon starts `h3d worker` per GPU.
//!
//!     h3d daemon --socket <path> --model <checkpoint> [--all | --gpu N ...] [--idle 600]
//!                [--gpu-lock <file>] [--llm-switcher <url>] [--shared-gpu N ...] [--threads 8]
//!                                              the engine service (daemon.rs)
//!     h3d worker --gpu N --model <checkpoint>  the process that holds one GPU (worker.rs); the daemon starts it
//!     h3d gpus [--json]                        the GPUs, numbered as --gpu takes them
//!
//! and, for development, one-shot checks that load, run and exit (run.sh runs them in the container):
//!
//!     h3d version | device | info <checkpoint> | load <checkpoint> [--threads 8]
//!     h3d check-linear <checkpoint> [--block 0] [--rows 64] [--bench-rows 16384]
//!     h3d check-block <checkpoint> <dump> [--blocks N]
//!     h3d bench-blocks <checkpoint> [--tokens 16500] [--blocks N]
//!     h3d denoise <checkpoint> <run dump> [--out latents.safetensors]
//!     h3d generate <checkpoint> --prompt-file <file> --out clip.mp4 [--width 384 --height 288 --seconds 2 --steps 8
//!                  --seed 0 --upscale 1] [--te .. --vae .. --audio-vae .. --upscaler .. --tokenizer ..]
//!     h3d check-encoders <video vae> <audio vae> <encoder dump>
//!     h3d encode <te.gguf> --prompt-file <file> [--tokenizer dir] [--out cond.safetensors] [--check run dump]
//!     h3d decode <vae checkpoint> <latents.safetensors> [--audio-vae <checkpoint>] [--upscaler <checkpoint> | --pixel-upscaler <esrgan.safetensors>] [--upscale 2]
//!                [--out clip.mp4] [--check decode dump]

mod daemon;
mod jobs;
mod media;
mod signals;
mod worker;

use std::collections::BTreeMap;
use std::path::Path;
use std::process::ExitCode;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use h3_core::device::{Device, Tensor};
use h3_core::dtype::DType;
use h3_core::ops::Int8Linear;
use h3_core::rng::Rng;
use h3_core::safetensors::Checkpoint;
use h3_core::{load, reference, Error, Result};

/// The repository's version, yy.mmdd.### (Cargo carries it without the leading zeros).
pub(crate) fn version() -> String {
    let p: Vec<u32> = env!("CARGO_PKG_VERSION").split('.').map(|x| x.parse().unwrap_or(0)).collect();
    format!("{:02}.{:04}.{:03}", p[0], p[1], p[2])
}

const USAGE: &str = "usage (the daemon side; people use sycl-h3 on the host):
  h3d daemon --socket <path> --model <checkpoint.safetensors> [--all | --gpu N ...] [--idle 600]
             [--gpu-lock <file>] [--llm-switcher <url>] [--shared-gpu N ...] [--threads 8]
  h3d worker --gpu N --model <checkpoint.safetensors>
  h3d gpus [--json]
  h3d version | device | info <checkpoint> | load <checkpoint> [--threads 8]
  h3d check-linear <checkpoint> [--block 0] [--rows 64] [--bench-rows 16384]
  h3d check-block <checkpoint> <dump.safetensors> [--blocks N] [--threads 8]
  h3d bench-blocks <checkpoint> [--tokens 16500] [--blocks N]
  h3d denoise <checkpoint> <rundump.safetensors> [--out latents.safetensors] [--threads 8]
  h3d generate <checkpoint> --prompt-file <file> --out clip.mp4 [--width 384] [--height 288] [--seconds 2] [--steps 8]
               [--seed 0] [--upscale 1] [--te <gguf>] [--vae <ckpt>] [--audio-vae <ckpt>] [--upscaler <ckpt>] [--tokenizer <dir>]
  h3d check-encoders <video vae> <audio vae> <encdump.safetensors>
  h3d encode <te.gguf> --prompt-file <file> [--tokenizer /app/tokenizer] [--out cond.safetensors] [--check rundump.safetensors]
  h3d decode <vae checkpoint> <latents.safetensors> [--audio-vae <checkpoint>] [--upscaler <checkpoint> | --pixel-upscaler <esrgan.safetensors>] [--upscale 2]
             [--out clip.mp4] [--check decodedump.safetensors]";

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
        let (xb, xv) = jobs::activations(&mut rng, rows * k);
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
        let (xb, _) = jobs::activations(&mut rng, bench_rows * k);
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

fn println_log() -> impl FnMut(String) {
    |l: String| println!("{l}")
}

fn cmd_check_block(args: &Args) -> Result<()> {
    let count = args.options.get("blocks").map(|_| args.number("blocks", 1)).transpose()?;
    let mut log = println_log();
    let e = jobs::Engine::load(args.path(0)?, count, args.number("threads", 8)?, &mut log)?;
    let cancel = AtomicBool::new(false);
    jobs::check_block(&e, args.path(1)?, &mut jobs::Ctl { log: &mut log, cancel: &cancel, progress: None }).map(|_| ())
}

fn cmd_bench_blocks(args: &Args) -> Result<()> {
    let count = args.options.get("blocks").map(|_| args.number("blocks", 1)).transpose()?;
    let mut log = println_log();
    let e = jobs::Engine::load(args.path(0)?, count, args.number("threads", 8)?, &mut log)?;
    let cancel = AtomicBool::new(false);
    jobs::bench_blocks(&e, args.number("tokens", 16500)?, None, &mut jobs::Ctl { log: &mut log, cancel: &cancel, progress: None }).map(|_| ())
}

fn cmd_denoise(args: &Args) -> Result<()> {
    let mut log = println_log();
    let e = jobs::Engine::load(args.path(0)?, None, args.number("threads", 8)?, &mut log)?;
    let cancel = AtomicBool::new(false);
    let out = args.options.get("out").map(Path::new);
    jobs::denoise(&e, args.path(1)?, out, &mut jobs::Ctl { log: &mut log, cancel: &cancel, progress: None }).map(|_| ())
}

fn cmd_generate(args: &Args) -> Result<()> {
    let mut log = println_log();
    let e = jobs::Engine::load(args.path(0)?, None, args.number("threads", 8)?, &mut log)?;
    // the same spec the daemon takes: options become fields (numbers as numbers), the prompt file read by the job
    let mut spec = serde_json::Map::new();
    spec.insert("kind".into(), serde_json::json!("generate"));
    for (k, v) in &args.options {
        let key = k.replace('-', "_");
        spec.insert(key, v.parse::<f64>().map(serde_json::Value::from).unwrap_or_else(|_| serde_json::json!(v)));
    }
    let cancel = AtomicBool::new(false);
    jobs::run(&e, &serde_json::Value::Object(spec), &mut jobs::Ctl { log: &mut log, cancel: &cancel, progress: None }).map(|_| ())
}

fn cmd_check_encoders(args: &Args) -> Result<()> {
    let mut log = println_log();
    let dev = Device::open()?;
    log(format!("device : {}", dev.name()));
    let cancel = AtomicBool::new(false);
    jobs::check_encoders(&dev, args.path(0)?, args.path(1)?, args.path(2)?, &mut jobs::Ctl { log: &mut log, cancel: &cancel, progress: None }).map(|_| ())
}

fn cmd_encode(args: &Args) -> Result<()> {
    let mut log = println_log();
    let dev = Device::open()?;
    log(format!("device : {}", dev.name()));
    let pf = args.options.get("prompt-file").ok_or("encode needs --prompt-file")?;
    let prompt = std::fs::read_to_string(pf).map_err(|e| Error(format!("{pf}: {e}")))?.trim().to_string();
    let cancel = AtomicBool::new(false);
    let opt = |k: &str| args.options.get(k).map(Path::new);
    let tokenizer = opt("tokenizer").unwrap_or(Path::new("/app/tokenizer"));
    jobs::encode(&dev, args.number("threads", 8)?, &prompt, &jobs::TeFiles { te: args.path(0)?, tokenizer }, opt("out"), opt("check"), &mut jobs::Ctl { log: &mut log, cancel: &cancel, progress: None })
        .map(|_| ())
}

fn cmd_decode(args: &Args) -> Result<()> {
    let mut log = println_log();
    let dev = Device::open()?;
    log(format!("device : {}", dev.name()));
    let cancel = AtomicBool::new(false);
    let opt = |k: &str| args.options.get(k).map(Path::new);
    let scale: f32 = args.options.get("upscale").map(|s| s.parse().map_err(|_| Error(format!("--upscale {s}: not a number")))).transpose()?.unwrap_or(2.0);
    let upscale = opt("upscaler").map(|p| (p, scale));
    let pixel = opt("pixel-upscaler").map(|p| (p, scale));
    jobs::decode(&dev, args.number("threads", 8)?, args.path(1)?, &jobs::Vaes { video: args.path(0)?, audio: opt("audio-vae"), upscale, pixel }, opt("out"), opt("check"), &mut jobs::Ctl { log: &mut log, cancel: &cancel, progress: None }).map(|_| ())
}

/// `h3d gpus [--json]`: the GPUs the runtime sees, numbered as `--gpu` takes them.
fn cmd_gpus(args: &Args) -> Result<()> {
    let list = h3_core::device::Device::list()?;
    if args.positional.iter().any(|a| a == "--json") || args.options.contains_key("json") {
        let v: Vec<serde_json::Value> = list
            .iter()
            .map(|g| serde_json::json!({"index": g.index, "name": g.name, "mem_gib": jobs::gib(g.mem_bytes), "pci": g.pci}))
            .collect();
        println!("{}", serde_json::Value::from(v));
    } else {
        println!("{:<4} {:<34} {:>8}  PCI", "GPU", "NAME", "MEMORY");
        for g in &list {
            println!("{:<4} {:<34} {:>5.1}GiB  {}", g.index, g.name, jobs::gib(g.mem_bytes), g.pci);
        }
    }
    Ok(())
}

/// The values of a repeatable option (`--gpu 0 --gpu 1`), and the arguments without them.
fn take_repeated(raw: &[String], name: &str) -> Result<(Vec<usize>, Vec<String>)> {
    let (mut vals, mut rest) = (Vec::new(), Vec::new());
    let mut it = raw.iter();
    while let Some(a) = it.next() {
        if a == name {
            let v = it.next().ok_or_else(|| Error(format!("{name} needs a GPU number")))?;
            vals.push(v.parse().map_err(|_| Error(format!("{name} {v}: not a GPU number (see `sycl-h3 gpus`)")))?);
        } else {
            rest.push(a.clone());
        }
    }
    Ok((vals, rest))
}

fn cmd_daemon(raw: &[String]) -> Result<()> {
    let (gpus, raw) = take_repeated(raw, "--gpu")?;
    let (shared, raw) = take_repeated(&raw, "--shared-gpu")?;
    let all = raw.iter().any(|a| a == "--all");
    let raw: Vec<String> = raw.into_iter().filter(|a| a != "--all").collect();
    if all && !gpus.is_empty() {
        return Err(Error("give --all or --gpu N ..., not both".into()));
    }
    let args = &Args::parse(&raw)?;
    let model = args.options.get("model").ok_or("daemon needs --model <checkpoint.safetensors>")?;
    let socket = args.options.get("socket").ok_or("daemon needs --socket <path>")?;
    let idle = args.number("idle", 600)?;
    daemon::serve(daemon::Options {
        socket: socket.into(),
        model: model.into(),
        idle: (idle > 0).then(|| std::time::Duration::from_secs(idle as u64)),
        gpu_lock: args.options.get("gpu-lock").map(Into::into),
        llm_switcher: args.options.get("llm-switcher").cloned(),
        threads: args.number("threads", 8)?,
        gpus: (!gpus.is_empty()).then_some(gpus),
        shared_gpus: (!shared.is_empty()).then_some(shared),
    })
}

fn run() -> Result<()> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = raw.first() else {
        return Err(Error(USAGE.into()));
    };
    // the client commands parse their own arguments (flags like -a, -f, --no-stream)
    let args = if matches!(cmd.as_str(), "daemon" | "gpus") { Args::parse(&[])? } else { Args::parse(&raw[1..])? };
    match cmd.as_str() {
        "version" | "--version" | "-V" => {
            println!("h3d {}", version());
            Ok(())
        }
        "device" => cmd_device(),
        "info" => cmd_info(&args),
        "load" => cmd_load(&args),
        "check-linear" => cmd_check_linear(&args),
        "check-block" => cmd_check_block(&args),
        "bench-blocks" => cmd_bench_blocks(&args),
        "denoise" => cmd_denoise(&args),
        "decode" => cmd_decode(&args),
        "encode" => cmd_encode(&args),
        "check-encoders" => cmd_check_encoders(&args),
        "generate" => cmd_generate(&args),
        "daemon" => cmd_daemon(&raw[1..]),
        "gpus" => cmd_gpus(&Args { positional: raw[1..].to_vec(), options: BTreeMap::new() }),
        "worker" => worker::run(args.number("gpu", 0)?, args.options.get("model").ok_or("worker needs --model")?.into(), args.number("threads", 8)?),
        _ => Err(Error(USAGE.into())),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("h3d: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn version_matches_the_version_file() {
        assert_eq!(super::version(), include_str!("../../../VERSION").trim());
    }
}
