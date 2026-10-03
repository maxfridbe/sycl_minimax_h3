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
//!
//!     h3 serve --model <checkpoint> ...       the engine as a resident daemon (daemon.rs)
//!     h3 status | jobs ps|add|stop|rem|details | unload | shutdown
//!                                             its client (client.rs)

mod client;
mod daemon;
mod http;
mod jobs;
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

const USAGE: &str = "usage:
  h3 version
  h3 device
  h3 info <checkpoint.safetensors>
  h3 load <checkpoint.safetensors> [--threads 8]
  h3 check-linear <checkpoint.safetensors> [--block 0] [--rows 64] [--bench-rows 16384]
  h3 check-block <checkpoint.safetensors> <dump.safetensors> [--blocks N] [--threads 8]
  h3 bench-blocks <checkpoint.safetensors> [--tokens 16500] [--blocks N]

the engine as a resident daemon (the model stays loaded between jobs):
  h3 gpus [--json]                                    the GPUs, numbered as --gpu takes them
  h3 serve --model <checkpoint.safetensors> [--all | --gpu N ...] [--bind 127.0.0.1] [--port 8095] [--idle 600]
           (or --listen ADDR:PORT in place of --bind/--port)
           [--gpu-lock <file>] [--llm-switcher <url>] [--shared-gpu N ...] [--threads 8]
           [--ui <dist/wfe>] [--legacy-api <url>]      the web front end on the same port
  h3 status [--no-stream]                             the engine, live (like docker stats)
  h3 jobs ps [-a]                                     queued and running jobs (-a: all)
  h3 jobs add <kind> [--name value ...] [-f]          queue a job (-f: follow its log); kinds:
                                                      bench-blocks (--tokens N --blocks N), check-block (--dump <file>)
  h3 jobs stop <id>... | rem <id>... | details <id>
  h3 unload | shutdown
  (the client finds the daemon at $H3_DAEMON, default 127.0.0.1:8095)";

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

/// `h3 gpus [--json]`: the GPUs the runtime sees, numbered as `--gpu` takes them.
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
            vals.push(v.parse().map_err(|_| Error(format!("{name} {v}: not a GPU number (see `h3 gpus`)")))?);
        } else {
            rest.push(a.clone());
        }
    }
    Ok((vals, rest))
}

fn cmd_serve(raw: &[String]) -> Result<()> {
    let (gpus, raw) = take_repeated(raw, "--gpu")?;
    let (shared, raw) = take_repeated(&raw, "--shared-gpu")?;
    let all = raw.iter().any(|a| a == "--all");
    let raw: Vec<String> = raw.into_iter().filter(|a| a != "--all").collect();
    if all && !gpus.is_empty() {
        return Err(Error("give --all or --gpu N ..., not both".into()));
    }
    let args = &Args::parse(&raw)?;
    let model = args.options.get("model").ok_or("serve needs --model <checkpoint.safetensors>")?;
    let idle = args.number("idle", 600)?;
    daemon::serve(daemon::Options {
        listen: listen_addr(args)?,
        model: model.into(),
        idle: (idle > 0).then(|| std::time::Duration::from_secs(idle as u64)),
        gpu_lock: args.options.get("gpu-lock").map(Into::into),
        llm_switcher: args.options.get("llm-switcher").cloned(),
        threads: args.number("threads", 8)?,
        gpus: (!gpus.is_empty()).then_some(gpus),
        shared_gpus: (!shared.is_empty()).then_some(shared),
        ui: args.options.get("ui").map(Into::into),
        legacy_api: args.options.get("legacy-api").cloned(),
    })
}

/// Where `h3 serve` listens: `--listen ADDR:PORT`, or `--bind ADDR` and `--port N` (each optional: 127.0.0.1, 8095).
fn listen_addr(args: &Args) -> Result<String> {
    if let Some(l) = args.options.get("listen") {
        if args.options.contains_key("bind") || args.options.contains_key("port") {
            return Err(Error("give --listen ADDR:PORT, or --bind / --port, not both".into()));
        }
        return Ok(l.clone());
    }
    let bind = args.options.get("bind").map_or("127.0.0.1", |b| b.as_str());
    let port = args.number("port", 8095)?;
    if port == 0 || port > 65535 {
        return Err(Error(format!("--port {port}: not a port")));
    }
    // an IPv6 address needs brackets before the port
    Ok(if bind.contains(':') && !bind.starts_with('[') { format!("[{bind}]:{port}") } else { format!("{bind}:{port}") })
}

fn run() -> Result<()> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = raw.first() else {
        return Err(Error(USAGE.into()));
    };
    // the client commands parse their own arguments (flags like -a, -f, --no-stream)
    let args = if matches!(cmd.as_str(), "status" | "jobs" | "serve" | "gpus") { Args::parse(&[])? } else { Args::parse(&raw[1..])? };
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
        "serve" => cmd_serve(&raw[1..]),
        "gpus" => cmd_gpus(&Args { positional: raw[1..].to_vec(), options: BTreeMap::new() }),
        "worker" => worker::run(args.number("gpu", 0)?, args.options.get("model").ok_or("worker needs --model")?.into(), args.number("threads", 8)?),
        "status" => client::status(&raw[1..]),
        "jobs" => client::jobs(&raw[1..]),
        "unload" | "shutdown" => client::simple(cmd),
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
