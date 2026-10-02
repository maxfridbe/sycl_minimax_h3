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
//!     h3 runjob <kind> ... | status | jobs | job <id> | killjob <id> | unload | shutdown
//!                                             its client

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
  h3 serve --model <checkpoint.safetensors> [--listen 127.0.0.1:8095] [--idle 600]
           [--gpu-lock <file>] [--llm-switcher <url>] [--threads 8]
           [--ui <dist/wfe>] [--legacy-api <url>]      the web front end on the same port
  h3 runjob <kind> [--name value ...] [--no-wait]     kinds: bench-blocks (--tokens N --blocks N),
                                                      check-block (--dump <file>)
  h3 status | jobs | job <id> | killjob <id> | unload | shutdown
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
    jobs::check_block(&e, args.path(1)?, &mut jobs::Ctl { log: &mut log, cancel: &cancel }).map(|_| ())
}

fn cmd_bench_blocks(args: &Args) -> Result<()> {
    let count = args.options.get("blocks").map(|_| args.number("blocks", 1)).transpose()?;
    let mut log = println_log();
    let e = jobs::Engine::load(args.path(0)?, count, args.number("threads", 8)?, &mut log)?;
    let cancel = AtomicBool::new(false);
    jobs::bench_blocks(&e, args.number("tokens", 16500)?, None, &mut jobs::Ctl { log: &mut log, cancel: &cancel }).map(|_| ())
}

fn cmd_serve(args: &Args) -> Result<()> {
    let model = args.options.get("model").ok_or("serve needs --model <checkpoint.safetensors>")?;
    let idle = args.number("idle", 600)?;
    daemon::serve(daemon::Options {
        listen: args.options.get("listen").cloned().unwrap_or_else(daemon_addr),
        model: model.into(),
        idle: (idle > 0).then(|| std::time::Duration::from_secs(idle as u64)),
        gpu_lock: args.options.get("gpu-lock").map(Into::into),
        llm_switcher: args.options.get("llm-switcher").cloned(),
        threads: args.number("threads", 8)?,
        ui: args.options.get("ui").map(Into::into),
        legacy_api: args.options.get("legacy-api").cloned(),
    })
}

/// Where the client finds the daemon: `$H3_DAEMON`, or the default port on the loopback interface.
fn daemon_addr() -> String {
    std::env::var("H3_DAEMON").unwrap_or_else(|_| "127.0.0.1:8095".into())
}

fn show(v: &serde_json::Value) {
    println!("{}", serde_json::to_string_pretty(v).unwrap_or_default());
}

/// `h3 runjob <kind> [--name value ...] [--no-wait]`: numbers become numbers, everything else strings.
fn cmd_runjob(raw: &[String]) -> Result<()> {
    let no_wait = raw.iter().any(|a| a == "--no-wait");
    let rest: Vec<String> = raw.iter().filter(|a| *a != "--no-wait").cloned().collect();
    let args = Args::parse(&rest)?;
    let kind = args.positional.first().ok_or("runjob needs a kind: bench-blocks, check-block")?;
    let mut spec = serde_json::Map::new();
    spec.insert("kind".into(), serde_json::json!(kind));
    for (k, v) in &args.options {
        let val = v.parse::<u64>().map(serde_json::Value::from).unwrap_or_else(|_| serde_json::json!(v));
        spec.insert(k.clone(), val);
    }
    let addr = daemon_addr();
    let id = http::call(&addr, "POST", "/engine/jobs", Some(&serde_json::Value::Object(spec)))?["id"].as_u64().ok_or("no job id in the answer")?;
    println!("job {id}");
    if no_wait {
        return Ok(());
    }
    // follow the log until the job ends
    let mut shown = 0;
    let mut engine_shown = String::new();
    loop {
        let j = http::call(&addr, "GET", &format!("/engine/jobs/{id}"), None)?;
        let log = j["log"].as_array().cloned().unwrap_or_default();
        for l in &log[shown.min(log.len())..] {
            println!("{}", l.as_str().unwrap_or(""));
        }
        shown = log.len();
        let state = j["state"].as_str().unwrap_or("");
        if state == "queued" || (state == "running" && shown == 0) {
            let st = http::call(&addr, "GET", "/engine/status", None)?;
            let e = st["engine"].as_str().unwrap_or("").to_string();
            if e != engine_shown && e != "loaded" {
                println!("({state}; engine: {e})");
                engine_shown = e;
            }
        }
        match state {
            "done" => return Ok(()),
            "failed" | "cancelled" => return Err(Error(format!("job {id} {state}: {}", j["error"].as_str().unwrap_or("")))),
            _ => std::thread::sleep(std::time::Duration::from_millis(700)),
        }
    }
}

fn cmd_client(cmd: &str, args: &Args) -> Result<()> {
    let addr = daemon_addr();
    let id = || args.positional.first().ok_or_else(|| Error(format!("{cmd} needs a job id")));
    let v = match cmd {
        "status" => http::call(&addr, "GET", "/engine/status", None)?,
        "jobs" => http::call(&addr, "GET", "/engine/jobs", None)?,
        "job" => http::call(&addr, "GET", &format!("/engine/jobs/{}", id()?), None)?,
        "killjob" => http::call(&addr, "POST", &format!("/engine/jobs/{}/cancel", id()?), None)?,
        "unload" => http::call(&addr, "POST", "/engine/unload", None)?,
        "shutdown" => http::call(&addr, "POST", "/engine/shutdown", None)?,
        _ => unreachable!(),
    };
    show(&v);
    Ok(())
}

fn run() -> Result<()> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = raw.first() else {
        return Err(Error(USAGE.into()));
    };
    let args = if cmd == "runjob" { Args::parse(&[])? } else { Args::parse(&raw[1..])? };
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
        "serve" => cmd_serve(&args),
        "worker" => worker::run(args.options.get("model").ok_or("worker needs --model")?.into(), args.number("threads", 8)?),
        "runjob" => cmd_runjob(&raw[1..]),
        "status" | "jobs" | "job" | "killjob" | "unload" | "shutdown" => cmd_client(cmd, &args),
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
