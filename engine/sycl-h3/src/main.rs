//! `sycl-h3` - MiniMax H3 on Intel Arc GPUs: the command line, on the host. It starts two services - the engine
//! daemon in a container, the studio (the web front end and its clip queue) as a host process - and talks to the
//! engine over a Unix socket; nothing of the engine runs in this process.
//!
//! ```text
//!   sycl-h3 (you) ---- start/stop (podman) ----> [sycl-h3]      h3d daemon -- pipes --> h3d worker per GPU
//!        |                                            ^ Unix socket (JSON over HTTP)
//!        +---------- status/jobs/unload -------------+
//!        +---- serve/stop --web (process) ---> studio            web front end + clip queue, :8095 -> the same socket
//! ```
//!
//! The studio runs on the host, not in a container, because the box's language models it switches are the host's
//! own programs (H3_LLM_MODES).
//!
//! A static binary: it runs on the host whatever the host's C library is.

mod client;
mod config;
mod container;
mod studio;
mod tools;

use std::path::PathBuf;
use std::process::{ExitCode, Stdio};
use std::time::{Duration, Instant};

use h3_http::{Error, Result, Target};

use config::Config;
use container::{mount, Ce, ENGINE, SOCKET_DIR_IN};

pub(crate) fn version() -> String {
    let p: Vec<u32> = env!("CARGO_PKG_VERSION").split('.').map(|x| x.parse().unwrap_or(0)).collect();
    format!("{:02}.{:04}.{:03}", p[0], p[1], p[2])
}

const USAGE: &str = "sycl-h3 - MiniMax H3 on Intel Arc GPUs

services (either runs without the other):
  sycl-h3 start [--all | --gpu N ...] [--shared-gpu N ...]
                                the engine daemon; each GPU (all by default) gets its own engine process when a job
                                needs it, and the model stays loaded on it between jobs
  sycl-h3 serve [--bind ADDR] [--port N]
                                the studio: the web front end and its clip queue (default 127.0.0.1:8095), a host
                                process; it hands the clips to the engine over the same socket
  sycl-h3 stop [--web | --all]  stop the engine (the default), the web front end, or both - gracefully

the engine (over its socket):
  sycl-h3 status [--no-stream]  live, one row per GPU (like docker stats)
  sycl-h3 jobs ps [-a]          queued and running jobs (-a: all)
  sycl-h3 jobs add <kind> [--name value ...] [-f]
                                queue a job (-f: follow its log); --gpu N pins it to one GPU
                                kinds: bench-blocks (--tokens N --blocks N), check-block (--dump /out/<file>),
                                denoise (--dump /out/<run dump> --out /out/<latents>),
                                generate (--prompt <text> --width 384 --height 288 --seconds 2 --steps 8 --seed 0
                                          --upscale 2 --out /out/<clip>.mp4): a whole clip
                                encode (--prompt <text> | --prompt_file f --te /models/<te>.gguf --out /out/<cond>),
                                decode (--latents /out/<latents> --vae /models/<vae> --audio_vae /models/<vae> --out /out/<clip>.mp4)
  sycl-h3 jobs stop <id>... | rem <id>... | details <id>
  sycl-h3 unload [--gpu N]      give a GPU back now; the next job loads again

  sycl-h3 gpus                  the GPUs, numbered as --gpu takes them

tools (talk to the studio, like the front end):
  sycl-h3 speech <text file> [options]   a speech as sized, chained character clips (sycl-h3 speech --help)
  sycl-h3 scene <scene.json> [options]   queue a scene file (from the front end's export)
  sycl-h3 join <prefix> [options]        join a series of finished clips into one film, frame-exact
  sycl-h3 speechpct <clip.mp4>...        how much of each clip is speech
  sycl-h3 logs [--web]          a service's log, followed
  sycl-h3 version

settings (environment, or NAME=value lines in sycl-h3.conf beside the repository or ~/.config/sycl-h3.conf):
  H3_MODELS        host directory with the checkpoints, seen as /models          (required for start)
  H3_MODEL         the checkpoint as seen in the container
                   (default /models/kitchen/minimax_h3_fl2va_pruned_int8_convrot.safetensors)
  H3_OUT           host directory for job files, seen as /out                    (default ./out)
  H3_GPUS          GPUs to serve, e.g. \"0 1\" (default all)   H3_SHARED_GPUS  GPUs the two hooks below are about
  H3_IDLE          seconds without a job before an engine unloads; 0 = never     (default 600)
  H3_GPU_LOCK      a lock file shared with the GPU's other users
  H3_LLM_SWITCHER  a front end's model switcher URL: its model stops before loading, comes back after
  H3S_ATTN         onednn: attention by oneDNN's kernel instead of SageAttention (int8 q, k; dist/libh3sage.so)
  H3S_SAGE_MIN_S   the shortest sequence SageAttention takes (default 8192; shorter ones run on oneDNN)
  H3_LISTEN, H3_PORT   where serve listens (default 127.0.0.1, 8095; 0.0.0.0 = the network, no password)
  H3_STUDIO_DIR    the studio's queue and state files (default ~/.local/share/sycl-h3)
  H3_LLM_MODES     a JSON file of the language models the studio switches (docs/LEGACY-API.md); point the
                   engine's H3_LLM_SWITCHER at the studio (http://127.0.0.1:8095/rpc/llm.mode) to use them
  H3_GPUSTAT       GPU telemetry JSON for the front end (default /run/gpustat.json)
  H3_TEMPLATES     a directory of prompt templates (*.txt)
  H3_STUDIO        the studio's URL for the tools (default http://127.0.0.1:8095)
  H3_SOCKET_DIR    where the engine's socket lives (default $XDG_RUNTIME_DIR/sycl-h3)
  H3_IMAGE, H3_CONTAINER_ENGINE   the image (h3-build) and podman / docker";

/// `--gpu 0 --gpu 1` -> [0, 1], and the arguments without them.
fn take_repeated(raw: &[String], name: &str) -> Result<(Vec<String>, Vec<String>)> {
    let (mut vals, mut rest) = (Vec::new(), Vec::new());
    let mut it = raw.iter();
    while let Some(a) = it.next() {
        if a == name {
            let v = it.next().ok_or_else(|| Error(format!("{name} needs a value")))?;
            vals.push(v.clone());
        } else {
            rest.push(a.clone());
        }
    }
    Ok((vals, rest))
}

fn take_value(raw: &[String], name: &str) -> Result<(Option<String>, Vec<String>)> {
    let (mut v, rest) = take_repeated(raw, name)?;
    if v.len() > 1 {
        return Err(Error(format!("{name} given twice")));
    }
    Ok((v.pop(), rest))
}

fn no_more(rest: &[String], cmd: &str) -> Result<()> {
    match rest.first() {
        None => Ok(()),
        Some(a) => Err(Error(format!("sycl-h3 {cmd}: unexpected {a:?} (sycl-h3 help)"))),
    }
}

fn wait_until(what: &str, secs: u64, mut ok: impl FnMut() -> bool) -> Result<()> {
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_secs(secs) {
        if ok() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    Err(Error(format!("{what} did not come up in {secs} s (sycl-h3 logs)")))
}

fn need_dist(cfg: &Config, file: &str) -> Result<PathBuf> {
    let p = cfg.dist.join(file);
    if p.exists() {
        Ok(p)
    } else {
        Err(Error(format!("{}: not built yet (./build.sh)", p.display())))
    }
}

fn socket_ready(cfg: &Config) -> bool {
    std::os::unix::net::UnixStream::connect(cfg.socket()).is_ok()
}

/// `sycl-h3 start`: the engine daemon's container.
fn start(cfg: &Config, raw: &[String]) -> Result<()> {
    let (gpus, rest) = take_repeated(raw, "--gpu")?;
    let (shared, rest) = take_repeated(&rest, "--shared-gpu")?;
    let all = rest.iter().any(|a| a == "--all");
    let rest: Vec<String> = rest.into_iter().filter(|a| a != "--all").collect();
    no_more(&rest, "start")?;
    if all && !gpus.is_empty() {
        return Err(Error("give --all or --gpu N ..., not both".into()));
    }
    let gpus: Vec<String> = if all || !gpus.is_empty() { gpus } else { cfg.or("H3_GPUS", "").split_whitespace().map(String::from).collect() };
    let shared: Vec<String> = if shared.is_empty() { cfg.or("H3_SHARED_GPUS", "").split_whitespace().map(String::from).collect() } else { shared };
    for g in gpus.iter().chain(&shared) {
        g.parse::<usize>().map_err(|_| Error(format!("{g}: not a GPU number (sycl-h3 gpus)")))?;
    }

    let ce = Ce::new(cfg)?;
    if ce.running(ENGINE) {
        println!("the engine is already running");
        return client::status(&["--no-stream".into()], &|| String::new());
    }
    ce.need_image()?;
    need_dist(cfg, "h3d")?;
    need_dist(cfg, "libh3sycl.so")?;
    let models = cfg.get("H3_MODELS").ok_or("set H3_MODELS (the host directory with the checkpoints)")?;
    ce.remove(ENGINE); // a stopped one from before
    let out = PathBuf::from(cfg.or("H3_OUT", &cfg.dist.join("../out").to_string_lossy()));
    std::fs::create_dir_all(&out)?;
    let sock_dir = cfg.socket_dir();
    std::fs::create_dir_all(&sock_dir)?;
    let _ = std::fs::remove_file(cfg.socket());

    let mut args: Vec<String> = vec!["run".into(), "-d".into(), "--name".into(), ENGINE.into(), "--network".into(), "host".into()];
    // a stop waits for the running job's next block boundary (about a second at production size), then unloads
    args.extend(["--stop-timeout".into(), "150".into()]);
    args.extend(ce.user_args());
    args.extend(ce.gpu_args());
    args.extend(mount(&cfg.dist, "/app", true));
    args.extend(mount(&PathBuf::from(&models), "/models", true));
    args.extend(mount(&out, "/out", false));
    args.extend(mount(&sock_dir, SOCKET_DIR_IN, false));
    let mut daemon_args: Vec<String> = vec![
        "--socket".into(),
        format!("{SOCKET_DIR_IN}/h3d.sock"),
        "--model".into(),
        cfg.or("H3_MODEL", "/models/kitchen/minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
        "--idle".into(),
        cfg.or("H3_IDLE", "600"),
    ];
    for g in &gpus {
        daemon_args.extend(["--gpu".into(), g.clone()]);
    }
    for g in &shared {
        daemon_args.extend(["--shared-gpu".into(), g.clone()]);
    }
    if let Some(lock) = cfg.get("H3_GPU_LOCK") {
        let dir = PathBuf::from(&lock).parent().map(|d| d.to_path_buf()).unwrap_or_default();
        args.extend(mount(&dir, &dir.to_string_lossy(), false));
        daemon_args.extend(["--gpu-lock".into(), lock]);
    }
    if let Some(sw) = cfg.get("H3_LLM_SWITCHER") {
        daemon_args.extend(["--llm-switcher".into(), sw]);
    }
    args.extend(["-e".into(), "H3SYCL_LIB=/app/libh3sycl.so".into(), "-e".into(), "ONEAPI_DEVICE_SELECTOR=level_zero:*".into()]);
    for k in ["H3S_MEM_FRACTION", "H3S_ATTN", "H3S_SAGE_MIN_S"] {
        if let Some(v) = cfg.get(k) {
            args.extend(["-e".into(), format!("{k}={v}")]);
        }
    }
    args.extend([ce.image.clone(), "bash".into(), "-c".into(),
                 "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec /app/h3d daemon \"$@\"".into(), "h3d".into()]);
    args.extend(daemon_args);
    let st = ce.cmd().args(&args).stdout(Stdio::null()).status()?;
    if !st.success() {
        return Err(Error(format!("{} run failed", ce.bin)));
    }
    wait_until("the engine", 60, || socket_ready(cfg))?;
    client::status(&["--no-stream".into()], &|| String::new())
}

/// The studio's process files: its pid and where it listens, beside the engine's socket.
fn studio_files(cfg: &Config) -> (PathBuf, PathBuf, PathBuf) {
    let d = cfg.socket_dir();
    (d.join("studio.pid"), d.join("studio.listen"), d.join("studio.log"))
}

fn studio_pid(cfg: &Config) -> Option<u32> {
    let (pidf, _, _) = studio_files(cfg);
    let pid: u32 = std::fs::read_to_string(pidf).ok()?.trim().parse().ok()?;
    let cmd = std::fs::read(format!("/proc/{pid}/cmdline")).ok()?;
    String::from_utf8_lossy(&cmd).contains("studio").then_some(pid)
}

/// `sycl-h3 serve`: the studio, as a host process of its own (setsid; it outlives this command).
fn serve(cfg: &Config, raw: &[String]) -> Result<()> {
    let (bind, rest) = take_value(raw, "--bind")?;
    let (port, rest) = take_value(&rest, "--port")?;
    no_more(&rest, "serve")?;
    let bind = bind.unwrap_or_else(|| cfg.or("H3_LISTEN", "127.0.0.1"));
    let port: u16 = port.unwrap_or_else(|| cfg.or("H3_PORT", "8095")).parse().map_err(|_| Error("--port: not a port".into()))?;
    if studio_pid(cfg).is_some() {
        println!("the studio is already running ({})", web_line(cfg));
        return Ok(());
    }
    need_dist(cfg, "wfe/index.html")?;
    let sock_dir = cfg.socket_dir();
    std::fs::create_dir_all(&sock_dir)?;
    let listen = if bind.contains(':') && !bind.starts_with('[') { format!("[{bind}]:{port}") } else { format!("{bind}:{port}") };
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    let dir = cfg.get("H3_STUDIO_DIR").unwrap_or_else(|| format!("{home}/.local/share/sycl-h3"));
    let out = cfg.or("H3_OUT", "./out");
    let (pidf, listenf, logf) = studio_files(cfg);
    let exe = std::env::current_exe()?;
    let mut c = std::process::Command::new("setsid");
    c.arg(exe).args(["studio", "--listen", &listen, "--socket"]).arg(cfg.socket()).arg("--ui").arg(cfg.dist.join("wfe"))
        .args(["--out", &out, "--dir", &dir, "--gpustat", &cfg.or("H3_GPUSTAT", "/run/gpustat.json"), "--logs", &format!("{dir}/logs")]);
    if let Some(t) = cfg.get("H3_TEMPLATES") {
        c.args(["--templates", &t]);
    }
    if let Some(m) = cfg.get("H3_LLM_MODES") {
        c.args(["--llm-modes", &m]);
    }
    let log = std::fs::OpenOptions::new().create(true).append(true).open(&logf)?;
    let child = c.stdin(Stdio::null()).stdout(log.try_clone()?).stderr(log).spawn()?;
    std::fs::write(&pidf, format!("{}\n", child.id()))?;
    std::fs::write(&listenf, &listen)?;
    let reach = match bind.as_str() {
        "0.0.0.0" => format!("127.0.0.1:{port}"),
        "::" | "[::]" => format!("[::1]:{port}"),
        _ => listen.clone(),
    };
    wait_until("the studio", 20, || std::net::TcpStream::connect(&reach).is_ok())?;
    // setsid forks: the studio's own pid is the listener's
    if let Ok(o) = std::process::Command::new("pgrep").args(["-f", &format!("studio --listen {listen}")]).output() {
        if let Some(p) = String::from_utf8_lossy(&o.stdout).lines().last() {
            std::fs::write(&pidf, format!("{p}\n"))?;
        }
    }
    println!("{}", web_line(cfg));
    Ok(())
}

/// "studio: http://... " or "not running", for status and serve.
fn web_line(cfg: &Config) -> String {
    if studio_pid(cfg).is_none() {
        return "studio: not running (sycl-h3 serve)".into();
    }
    let listen = std::fs::read_to_string(studio_files(cfg).1).unwrap_or_default();
    let shown = match listen.split_once(':') {
        Some(("0.0.0.0", port)) => format!("{}:{port}", std::fs::read_to_string("/etc/hostname").unwrap_or_default().trim()),
        _ => listen,
    };
    format!("studio: http://{shown}/")
}

/// `sycl-h3 stop [--web | --all]`
fn stop(cfg: &Config, raw: &[String]) -> Result<()> {
    let web = raw.iter().any(|a| a == "--web");
    let all = raw.iter().any(|a| a == "--all");
    let rest: Vec<String> = raw.iter().filter(|a| *a != "--web" && *a != "--all").cloned().collect();
    no_more(&rest, "stop")?;
    if web || all {
        match studio_pid(cfg) {
            Some(pid) => {
                let _ = std::process::Command::new("kill").args(["-TERM", &pid.to_string()]).status();
                let t0 = Instant::now();
                while studio_pid(cfg).is_some() && t0.elapsed() < Duration::from_secs(10) {
                    std::thread::sleep(Duration::from_millis(200));
                }
                let _ = std::fs::remove_file(studio_files(cfg).0);
                println!("studio: stopped");
            }
            None => println!("studio: not running"),
        }
    }
    let ce = Ce::new(cfg)?;
    if !web || all {
        if ce.running(ENGINE) {
            // ask the daemon: the running jobs stop at their next block boundary, the engines unload, it ends
            let _ = client::post("/engine/shutdown", None);
            let t0 = Instant::now();
            while ce.running(ENGINE) && t0.elapsed() < Duration::from_secs(150) {
                std::thread::sleep(Duration::from_millis(500));
            }
            if ce.running(ENGINE) {
                ce.stop(ENGINE, 150); // the backstop: SIGTERM means the same to the daemon
            }
            ce.remove(ENGINE);
            println!("engine: stopped");
        } else {
            println!("engine: not running");
        }
    }
    Ok(())
}

/// `sycl-h3 gpus`: from the running daemon, or from a short-lived container.
fn gpus(cfg: &Config) -> Result<()> {
    let list = match client::get("/engine/gpus") {
        Ok(v) => v,
        Err(_) => {
            let ce = Ce::new(cfg)?;
            ce.need_image()?;
            need_dist(cfg, "h3d")?;
            let mut args: Vec<String> = vec!["run".into(), "--rm".into()];
            args.extend(ce.user_args());
            args.extend(ce.gpu_args());
            args.extend(mount(&cfg.dist, "/app", true));
            args.extend(["-e".into(), "H3SYCL_LIB=/app/libh3sycl.so".into(), "-e".into(), "ONEAPI_DEVICE_SELECTOR=level_zero:*".into()]);
            args.extend([ce.image.clone(), "bash".into(), "-c".into(),
                         "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec /app/h3d gpus --json".into()]);
            let out = ce.cmd().args(&args).stderr(Stdio::inherit()).output()?;
            if !out.status.success() {
                return Err(Error("could not list the GPUs".into()));
            }
            serde_json::from_slice(&out.stdout).map_err(|e| Error(format!("h3d gpus: {e}")))?
        }
    };
    println!("{:<4} {:<34} {:>8}  {:<14} SERVED", "GPU", "NAME", "MEMORY", "PCI");
    for g in list.as_array().cloned().unwrap_or_default() {
        println!("{:<4} {:<34} {:>5.1}GiB  {:<14} {}", g["index"].to_string(), g["name"].as_str().unwrap_or("?"), g["mem_gib"].as_f64().unwrap_or(0.0),
                 g["pci"].as_str().unwrap_or(""), match g["served"].as_bool() { Some(true) => "yes", Some(false) => "no", None => "-" });
    }
    Ok(())
}

fn logs(cfg: &Config, raw: &[String]) -> Result<()> {
    if raw.iter().any(|a| a == "--web") {
        let f = studio_files(cfg).2;
        let st = std::process::Command::new("tail").args(["-n", "60", "-f"]).arg(&f).status()?;
        return if st.success() { Ok(()) } else { Err(Error(format!("no log at {}", f.display()))) };
    }
    let ce = Ce::new(cfg)?;
    let st = ce.cmd().args(["logs", "-f", ENGINE]).status()?;
    if st.success() {
        Ok(())
    } else {
        Err(Error(format!("no log for {ENGINE} (is it running?)")))
    }
}

fn run() -> Result<()> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let cfg = Config::load();
    client::DAEMON.set(Target::Unix(cfg.socket())).expect("set once");
    let (cmd, rest) = match raw.split_first() {
        Some((c, r)) => (c.as_str(), r),
        None => ("help", &raw[..]),
    };
    match cmd {
        "start" => start(&cfg, rest),
        "serve" => serve(&cfg, rest),
        "stop" => stop(&cfg, rest),
        "status" => client::status(rest, &|| web_line(&cfg) + "\n"),
        "jobs" => client::jobs(rest),
        "unload" => client::unload(rest),
        "gpus" => gpus(&cfg),
        "logs" => logs(&cfg, rest),
        "version" | "--version" | "-V" => {
            println!("sycl-h3 {}", version());
            Ok(())
        }
        "speech" => tools::speech(&cfg, rest),
        "scene" => tools::scene(&cfg, rest),
        "join" => tools::join(&cfg, rest),
        "speechpct" => tools::speechpct(rest),
        // the studio's process (sycl-h3 serve starts it)
        "studio" => {
            let (listen, r) = take_value(rest, "--listen")?;
            let (socket, r) = take_value(&r, "--socket")?;
            let (ui, r) = take_value(&r, "--ui")?;
            let (out, r) = take_value(&r, "--out")?;
            let (dir, r) = take_value(&r, "--dir")?;
            let (gpustat, r) = take_value(&r, "--gpustat")?;
            let (logs, r) = take_value(&r, "--logs")?;
            let (templates, r) = take_value(&r, "--templates")?;
            let (llm, r) = take_value(&r, "--llm-modes")?;
            no_more(&r, "studio")?;
            let dir: PathBuf = dir.ok_or("--dir")?.into();
            studio::run(studio::Options {
                listen: listen.ok_or("--listen")?,
                socket: socket.ok_or("--socket")?.into(),
                ui: ui.ok_or("--ui")?.into(),
                out: out.ok_or("--out")?.into(),
                out_in: "/out".into(),
                logs: logs.map(PathBuf::from).unwrap_or_else(|| dir.join("logs")),
                dir,
                templates: templates.map(PathBuf::from),
                gpustat: gpustat.unwrap_or_else(|| "/run/gpustat.json".into()).into(),
                llm_modes: llm.map(PathBuf::from),
            })
        }
        "help" | "--help" | "-h" => {
            println!("{USAGE}");
            Ok(())
        }
        other => Err(Error(format!("unknown command {other:?}\n\n{USAGE}"))),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("sycl-h3: {e}");
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

    #[test]
    fn repeated_options() {
        let raw: Vec<String> = ["--gpu", "0", "--x", "--gpu", "1"].iter().map(|s| s.to_string()).collect();
        let (v, rest) = super::take_repeated(&raw, "--gpu").unwrap();
        assert_eq!(v, ["0", "1"]);
        assert_eq!(rest, ["--x"]);
        assert!(super::take_value(&raw, "--gpu").is_err());
    }
}
