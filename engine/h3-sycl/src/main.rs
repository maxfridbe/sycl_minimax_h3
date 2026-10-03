//! `h3-sycl` - MiniMax H3 on Intel Arc GPUs: the command line, on the host. It starts two services in containers and
//! talks to the engine over a Unix socket; nothing of the engine runs in this process.
//!
//! ```text
//!   h3-sycl (you) ---- start/stop (podman) ----> [h3-sycl]      h3d daemon -- pipes --> h3d worker per GPU
//!        |                                            ^ Unix socket (JSON over HTTP)
//!        +---------- status/jobs/unload -------------+
//!        +---- serve/stop --web (podman) ----> [h3-sycl-web]   web front end, :8095 -> the same socket
//! ```
//!
//! A static binary: it runs on the host whatever the host's C library is.

mod client;
mod config;
mod container;
mod web;

use std::path::PathBuf;
use std::process::{ExitCode, Stdio};
use std::time::{Duration, Instant};

use h3_http::{Error, Result, Target};

use config::Config;
use container::{mount, Ce, ENGINE, SOCKET_DIR_IN, WEB};

pub(crate) fn version() -> String {
    let p: Vec<u32> = env!("CARGO_PKG_VERSION").split('.').map(|x| x.parse().unwrap_or(0)).collect();
    format!("{:02}.{:04}.{:03}", p[0], p[1], p[2])
}

const USAGE: &str = "h3-sycl - MiniMax H3 on Intel Arc GPUs

services (each in its own container; either runs without the other):
  h3-sycl start [--all | --gpu N ...] [--shared-gpu N ...]
                                the engine daemon; each GPU (all by default) gets its own engine process when a job
                                needs it, and the model stays loaded on it between jobs
  h3-sycl serve [--bind ADDR] [--port N]
                                the web front end (default 127.0.0.1:8095); it reaches the engine over the same socket
  h3-sycl stop [--web | --all]  stop the engine (the default), the web front end, or both - gracefully

the engine (over its socket):
  h3-sycl status [--no-stream]  live, one row per GPU (like docker stats)
  h3-sycl jobs ps [-a]          queued and running jobs (-a: all)
  h3-sycl jobs add <kind> [--name value ...] [-f]
                                queue a job (-f: follow its log); --gpu N pins it to one GPU
                                kinds: bench-blocks (--tokens N --blocks N), check-block (--dump /out/<file>)
  h3-sycl jobs stop <id>... | rem <id>... | details <id>
  h3-sycl unload [--gpu N]      give a GPU back now; the next job loads again

  h3-sycl gpus                  the GPUs, numbered as --gpu takes them
  h3-sycl logs [--web]          a service's log, followed
  h3-sycl version

settings (environment, or NAME=value lines in h3-sycl.conf beside the repository or ~/.config/h3-sycl.conf):
  H3_MODELS        host directory with the checkpoints, seen as /models          (required for start)
  H3_MODEL         the checkpoint as seen in the container
                   (default /models/kitchen/minimax_h3_fl2va_pruned_int8_convrot.safetensors)
  H3_OUT           host directory for job files, seen as /out                    (default ./out)
  H3_GPUS          GPUs to serve, e.g. \"0 1\" (default all)   H3_SHARED_GPUS  GPUs the two hooks below are about
  H3_IDLE          seconds without a job before an engine unloads; 0 = never     (default 600)
  H3_GPU_LOCK      a lock file shared with the GPU's other users
  H3_LLM_SWITCHER  a front end's model switcher URL: its model stops before loading, comes back after
  H3_LISTEN, H3_PORT   where serve listens (default 127.0.0.1, 8095; 0.0.0.0 = the network, no password)
  H3_LEGACY_API    the server the front end's not yet ported calls go to (e.g. http://127.0.0.1:8090)
  H3_SOCKET_DIR    where the engine's socket lives (default $XDG_RUNTIME_DIR/h3-sycl)
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
        Some(a) => Err(Error(format!("h3-sycl {cmd}: unexpected {a:?} (h3-sycl help)"))),
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
    Err(Error(format!("{what} did not come up in {secs} s (h3-sycl logs)")))
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

/// `h3-sycl start`: the engine daemon's container.
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
        g.parse::<usize>().map_err(|_| Error(format!("{g}: not a GPU number (h3-sycl gpus)")))?;
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
    if let Some(f) = cfg.get("H3S_MEM_FRACTION") {
        args.extend(["-e".into(), format!("H3S_MEM_FRACTION={f}")]);
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

/// `h3-sycl serve`: the web service's container.
fn serve(cfg: &Config, raw: &[String]) -> Result<()> {
    let (bind, rest) = take_value(raw, "--bind")?;
    let (port, rest) = take_value(&rest, "--port")?;
    no_more(&rest, "serve")?;
    let bind = bind.unwrap_or_else(|| cfg.or("H3_LISTEN", "127.0.0.1"));
    let port: u16 = port.unwrap_or_else(|| cfg.or("H3_PORT", "8095")).parse().map_err(|_| Error("--port: not a port".into()))?;
    let ce = Ce::new(cfg)?;
    if ce.running(WEB) {
        println!("the web front end is already running ({})", web_line(cfg, &ce));
        return Ok(());
    }
    ce.need_image()?;
    need_dist(cfg, "h3-sycl")?;
    need_dist(cfg, "wfe/index.html")?;
    ce.remove(WEB);
    let sock_dir = cfg.socket_dir();
    std::fs::create_dir_all(&sock_dir)?;
    let listen = if bind.contains(':') && !bind.starts_with('[') { format!("[{bind}]:{port}") } else { format!("{bind}:{port}") };
    let mut args: Vec<String> = vec!["run".into(), "-d".into(), "--name".into(), WEB.into(), "--network".into(), "host".into()];
    args.extend(["--stop-timeout".into(), "10".into()]);
    args.extend(ce.user_args());
    args.extend(mount(&cfg.dist, "/app", true));
    args.extend(mount(&sock_dir, SOCKET_DIR_IN, false));
    args.extend(["--label".into(), format!("h3-sycl.listen={listen}")]);
    args.extend([ce.image.clone(), "/app/h3-sycl".into(), "web-service".into(), "--listen".into(), listen.clone(),
                 "--socket".into(), format!("{SOCKET_DIR_IN}/h3d.sock"), "--ui".into(), "/app/wfe".into()]);
    if let Some(l) = cfg.get("H3_LEGACY_API") {
        args.extend(["--legacy-api".into(), l]);
    }
    let st = ce.cmd().args(&args).stdout(Stdio::null()).status()?;
    if !st.success() {
        return Err(Error(format!("{} run failed", ce.bin)));
    }
    let reach = match bind.as_str() {
        "0.0.0.0" => format!("127.0.0.1:{port}"),
        "::" | "[::]" => format!("[::1]:{port}"),
        _ => listen.clone(),
    };
    wait_until("the web front end", 20, || std::net::TcpStream::connect(&reach).is_ok())?;
    println!("{}", web_line(cfg, &ce));
    Ok(())
}

/// "web front end: http://... " or "not running", for status and serve.
fn web_line(_cfg: &Config, ce: &Ce) -> String {
    if !ce.running(WEB) {
        return "web front end: not running (h3-sycl serve)".into();
    }
    let listen = ce
        .cmd()
        .args(["container", "inspect", "-f", "{{index .Config.Labels \"h3-sycl.listen\"}}", WEB])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    let shown = match listen.split_once(':') {
        Some(("0.0.0.0", port)) => format!("{}:{port}", std::fs::read_to_string("/etc/hostname").unwrap_or_default().trim()),
        _ => listen,
    };
    format!("web front end: http://{shown}/")
}

/// `h3-sycl stop [--web | --all]`
fn stop(cfg: &Config, raw: &[String]) -> Result<()> {
    let web = raw.iter().any(|a| a == "--web");
    let all = raw.iter().any(|a| a == "--all");
    let rest: Vec<String> = raw.iter().filter(|a| *a != "--web" && *a != "--all").cloned().collect();
    no_more(&rest, "stop")?;
    let ce = Ce::new(cfg)?;
    if web || all {
        if ce.running(WEB) {
            ce.stop(WEB, 10);
            ce.remove(WEB);
            println!("web front end: stopped");
        } else {
            println!("web front end: not running");
        }
    }
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

/// `h3-sycl gpus`: from the running daemon, or from a short-lived container.
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
    let name = if raw.iter().any(|a| a == "--web") { WEB } else { ENGINE };
    let ce = Ce::new(cfg)?;
    let st = ce.cmd().args(["logs", "-f", name]).status()?;
    if st.success() {
        Ok(())
    } else {
        Err(Error(format!("no log for {name} (is it running?)")))
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
        "status" => {
            let ce = Ce::new(&cfg).ok();
            client::status(rest, &|| ce.as_ref().map_or(String::new(), |ce| web_line(&cfg, ce) + "\n"))
        }
        "jobs" => client::jobs(rest),
        "unload" => client::unload(rest),
        "gpus" => gpus(&cfg),
        "logs" => logs(&cfg, rest),
        "version" | "--version" | "-V" => {
            println!("h3-sycl {}", version());
            Ok(())
        }
        // inside the web service's container (h3-sycl serve starts it)
        "web-service" => {
            let (listen, r) = take_value(rest, "--listen")?;
            let (socket, r) = take_value(&r, "--socket")?;
            let (ui, r) = take_value(&r, "--ui")?;
            let (legacy, r) = take_value(&r, "--legacy-api")?;
            no_more(&r, "web-service")?;
            web::run(web::Options {
                listen: listen.ok_or("--listen")?,
                socket: socket.ok_or("--socket")?.into(),
                ui: ui.ok_or("--ui")?.into(),
                legacy_api: legacy,
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
            eprintln!("h3-sycl: {e}");
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
