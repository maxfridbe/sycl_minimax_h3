//! Talking to the daemon over its socket: the status view and the job commands.
//!
//!     h3-sycl status [--no-stream]          the engine, live (like `docker stats`): state, the engine process, the GPU,
//!                                      the job running, the queue; --no-stream: once
//!     h3-sycl jobs ps [-a]                  queued and running jobs (-a: every job the daemon remembers)
//!     h3-sycl jobs add <kind> [--name value ...] [-f]
//!                                      queue a job; -f: follow its log until it ends
//!     h3-sycl jobs stop <id>...             cancel: a queued job is dropped, a running one stops at its next block boundary
//!     h3-sycl jobs rem <id>...              forget finished or queued jobs
//!     h3-sycl jobs details <id>             everything about one job: its request, progress, log, result
//!     h3-sycl unload [--gpu N]
//!
//! The daemon answers on a Unix socket (see config.rs); `main` sets where.

use std::io::{IsTerminal, Write};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use std::sync::OnceLock;

use h3_http::{self as http, Error, Result, Target};
use serde_json::{json, Value};

/// The daemon's socket; set once by `main`.
pub static DAEMON: OnceLock<Target> = OnceLock::new();

fn target() -> &'static Target {
    DAEMON.get().expect("the daemon's socket is set before any call")
}

pub fn get(path: &str) -> Result<Value> {
    http::call(target(), "GET", path, None).map_err(not_running)
}

pub fn post(path: &str, body: Option<&Value>) -> Result<Value> {
    http::call(target(), "POST", path, body).map_err(not_running)
}

fn not_running(e: Error) -> Error {
    if e.0.starts_with("nothing answers") {
        Error("the engine is not running: h3-sycl start".into())
    } else {
        e
    }
}

fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

/// 75 -> "1m15s"
fn dur(s: f64) -> String {
    let s = s.max(0.0) as u64;
    match s {
        0..=59 => format!("{s}s"),
        60..=3599 => format!("{}m{:02}s", s / 60, s % 60),
        _ => format!("{}h{:02}m", s / 3600, s % 3600 / 60),
    }
}

fn progress(v: &Value) -> String {
    match (v["progress"]["done"].as_u64(), v["progress"]["total"].as_u64()) {
        (Some(d), Some(t)) if t > 0 => format!("{d}/{t}"),
        _ => "-".into(),
    }
}

fn status_table(st: &Value) -> String {
    let row = |c: [&str; 10]| {
        format!("{:<4} {:<24} {:>7} {:>8} {:>8} {:>17} {:>9}  {:<30} {:>9}  {}\n", c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9])
    };
    let mut out = row(["GPU", "ENGINE", "PID", "RSS", "BUSY", "ENGINE GPU MEM", "CARD FREE", "JOB", "UNLOAD IN", "CARD"]);
    let mut notes = String::new();
    for g in st["gpus"].as_array().cloned().unwrap_or_default() {
        let engine = g["engine"].as_str().unwrap_or("?");
        // a long state ("waiting for the GPU: 2.4 GiB free, ...") gets its own line under the table
        if engine.contains(':') {
            notes += &format!("  GPU {}: {engine}\n", g["gpu"]);
        }
        let job = match g["running"].as_object() {
            Some(_) => {
                let r = &g["running"];
                format!("{} {} {} ({})", r["id"], r["kind"].as_str().unwrap_or("?"), progress(r),
                        dur(r["elapsed_seconds"].as_f64().unwrap_or(0.0)))
            }
            None => "-".into(),
        };
        let mem = match (g["engine_gib"].as_f64(), g["cap_gib"].as_f64()) {
            (Some(u), Some(c)) => format!("{u:.1} / {c:.1} GiB"),
            _ => "-".into(),
        };
        let card = format!("{} {:.0}GiB{}", g["name"].as_str().unwrap_or("?").replace("Intel(R) ", "").replace("(TM)", ""),
                           g["mem_gib"].as_f64().unwrap_or(0.0), if g["shared"].as_bool() == Some(true) { " shared" } else { "" });
        out += &row([
            &g["gpu"].to_string(),
            engine.split(':').next().unwrap_or(engine),
            &g["worker"]["pid"].as_u64().map_or("-".into(), |p| p.to_string()),
            &g["worker"]["rss_gib"].as_f64().map_or("-".into(), |r| format!("{r:.1}GiB")),
            &g["busy_pct"].as_f64().map_or("-".into(), |b| format!("{b:.0}%")),
            &mem,
            &g["card_free_gib"].as_f64().map_or("-".into(), |f| format!("{f:.1}GiB")),
            &job,
            &g["unload_in_seconds"].as_f64().map_or("-".into(), dur),
            &card,
        ]);
    }
    let queued = st["queued"].as_array().map_or(0, |q| q.len());
    out + &notes + &format!("queued: {queued}\n")
}

/// `h3-sycl status [--no-stream]`; `extra` adds lines under the table (the web service's state).
pub fn status(raw: &[String], extra: &dyn Fn() -> String) -> Result<()> {
    let once = raw.iter().any(|a| a == "--no-stream");
    let tty = std::io::stdout().is_terminal();
    loop {
        let table = match get("/engine/status") {
            Ok(st) => status_table(&st),
            Err(e) => format!("engine: {e}\n"),
        } + &extra();
        let mut out = std::io::stdout().lock();
        if once {
            write!(out, "{table}")?;
            return Ok(());
        }
        if tty {
            // redraw in place: home, the table, clear what is left of the screen
            write!(out, "\x1b[H{}\x1b[J", table.replace('\n', "\x1b[K\n"))?;
        } else {
            write!(out, "{table}")?;
        }
        out.flush()?;
        drop(out);
        std::thread::sleep(Duration::from_secs(1));
    }
}

fn jobs_table(list: &[Value]) -> String {
    let mut out = format!("{:<5} {:<14} {:<10} {:>4} {:>9} {:>9} {:>9}  {}\n", "ID", "KIND", "STATE", "GPU", "PROGRESS", "CREATED", "TOOK", "LAST");
    let t = now();
    for j in list {
        let took = match (j["started"].as_f64(), j["finished"].as_f64()) {
            (Some(a), Some(b)) => dur(b - a),
            (Some(a), None) => dur(t - a),
            _ => "-".into(),
        };
        let last = j["error"].as_str().or(j["last"].as_str()).unwrap_or("").trim().to_string();
        let last: String = last.chars().take(60).collect();
        out += &format!(
            "{:<5} {:<14} {:<10} {:>4} {:>9} {:>9} {:>9}  {}\n",
            j["id"].to_string(),
            j["kind"].as_str().unwrap_or("?"),
            j["state"].as_str().unwrap_or("?"),
            j["gpu"].as_u64().map_or("-".into(), |g| g.to_string()),
            progress(j),
            j["created"].as_f64().map_or("-".into(), |c| format!("{} ago", dur(t - c))),
            took,
            last
        );
    }
    out
}

/// `h3-sycl jobs ps|add|stop|rem|details ...`
pub fn jobs(raw: &[String]) -> Result<()> {
    let usage = "usage: h3-sycl jobs ps [-a] | add <kind> [--name value ...] [-f] | stop <id>... | rem <id>... | details <id>";
    let sub = raw.first().ok_or(usage)?.as_str();
    let rest = &raw[1..];
    let ids = || -> Result<Vec<u64>> {
        if rest.is_empty() {
            return Err(Error(format!("h3-sycl jobs {sub} needs at least one job id")));
        }
        rest.iter().map(|s| s.parse::<u64>().map_err(|_| Error(format!("{s}: not a job id")))).collect()
    };
    match sub {
        "ps" | "ls" | "list" => {
            let all = rest.iter().any(|a| a == "-a" || a == "--all");
            let list = get("/engine/jobs")?;
            let list: Vec<Value> = list
                .as_array()
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .filter(|j| all || matches!(j["state"].as_str(), Some("queued" | "running")))
                .collect();
            print!("{}", jobs_table(&list));
            Ok(())
        }
        "add" => add(rest),
        "stop" | "cancel" | "kill" => {
            for id in ids()? {
                let v = post(&format!("/engine/jobs/{id}/cancel"), None)?;
                println!("{id}: {}", v["state"].as_str().unwrap_or("?"));
            }
            Ok(())
        }
        "rem" | "rm" | "remove" => {
            let mut failed = false;
            for id in ids()? {
                match post(&format!("/engine/jobs/{id}/remove"), None) {
                    Ok(_) => println!("{id}: removed"),
                    Err(e) => {
                        eprintln!("{id}: {e}");
                        failed = true;
                    }
                }
            }
            if failed {
                Err(Error("not every job was removed".into()))
            } else {
                Ok(())
            }
        }
        "details" | "inspect" | "show" => {
            let id = ids()?;
            let v = get(&format!("/engine/jobs/{}", id[0]))?;
            println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
            Ok(())
        }
        _ => Err(Error(usage.into())),
    }
}

/// `h3-sycl jobs add <kind> [--name value ...] [-f]`: numbers become numbers, everything else strings.
fn add(raw: &[String]) -> Result<()> {
    let follow = raw.iter().any(|a| a == "-f" || a == "--follow");
    let rest: Vec<&String> = raw.iter().filter(|a| *a != "-f" && *a != "--follow").collect();
    let kind = rest.first().ok_or("h3-sycl jobs add needs a kind: bench-blocks, check-block")?;
    let mut spec = serde_json::Map::new();
    spec.insert("kind".into(), json!(kind));
    let mut it = rest[1..].iter();
    while let Some(k) = it.next() {
        let name = k.strip_prefix("--").ok_or_else(|| Error(format!("{k}: job options are --name value")))?;
        let v = it.next().ok_or_else(|| Error(format!("--{name} needs a value")))?;
        spec.insert(name.to_string(), v.parse::<u64>().map(Value::from).unwrap_or_else(|_| json!(v)));
    }
    let id = post("/engine/jobs", Some(&Value::Object(spec)))?["id"].as_u64().ok_or("no job id in the answer")?;
    println!("{id}");
    if !follow {
        return Ok(());
    }
    // follow the log until the job ends
    let mut shown = 0;
    let mut engine_shown = String::new();
    loop {
        let j = get(&format!("/engine/jobs/{id}"))?;
        let log = j["log"].as_array().cloned().unwrap_or_default();
        for l in &log[shown.min(log.len())..] {
            println!("{}", l.as_str().unwrap_or(""));
        }
        shown = log.len();
        let state = j["state"].as_str().unwrap_or("");
        if state == "queued" || (state == "running" && shown == 0) {
            let e = get("/engine/status")?["engine"].as_str().unwrap_or("").to_string();
            if e != engine_shown && e != "loaded" {
                println!("({state}; engine: {e})");
                engine_shown = e;
            }
        }
        match state {
            "done" => return Ok(()),
            "failed" | "cancelled" => return Err(Error(format!("job {id} {state}: {}", j["error"].as_str().unwrap_or("")))),
            _ => std::thread::sleep(Duration::from_millis(700)),
        }
    }
}

/// `h3-sycl unload [--gpu N]`
pub fn unload(raw: &[String]) -> Result<()> {
    let body = match raw {
        [] => json!({}),
        [flag, n] if flag == "--gpu" => json!({"gpu": n.parse::<u64>().map_err(|_| Error(format!("--gpu {n}: not a GPU number")))?}),
        _ => return Err(Error("usage: h3-sycl unload [--gpu N]".into())),
    };
    let v = post("/engine/unload", Some(&body))?;
    for u in v["unload"].as_array().cloned().unwrap_or_default() {
        println!("GPU {}: unload {}", u["gpu"], u["unload"].as_str().unwrap_or("?"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations() {
        assert_eq!(dur(5.4), "5s");
        assert_eq!(dur(75.0), "1m15s");
        assert_eq!(dur(3725.0), "1h02m");
    }

    #[test]
    fn status_rows_per_gpu() {
        let st = json!({"gpus": [
            {"gpu": 0, "name": "Intel(R) Arc(TM) Pro B70 Graphics", "mem_gib": 31.9, "shared": true, "engine": "loaded",
             "worker": {"pid": 42, "rss_gib": 2.04}, "busy_pct": 97.4, "engine_gib": 21.3, "cap_gib": 30.0, "card_free_gib": 8.1,
             "running": {"id": 3, "kind": "bench-blocks", "progress": {"done": 24, "total": 100}, "elapsed_seconds": 14.2},
             "unload_in_seconds": null},
            {"gpu": 1, "name": "Intel(R) Arc(TM) Pro B65 Graphics", "mem_gib": 24.0, "shared": false,
             "engine": "waiting for the GPU: 2.4 GiB free, 27.5 GiB needed", "worker": {"pid": 43, "rss_gib": 0.3},
             "busy_pct": null, "running": null, "unload_in_seconds": null}],
            "queued": [4, 5]});
        let t = status_table(&st);
        let lines: Vec<&str> = t.lines().collect();
        for want in ["loaded", "42", "2.0GiB", "97%", "21.3 / 30.0 GiB", "8.1GiB", "3 bench-blocks 24/100 (14s)", "Arc Pro B70 Graphics 32GiB shared"] {
            assert!(lines[1].contains(want), "{want:?} missing from {:?}", lines[1]);
        }
        assert!(lines[2].starts_with("1    waiting for the GPU "), "{:?}", lines[2]);
        assert!(t.contains("GPU 1: waiting for the GPU: 2.4 GiB free"));
        assert!(t.ends_with("queued: 2\n"));
    }
}
