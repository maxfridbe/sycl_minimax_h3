//! The daemon's command-line client.
//!
//!     h3 status [--no-stream]          the engine, live (like `docker stats`): state, the engine process, the GPU,
//!                                      the job running, the queue; --no-stream: once
//!     h3 jobs ps [-a]                  queued and running jobs (-a: every job the daemon remembers)
//!     h3 jobs add <kind> [--name value ...] [-f]
//!                                      queue a job; -f: follow its log until it ends
//!     h3 jobs stop <id>...             cancel: a queued job is dropped, a running one stops at its next block boundary
//!     h3 jobs rem <id>...              forget finished or queued jobs
//!     h3 jobs details <id>             everything about one job: its request, progress, log, result
//!     h3 unload | shutdown
//!
//! The daemon is found at `$H3_DAEMON` (default 127.0.0.1:8095).

use std::io::{IsTerminal, Write};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use h3_core::{Error, Result};
use serde_json::{json, Value};

use crate::http;

fn addr() -> String {
    std::env::var("H3_DAEMON").unwrap_or_else(|_| "127.0.0.1:8095".into())
}

fn get(path: &str) -> Result<Value> {
    http::call(&addr(), "GET", path, None)
}

fn post(path: &str, body: Option<&Value>) -> Result<Value> {
    http::call(&addr(), "POST", path, body)
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
    let short = |e: &str| -> String {
        // the long states ("waiting for the GPU: 2.4 GiB free, ...") go on their own line below
        e.split(':').next().unwrap_or(e).to_string()
    };
    let engine = st["engine"].as_str().unwrap_or("?");
    let gpu = &st["gpu"];
    let job = match st["running"].as_object() {
        Some(_) => {
            let r = &st["running"];
            format!("{} {} {} ({})", r["id"], r["kind"].as_str().unwrap_or("?"), progress(r),
                    dur(r["elapsed_seconds"].as_f64().unwrap_or(0.0)))
        }
        None => "-".into(),
    };
    let engine_mem = match (gpu["engine_gib"].as_f64(), gpu["cap_gib"].as_f64()) {
        (Some(u), Some(c)) => format!("{u:.1} / {c:.1} GiB"),
        _ => "-".into(),
    };
    let mut out = String::new();
    out += &format!(
        "{:<22} {:>7} {:>8} {:>8} {:>17} {:>9}  {:<30} {:>5} {:>9}\n",
        "ENGINE", "PID", "RSS", "GPU BUSY", "ENGINE GPU MEM", "CARD FREE", "JOB", "QUEUE", "UNLOAD IN"
    );
    out += &format!(
        "{:<22} {:>7} {:>8} {:>8} {:>17} {:>9}  {:<30} {:>5} {:>9}\n",
        short(engine),
        st["worker"]["pid"].as_u64().map_or("-".into(), |p| p.to_string()),
        st["worker"]["rss_gib"].as_f64().map_or("-".into(), |r| format!("{r:.1}GiB")),
        gpu["busy_pct"].as_f64().map_or("-".into(), |b| format!("{b:.0}%")),
        engine_mem,
        gpu["card_free_gib"].as_f64().map_or("-".into(), |f| format!("{f:.1}GiB")),
        job,
        st["queued"].as_array().map_or(0, |q| q.len()),
        st["unload_in_seconds"].as_f64().map_or("-".into(), dur),
    );
    if engine.contains(':') {
        out += &format!("  {engine}\n");
    }
    out
}

/// `h3 status [--no-stream]`
pub fn status(raw: &[String]) -> Result<()> {
    let once = raw.iter().any(|a| a == "--no-stream");
    let tty = std::io::stdout().is_terminal();
    loop {
        let st = get("/engine/status")?;
        let table = status_table(&st);
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
    let mut out = format!("{:<5} {:<14} {:<10} {:>9} {:>9} {:>9}  {}\n", "ID", "KIND", "STATE", "PROGRESS", "CREATED", "TOOK", "LAST");
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
            "{:<5} {:<14} {:<10} {:>9} {:>9} {:>9}  {}\n",
            j["id"],
            j["kind"].as_str().unwrap_or("?"),
            j["state"].as_str().unwrap_or("?"),
            progress(j),
            j["created"].as_f64().map_or("-".into(), |c| format!("{} ago", dur(t - c))),
            took,
            last
        );
    }
    out
}

/// `h3 jobs ps|add|stop|rem|details ...`
pub fn jobs(raw: &[String]) -> Result<()> {
    let usage = "usage: h3 jobs ps [-a] | add <kind> [--name value ...] [-f] | stop <id>... | rem <id>... | details <id>";
    let sub = raw.first().ok_or(usage)?.as_str();
    let rest = &raw[1..];
    let ids = || -> Result<Vec<u64>> {
        if rest.is_empty() {
            return Err(Error(format!("h3 jobs {sub} needs at least one job id")));
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

/// `h3 jobs add <kind> [--name value ...] [-f]`: numbers become numbers, everything else strings.
fn add(raw: &[String]) -> Result<()> {
    let follow = raw.iter().any(|a| a == "-f" || a == "--follow");
    let rest: Vec<&String> = raw.iter().filter(|a| *a != "-f" && *a != "--follow").collect();
    let kind = rest.first().ok_or("h3 jobs add needs a kind: bench-blocks, check-block")?;
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

/// `h3 unload` / `h3 shutdown`
pub fn simple(cmd: &str) -> Result<()> {
    let v = post(&format!("/engine/{cmd}"), None)?;
    let msg = v.as_object().and_then(|o| o.values().next()).and_then(|m| m.as_str()).unwrap_or("ok");
    println!("{cmd}: {msg}");
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
    fn status_line_has_the_running_job() {
        let st = json!({"engine": "loaded", "worker": {"pid": 42, "rss_gib": 2.04},
                        "gpu": {"busy_pct": 97.4, "engine_gib": 21.3, "cap_gib": 30.0, "card_free_gib": 8.1},
                        "running": {"id": 3, "kind": "bench-blocks", "progress": {"done": 24, "total": 100}, "elapsed_seconds": 14.2},
                        "queued": [4, 5], "unload_in_seconds": null});
        let t = status_table(&st);
        let row = t.lines().nth(1).unwrap();
        for want in ["loaded", "42", "2.0GiB", "97%", "21.3 / 30.0 GiB", "8.1GiB", "3 bench-blocks 24/100 (14s)", " 2 "] {
            assert!(row.contains(want), "{want:?} missing from {row:?}");
        }
    }
}
