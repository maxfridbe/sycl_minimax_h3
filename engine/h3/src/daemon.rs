//! `h3 serve`: the engine as a resident service, in two processes. This one - the daemon - serves the web front end
//! and the API, keeps the job queue, and never opens the GPU. The GPU belongs to a child process, `h3 worker`
//! (worker.rs), which the daemon starts when a job needs the engine: it loads the model and keeps it for the next
//! jobs. The daemon ends it when told to (`unload`, `shutdown`) or after a while without jobs, so a GPU shared with
//! other programs is not held for nothing - and the memory comes back with the process, whatever state the driver
//! was in. A crash in the engine ends the worker, not the front end: the job is marked failed and the next one starts
//! a fresh worker.
//!
//! Jobs run one at a time, in the order they came. A job can be cancelled; it stops at the next block boundary, never
//! inside a kernel (a GPU process stopped mid-kernel can leave the xe driver stuck).
//!
//! The same port serves the web front end (wfe/, built into dist/wfe) at `/`, and the engine's API (JSON) under
//! `/engine`:
//!
//! ```text
//!   GET  /engine/status              engine state, the job running, the queue
//!   GET  /engine/jobs                every job, newest first, without logs
//!   GET  /engine/jobs/<id>           one job with its log and result
//!   POST /engine/jobs                {"kind": "bench-blocks", ...}  ->  {"id": 3}
//!   POST /engine/jobs/<id>/cancel
//!   POST /engine/unload              give the GPU back now (after the running job)
//!   POST /engine/shutdown            cancel the queue, finish or cancel the running job, unload, exit
//! ```
//!
//! The front end also calls the API of the clip-production server it was written for (`/api/...`, `/rpc/...`:
//! the clip queue, projects, scenes, films, the LLM switch). Until each of those is ported here, a request the daemon
//! does not handle itself is passed on, byte for byte, to that server (`--legacy-api`), so the front end keeps
//! working whole while its pieces move over.
//!
//! On a GPU that another program normally holds, two hooks make the daemon share it politely before loading:
//! a lock file (wait until it is free, then hold it while loaded) and a front end's model switcher (ask the model it
//! serves to stop, and put it back after unloading). Either way the daemon then waits until the card really has the
//! memory free - it never loads into a card that is still occupied.

use std::collections::{BTreeMap, VecDeque};
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use h3_core::{Error, Result};
use serde_json::{json, Value};

use crate::http;

/// The worker process and what the daemon took on its behalf (to give back when it ends).
struct Proc {
    child: Child,
    stdin: ChildStdin,
    events: mpsc::Receiver<Value>,
    saved_mode: Option<String>,
}

pub struct Options {
    pub listen: String,
    pub model: PathBuf,
    /// Unload after this long without a job; `None`: stay loaded until told.
    pub idle: Option<Duration>,
    /// A lock file shared with the GPU's other users: wait while someone else's name is in it.
    pub gpu_lock: Option<PathBuf>,
    /// A front end's model switcher (`POST {"mode": ...}`, answers `{"result": {"mode": ...}}`): its model is asked
    /// to stop before loading and restored after unloading.
    pub llm_switcher: Option<String>,
    pub threads: usize,
    /// The built web front end (dist/wfe), served at `/`.
    pub ui: Option<PathBuf>,
    /// The legacy front-end server (`http://host:port`) that answers what is not ported yet.
    pub legacy_api: Option<String>,
}

struct JobRec {
    id: u64,
    spec: Value,
    state: &'static str, // queued, running, done, failed, cancelled
    log: Vec<String>,
    result: Option<Value>,
    error: Option<String>,
    cancel: Arc<AtomicBool>,
    created: f64,
    started: Option<f64>,
    finished: Option<f64>,
}

impl JobRec {
    fn summary(&self) -> Value {
        json!({"id": self.id, "kind": self.spec.get("kind"), "state": self.state, "created": self.created,
               "started": self.started, "finished": self.finished, "error": self.error,
               "last": self.log.last()})
    }
    fn full(&self) -> Value {
        let mut v = self.summary();
        v["spec"] = self.spec.clone();
        v["log"] = json!(self.log);
        v["result"] = self.result.clone().unwrap_or(Value::Null);
        v
    }
}

struct Shared {
    jobs: BTreeMap<u64, JobRec>,
    queue: VecDeque<u64>,
    next: u64,
    engine: String, // unloaded, waiting for the GPU..., loading, loaded, unloading
    running: Option<u64>,
    last_active: Instant,
    unload_requested: bool,
    shutdown: bool,
    info: Value,
}

struct Daemon {
    opts: Options,
    s: Mutex<Shared>,
    cv: Condvar,
}

fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

impl Daemon {
    fn log_job(&self, id: u64, line: String) {
        eprintln!("[job {id}] {line}");
        if let Some(j) = self.s.lock().unwrap().jobs.get_mut(&id) {
            j.log.push(line);
        }
    }

    fn set_engine(&self, state: impl Into<String>) {
        let st = state.into();
        eprintln!("[engine] {st}");
        self.s.lock().unwrap().engine = st;
    }

    fn stopping(&self, cancel: &AtomicBool) -> bool {
        cancel.load(Ordering::Relaxed) || self.s.lock().unwrap().shutdown
    }

    /// The lock file: wait until it is absent or ours, then write our name into it.
    fn take_lock(&self, cancel: &AtomicBool) -> Result<()> {
        let Some(path) = &self.opts.gpu_lock else { return Ok(()) };
        loop {
            let holder = std::fs::read_to_string(path).ok();
            match holder {
                Some(h) if !h.starts_with("h3-sycl ") => {
                    self.set_engine(format!("waiting for the GPU lock (held: {})", h.trim()));
                    if self.stopping(cancel) {
                        return Err(Error("cancelled while waiting for the GPU lock".into()));
                    }
                    std::thread::sleep(Duration::from_secs(5));
                }
                _ => {
                    std::fs::write(path, format!("h3-sycl {} engine loaded\n", now() as u64))?;
                    return Ok(());
                }
            }
        }
    }

    fn release_lock(&self) {
        if let Some(path) = &self.opts.gpu_lock {
            if std::fs::read_to_string(path).is_ok_and(|h| h.starts_with("h3-sycl ")) {
                let _ = std::fs::remove_file(path);
            }
        }
    }

    /// Asks the front end's model to stop; returns the mode to restore later.
    fn switch_llm_off(&self) -> Result<Option<String>> {
        let Some(url) = &self.opts.llm_switcher else { return Ok(None) };
        let (host, path) = http::split_url(url)?;
        let cur = http::call(&host, "POST", &path, Some(&json!({})))?;
        let mode = cur.pointer("/result/mode").and_then(|m| m.as_str()).unwrap_or("none").to_string();
        if mode != "none" {
            http::call(&host, "POST", &path, Some(&json!({"mode": "none"})))?;
        }
        Ok(Some(mode))
    }

    fn restore_llm(&self, mode: &Option<String>) {
        let (Some(url), Some(mode)) = (&self.opts.llm_switcher, mode) else { return };
        if mode == "none" {
            return;
        }
        match http::split_url(url).and_then(|(h, p)| http::call(&h, "POST", &p, Some(&json!({"mode": mode})))) {
            Ok(_) => eprintln!("[engine] the front end's model {mode:?} restored"),
            Err(e) => eprintln!("[engine] could not restore the front end's model {mode:?}: {e}"),
        }
    }

    /// Lock and switcher (both cheap, no GPU), then the worker process, which waits for the card's memory and loads.
    fn start_worker(&self, cancel: &AtomicBool) -> Result<Proc> {
        self.take_lock(cancel)?;
        let saved_mode = match self.switch_llm_off() {
            Ok(m) => m,
            Err(e) => {
                self.release_lock();
                return Err(e);
            }
        };
        let spawned = Command::new(std::env::current_exe()?)
            .args(["worker", "--model"])
            .arg(&self.opts.model)
            .args(["--threads", &self.opts.threads.to_string()])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn();
        let mut child = match spawned {
            Ok(c) => c,
            Err(e) => {
                self.restore_llm(&saved_mode);
                self.release_lock();
                return Err(Error(format!("cannot start the engine process: {e}")));
            }
        };
        let (tx, events) = mpsc::channel();
        let out = child.stdout.take().unwrap();
        std::thread::spawn(move || {
            for line in BufReader::new(out).lines() {
                let Ok(line) = line else { break };
                if let Ok(v) = serde_json::from_str::<Value>(&line) {
                    if tx.send(v).is_err() {
                        break;
                    }
                }
            }
        });
        let stdin = child.stdin.take().unwrap();
        self.set_engine("starting the engine process");
        Ok(Proc { child, stdin, events, saved_mode })
    }

    /// What the worker said that is not about a job.
    fn engine_event(&self, ev: &Value) {
        match ev["event"].as_str() {
            Some("engine") => {
                if let Some(st) = ev["state"].as_str() {
                    self.set_engine(st);
                }
            }
            Some("ready") => {
                self.s.lock().unwrap().info = ev["info"].clone();
                self.set_engine("loaded");
            }
            _ => {}
        }
    }

    /// Ends the worker: asks it to exit (it finishes the kernel it is in, unloads, ends), waits for it, and gives back
    /// the lock and the front end's model. Never a kill: a GPU process stopped mid-kernel can leave the driver stuck.
    fn stop_worker(&self, proc: &mut Option<Proc>) {
        let Some(mut p) = proc.take() else { return };
        self.set_engine("unloading");
        let _ = writeln!(p.stdin, "{}", json!({"exit": true}));
        let _ = p.stdin.flush();
        drop(p.stdin);
        while let Ok(ev) = p.events.recv() {
            self.engine_event(&ev);
        }
        let status = p.child.wait();
        eprintln!("[engine] the engine process ended ({})", status.map_or("unknown".into(), |s| s.to_string()));
        self.restore_llm(&p.saved_mode);
        self.release_lock();
        self.s.lock().unwrap().info = Value::Null;
        self.set_engine("unloaded");
    }

    /// The worker ended on its own (a crash): clean up after it.
    fn reap_if_dead(&self, proc: &mut Option<Proc>) -> Option<String> {
        let p = proc.as_mut()?;
        match p.child.try_wait() {
            Ok(Some(status)) => {
                let p = proc.take().unwrap();
                self.restore_llm(&p.saved_mode);
                self.release_lock();
                self.s.lock().unwrap().info = Value::Null;
                self.set_engine("unloaded");
                Some(format!("the engine process ended unexpectedly ({status})"))
            }
            _ => None,
        }
    }

    /// Hands a job to the worker and relays its events until the result.
    fn run_job(&self, proc: &mut Option<Proc>, id: u64, spec: &Value, cancel: &AtomicBool) -> std::result::Result<Value, String> {
        if proc.is_none() {
            *proc = Some(self.start_worker(cancel).map_err(|e| e.0)?);
        }
        let p = proc.as_mut().unwrap();
        writeln!(p.stdin, "{}", json!({"job": id, "spec": spec})).map_err(|e| format!("the engine process does not listen: {e}"))?;
        let _ = p.stdin.flush();
        let mut cancel_sent = false;
        loop {
            if cancel.load(Ordering::Relaxed) && !cancel_sent {
                let p = proc.as_mut().unwrap();
                let _ = writeln!(p.stdin, "{}", json!({"cancel": id}));
                let _ = p.stdin.flush();
                cancel_sent = true;
            }
            let ev = proc.as_ref().unwrap().events.recv_timeout(Duration::from_millis(200));
            match ev {
                Ok(ev) => match ev["event"].as_str() {
                    Some("log") => self.log_job(id, ev["line"].as_str().unwrap_or("").to_string()),
                    Some("result") if ev["job"].as_u64() == Some(id) => {
                        return if ev["ok"].as_bool() == Some(true) {
                            Ok(ev["value"].clone())
                        } else {
                            Err(ev["error"].as_str().unwrap_or("failed").to_string())
                        };
                    }
                    _ => {
                        if let Some(l) = ev["line"].as_str() {
                            self.log_job(id, l.to_string());
                        }
                        self.engine_event(&ev);
                    }
                },
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    // its output closed: it has ended or is ending
                    let mut waited = 0;
                    loop {
                        if let Some(why) = self.reap_if_dead(proc) {
                            return Err(why);
                        }
                        std::thread::sleep(Duration::from_millis(200));
                        waited += 1;
                        if waited > 600 {
                            return Err("the engine process closed its output but did not end".into());
                        }
                    }
                }
            }
        }
    }

    /// The one thread that deals with the worker process.
    fn worker(self: Arc<Self>) {
        enum Next {
            Run(u64, Value, Arc<AtomicBool>),
            Unload,
            Exit,
        }
        let mut proc: Option<Proc> = None;
        loop {
            if let Some(why) = self.reap_if_dead(&mut proc) {
                eprintln!("[engine] {why}");
            }
            // events the worker sends between jobs (none expected, but keep the pipe drained)
            if let Some(p) = &proc {
                while let Ok(ev) = p.events.try_recv() {
                    self.engine_event(&ev);
                }
            }
            let next = {
                let mut s = self.s.lock().unwrap();
                loop {
                    if s.shutdown {
                        break Next::Exit;
                    }
                    if s.unload_requested {
                        s.unload_requested = false;
                        break Next::Unload;
                    }
                    if let Some(id) = s.queue.pop_front() {
                        let j = s.jobs.get_mut(&id).unwrap();
                        j.state = "running";
                        j.started = Some(now());
                        let (spec, cancel) = (j.spec.clone(), j.cancel.clone());
                        s.running = Some(id);
                        break Next::Run(id, spec, cancel);
                    }
                    let idle = s.last_active.elapsed();
                    match self.opts.idle {
                        Some(limit) if proc.is_some() && idle >= limit => break Next::Unload,
                        _ => {}
                    }
                    let (guard, timeout) = self.cv.wait_timeout(s, Duration::from_secs(1)).unwrap();
                    s = guard;
                    if timeout.timed_out() && proc.is_some() {
                        break Next::Run(0, Value::Null, Arc::new(AtomicBool::new(false))); // a turn of the outer loop
                    }
                }
            };
            match next {
                Next::Exit => {
                    self.stop_worker(&mut proc);
                    return;
                }
                Next::Unload => self.stop_worker(&mut proc),
                Next::Run(0, _, _) => {}
                Next::Run(id, spec, cancel) => {
                    let outcome = self.run_job(&mut proc, id, &spec, &cancel);
                    let mut s = self.s.lock().unwrap();
                    let j = s.jobs.get_mut(&id).unwrap();
                    j.finished = Some(now());
                    match outcome {
                        Ok(v) => {
                            j.state = "done";
                            j.result = Some(v);
                        }
                        Err(e) => {
                            j.state = if cancel.load(Ordering::Relaxed) { "cancelled" } else { "failed" };
                            j.error = Some(e);
                        }
                    }
                    eprintln!("[job {id}] {}", j.state);
                    s.running = None;
                    s.last_active = Instant::now();
                }
            }
        }
    }

    /// Everything that arrives on the port: the engine's API, the front end's files, or the pass-through.
    fn route(&self, stream: &std::net::TcpStream, req: http::Request) -> Result<()> {
        let path = req.path.clone();
        if let Some(rest) = path.strip_prefix("/engine/") {
            let (status, body) = match req.json() {
                Ok(v) => self.handle(&req.method, rest, v),
                Err(e) => (400, json!({"error": e.0})),
            };
            return http::respond(stream, status, &body);
        }
        if let Some(ui) = &self.opts.ui {
            if req.method == "GET" && (path == "/" || path == "/index.html") {
                // the page carries its style sheet inline (the legacy server's layout of the same build)
                let html = std::fs::read_to_string(ui.join("index.html"))?;
                let css = std::fs::read_to_string(ui.join("style.css")).unwrap_or_default();
                return http::respond_bytes(stream, 200, "text/html; charset=utf-8", html.replace("__CSS__", &css).as_bytes());
            }
            if let Some(rel) = path.strip_prefix("/ui/") {
                if req.method == "GET" && !rel.split('/').any(|c| c == ".." || c.is_empty()) {
                    return match std::fs::read(ui.join(rel)) {
                        Ok(b) => http::respond_bytes(stream, 200, http::content_type(rel), &b),
                        Err(_) => http::respond(stream, 404, &json!({"error": format!("no front-end file {rel}")})),
                    };
                }
            }
        }
        match &self.opts.legacy_api {
            Some(url) => {
                let (host, _) = http::split_url(url)?;
                if let Err(e) = http::forward(stream, &host, &req) {
                    let _ = http::respond(stream, 502, &json!({"error": e.0}));
                }
                Ok(())
            }
            None => http::respond(stream, 404, &json!({"error": format!("no route {} {}", req.method, path)})),
        }
    }

    fn handle(&self, method: &str, path: &str, body: Value) -> (u16, Value) {
        let parts: Vec<&str> = path.trim_matches('/').split('/').collect();
        let mut s = self.s.lock().unwrap();
        match (method, parts.as_slice()) {
            ("GET", ["status"]) => {
                let idle = s.last_active.elapsed().as_secs();
                (200, json!({"version": crate::version(), "engine": s.engine, "info": s.info, "running": s.running,
                             "queued": s.queue.iter().collect::<Vec<_>>(), "idle_seconds": idle,
                             "unload_after_idle_seconds": self.opts.idle.map(|d| d.as_secs()), "model": self.opts.model}))
            }
            ("GET", ["jobs"]) => (200, json!(s.jobs.values().rev().map(JobRec::summary).collect::<Vec<_>>())),
            ("GET", ["jobs", id]) => match id.parse::<u64>().ok().and_then(|i| s.jobs.get(&i)) {
                Some(j) => (200, j.full()),
                None => (404, json!({"error": format!("no job {id}")})),
            },
            ("POST", ["jobs"]) => {
                if s.shutdown {
                    return (409, json!({"error": "the daemon is shutting down"}));
                }
                if body.get("kind").and_then(|k| k.as_str()).is_none() {
                    return (400, json!({"error": "a job needs a \"kind\" (bench-blocks, check-block)"}));
                }
                let id = s.next;
                s.next += 1;
                s.jobs.insert(id, JobRec { id, spec: body, state: "queued", log: Vec::new(), result: None, error: None,
                                           cancel: Arc::new(AtomicBool::new(false)), created: now(), started: None, finished: None });
                s.queue.push_back(id);
                self.cv.notify_all();
                (200, json!({"id": id}))
            }
            ("POST", ["jobs", id, "cancel"]) => {
                let Some(i) = id.parse::<u64>().ok() else { return (404, json!({"error": format!("no job {id}")})) };
                s.queue.retain(|q| *q != i);
                match s.jobs.get_mut(&i) {
                    Some(j) if j.state == "queued" => {
                        j.state = "cancelled";
                        j.finished = Some(now());
                        (200, json!({"id": i, "state": "cancelled"}))
                    }
                    Some(j) if j.state == "running" => {
                        j.cancel.store(true, Ordering::Relaxed);
                        (200, json!({"id": i, "state": "cancelling: it stops at the next block boundary"}))
                    }
                    Some(j) => (200, json!({"id": i, "state": j.state})),
                    None => (404, json!({"error": format!("no job {i}")})),
                }
            }
            ("POST", ["unload"]) => {
                s.unload_requested = true;
                self.cv.notify_all();
                (200, json!({"unload": if s.running.is_some() { "after the running job" } else { "now" }}))
            }
            ("POST", ["shutdown"]) => {
                s.shutdown = true;
                for id in std::mem::take(&mut s.queue) {
                    if let Some(j) = s.jobs.get_mut(&id) {
                        j.state = "cancelled";
                    }
                }
                if let Some(r) = s.running {
                    s.jobs.get(&r).unwrap().cancel.store(true, Ordering::Relaxed);
                }
                self.cv.notify_all();
                (200, json!({"shutdown": "the running job stops at its next block boundary, then the engine unloads"}))
            }
            _ => (404, json!({"error": format!("no route {method} /engine/{path}")})),
        }
    }
}

pub fn serve(opts: Options) -> Result<()> {
    if !opts.model.exists() {
        return Err(Error(format!("{}: no such checkpoint", opts.model.display())));
    }
    let listener = TcpListener::bind(&opts.listen).map_err(|e| Error(format!("cannot listen on {}: {e}", opts.listen)))?;
    eprintln!("h3 {} serving on {} - model {} (loaded on the first job){}", crate::version(), opts.listen, opts.model.display(),
              opts.idle.map_or(String::new(), |d| format!(", unloaded after {} s idle", d.as_secs())));
    if let Some(ui) = &opts.ui {
        eprintln!("  web front end: {} at http://{}/", ui.display(), opts.listen);
    }
    if let Some(l) = &opts.legacy_api {
        eprintln!("  not yet ported front-end calls go to {l}");
    }
    let d = Arc::new(Daemon {
        opts,
        s: Mutex::new(Shared { jobs: BTreeMap::new(), queue: VecDeque::new(), next: 1, engine: "unloaded".into(), running: None,
                               last_active: Instant::now(), unload_requested: false, shutdown: false, info: Value::Null }),
        cv: Condvar::new(),
    });
    let worker = {
        let d = d.clone();
        std::thread::spawn(move || d.worker())
    };
    // SIGTERM (podman stop) -> the same as POST /shutdown
    crate::signals::install();
    listener.set_nonblocking(true)?;
    loop {
        if crate::signals::terminated() {
            let mut s = d.s.lock().unwrap();
            if !s.shutdown {
                eprintln!("[daemon] terminate signal: shutting down");
                s.shutdown = true;
                if let Some(r) = s.running {
                    s.jobs.get(&r).unwrap().cancel.store(true, Ordering::Relaxed);
                }
                d.cv.notify_all();
            }
        }
        if worker.is_finished() {
            break;
        }
        match listener.accept() {
            Ok((stream, _)) => {
                let d = d.clone();
                std::thread::spawn(move || {
                    let _ = stream.set_nonblocking(false);
                    match http::read_request(&stream) {
                        Ok(req) => {
                            if let Err(e) = d.route(&stream, req) {
                                let _ = http::respond(&stream, 500, &json!({"error": e.0}));
                            }
                        }
                        Err(e) => {
                            let _ = http::respond(&stream, 400, &json!({"error": e.0}));
                        }
                    }
                });
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => std::thread::sleep(Duration::from_millis(50)),
            Err(e) => eprintln!("[daemon] accept: {e}"),
        }
    }
    let _ = worker.join();
    eprintln!("[daemon] stopped");
    Ok(())
}
