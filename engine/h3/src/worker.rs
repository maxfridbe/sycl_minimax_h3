//! `h3 worker --gpu N`: the process that holds GPU N (one of `h3 gpus`). `h3 serve` starts it when a job needs the engine and lets it end
//! when the engine is to be unloaded, so the GPU's memory goes back when the process exits - whatever state the
//! driver was in - and a fault in a kernel takes down this process, not the daemon or the front end.
//!
//! It speaks JSON lines: requests on stdin, events on stdout (logs go to stderr as well, for the container log).
//!
//! ```text
//!   in:   {"job": 3, "spec": {"kind": "bench-blocks", ...}}     run a job (one at a time, in order)
//!         {"cancel": 3}                                       stop job 3 at its next block boundary
//!         {"exit": true}                                      unload and end (also: stdin closed)
//!   out:  {"event": "engine", "state": "waiting for the GPU: ..."}
//!         {"event": "ready", "info": {...}}                   the model is on the GPU
//!         {"event": "log", "job": 3, "line": "..."}
//!         {"event": "result", "job": 3, "ok": true, "value": {...}}
//!         {"event": "result", "job": 3, "ok": false, "error": "...", "cancelled": false}
//!         {"event": "progress", "job": 3, "done": 12, "total": 100}
//!         {"event": "stats", "gib_in_use": 21.3, "gib_cap": 30.0, "gib_free_card": 8.1}   once a second
//! ```

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use h3_core::device::Device;
use h3_core::safetensors::Checkpoint;
use h3_core::Result;
use serde_json::{json, Value};

use crate::jobs::{self, gib, Ctl, Engine};

fn emit(v: Value) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{v}");
    let _ = out.flush();
}

pub fn run(gpu: usize, model: PathBuf, threads: usize) -> Result<()> {
    // stdin is read on its own thread: a cancel must get through while a job runs on this one
    let current = Arc::new(AtomicU64::new(0)); // the job running now (0: none)
    let cancel = Arc::new(AtomicBool::new(false)); // for the job running now
    let abort = Arc::new(AtomicBool::new(false)); // exit requested
    let (tx, rx) = mpsc::channel::<(u64, Value)>();
    {
        let (current, cancel, abort) = (current.clone(), cancel.clone(), abort.clone());
        std::thread::spawn(move || {
            for line in std::io::stdin().lock().lines() {
                let Ok(line) = line else { break };
                let Ok(v) = serde_json::from_str::<Value>(&line) else { continue };
                if let Some(id) = v.get("cancel").and_then(|c| c.as_u64()) {
                    if current.load(Ordering::SeqCst) == id {
                        cancel.store(true, Ordering::SeqCst);
                    }
                } else if v.get("exit").is_some() {
                    break;
                } else if let (Some(id), Some(spec)) = (v.get("job").and_then(|j| j.as_u64()), v.get("spec")) {
                    let _ = tx.send((id, spec.clone()));
                }
            }
            // exit asked for, or the daemon went away: stop what runs and end
            abort.store(true, Ordering::SeqCst);
            cancel.store(true, Ordering::SeqCst);
        });
    }

    // wait until the card has the room, then load
    let dev = Device::open_index(gpu)?;
    let need = (Checkpoint::open(&model)?.data_bytes() + (8u64 << 30)).min(dev.mem_cap());
    loop {
        if abort.load(Ordering::SeqCst) {
            return Ok(());
        }
        match dev.mem_free() {
            Some(free) if free < need => {
                emit(json!({"event": "engine", "state": format!("waiting for the GPU: {:.1} GiB free, {:.1} GiB needed", gib(free), gib(need))}));
                std::thread::sleep(Duration::from_secs(5));
            }
            _ => break,
        }
    }
    emit(json!({"event": "engine", "state": "loading"}));
    let mut log = |l: String| {
        eprintln!("[worker] {l}");
        emit(json!({"event": "engine", "state": "loading", "line": l}));
    };
    let e = Engine::load_on(dev, &model, None, threads, &mut log)?;
    emit(json!({"event": "ready", "info": {"gpu": gpu, "device": e.dev.name(), "model": model, "blocks": e.model.blocks.len(),
                                           "gib_in_use": gib(e.dev.mem_used()), "gib_cap": gib(e.dev.mem_cap())}}));
    // once a second: what the engine holds on the card and what the card has free (for `h3 status`)
    let stats_stop = Arc::new(AtomicBool::new(false));
    let stats = {
        let (dev, stop) = (e.dev.clone(), stats_stop.clone());
        std::thread::spawn(move || {
            while !stop.load(Ordering::SeqCst) {
                emit(json!({"event": "stats", "gib_in_use": gib(dev.mem_used()), "gib_cap": gib(dev.mem_cap()),
                            "gib_free_card": dev.mem_free().map(gib)}));
                std::thread::sleep(Duration::from_secs(1));
            }
        })
    };

    loop {
        let (id, spec) = match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(j) => j,
            Err(mpsc::RecvTimeoutError::Timeout) if !abort.load(Ordering::SeqCst) => continue,
            _ => break,
        };
        cancel.store(false, Ordering::SeqCst);
        current.store(id, Ordering::SeqCst);
        let mut log = |l: String| {
            eprintln!("[job {id}] {l}");
            emit(json!({"event": "log", "job": id, "line": l}));
        };
        let mut progress = |done: usize, total: usize| emit(json!({"event": "progress", "job": id, "done": done, "total": total}));
        let out = jobs::run(&e, &spec, &mut Ctl { log: &mut log, cancel: &cancel, progress: Some(&mut progress) });
        current.store(0, Ordering::SeqCst);
        match out {
            Ok(v) => emit(json!({"event": "result", "job": id, "ok": true, "value": v})),
            Err(err) => emit(json!({"event": "result", "job": id, "ok": false, "error": err.0,
                                    "cancelled": cancel.load(Ordering::SeqCst)})),
        }
        if abort.load(Ordering::SeqCst) {
            break;
        }
    }
    stats_stop.store(true, Ordering::SeqCst);
    let _ = stats.join(); // it holds the device: it must be gone before the context can go
    drop(e); // every GPU buffer, then the context; the process ends right after
    emit(json!({"event": "engine", "state": "unloaded"}));
    Ok(())
}
