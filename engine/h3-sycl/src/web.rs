//! The web service (`h3-sycl serve` starts it in its own container; it runs `h3-sycl web-service`): the web front end
//! (dist/wfe) over HTTP, the engine's API passed through to the daemon's Unix socket, and the front end's calls that
//! are not ported yet passed through to the server they were written for. It holds no state: stopping or restarting
//! it touches no job and no GPU.

use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;

use h3_http::{self as http, Error, Result, Target};
use serde_json::json;

pub struct Options {
    pub listen: String,
    pub socket: PathBuf,
    pub ui: PathBuf,
    pub legacy_api: Option<String>,
}

fn route(o: &Options, stream: &TcpStream, req: http::Request) -> Result<()> {
    let path = req.path.as_str();
    if path.starts_with("/engine/") {
        // the engine's API: the daemon answers on its socket
        if let Err(e) = http::forward(stream, &Target::Unix(o.socket.clone()), &req) {
            let _ = http::respond(stream, 503, &json!({"error": format!("the engine is not running (h3-sycl start): {}", e.0)}));
        }
        return Ok(());
    }
    if req.method == "GET" && (path == "/" || path == "/index.html") {
        // the page carries its style sheet inline (the layout the legacy server used for the same build)
        let html = std::fs::read_to_string(o.ui.join("index.html"))?;
        let css = std::fs::read_to_string(o.ui.join("style.css")).unwrap_or_default();
        return http::respond_bytes(stream, 200, "text/html; charset=utf-8", html.replace("__CSS__", &css).as_bytes());
    }
    if let Some(rel) = path.strip_prefix("/ui/") {
        if req.method == "GET" && !rel.split('/').any(|c| c == ".." || c.is_empty()) {
            return match std::fs::read(o.ui.join(rel)) {
                Ok(b) => http::respond_bytes(stream, 200, http::content_type(rel), &b),
                Err(_) => http::respond(stream, 404, &json!({"error": format!("no front-end file {rel}")})),
            };
        }
    }
    match &o.legacy_api {
        Some(url) => {
            let (host, _) = http::split_url(url)?;
            if let Err(e) = http::forward(stream, &Target::Tcp(host), &req) {
                let _ = http::respond(stream, 502, &json!({"error": e.0}));
            }
            Ok(())
        }
        None => http::respond(stream, 404, &json!({"error": format!("no route {} {path}", req.method)})),
    }
}

pub fn run(o: Options) -> Result<()> {
    if !o.ui.join("index.html").exists() {
        return Err(Error(format!("{}: no built front end there (./build.sh wfe)", o.ui.display())));
    }
    let listener = TcpListener::bind(&o.listen).map_err(|e| Error(format!("cannot listen on {}: {e}", o.listen)))?;
    eprintln!("h3-sycl web service {} on http://{}/ - engine API via {}{}", crate::version(), o.listen, o.socket.display(),
              o.legacy_api.as_ref().map_or(String::new(), |l| format!(", the rest to {l}")));
    let o = std::sync::Arc::new(o);
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let o = o.clone();
        std::thread::spawn(move || match http::read_request(&stream) {
            Ok(req) => {
                if let Err(e) = route(&o, &stream, req) {
                    let _ = http::respond(&stream, 500, &json!({"error": e.0}));
                }
            }
            Err(e) => {
                let _ = http::respond(&stream, 400, &json!({"error": e.0}));
            }
        });
    }
    Ok(())
}
