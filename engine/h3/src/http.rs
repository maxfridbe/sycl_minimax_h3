//! Just enough HTTP/1.1 for the daemon: one request per connection. JSON for the engine's API, files for the web
//! front end, and a byte-for-byte pass-through to the legacy front-end server for what is not ported yet.
//! Standard library only.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use h3_core::{Error, Result};
use serde_json::Value;

pub struct Request {
    pub method: String,
    /// the path without the query
    pub path: String,
    /// the request line and headers exactly as received, for passing on
    pub head: Vec<u8>,
    pub body: Vec<u8>,
}

impl Request {
    pub fn json(&self) -> Result<Value> {
        if self.body.is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_slice(&self.body).map_err(|e| Error(format!("the body is not JSON: {e}")))
    }
}

/// Reads one request from a connection.
pub fn read_request(stream: &TcpStream) -> Result<Request> {
    let mut r = BufReader::new(stream);
    let mut head = Vec::new();
    let mut line = String::new();
    r.read_line(&mut line)?;
    head.extend_from_slice(line.as_bytes());
    let mut parts = line.split_whitespace();
    let method = parts.next().ok_or("empty request")?.to_string();
    let target = parts.next().ok_or("no path in the request")?;
    let path = target.split('?').next().unwrap_or("/").to_string();
    let mut length = 0usize;
    loop {
        let mut h = String::new();
        let n = r.read_line(&mut h)?;
        head.extend_from_slice(h.as_bytes());
        if n == 0 || h == "\r\n" || h == "\n" {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                length = v.trim().parse().map_err(|_| Error("bad Content-Length".into()))?;
            }
        }
    }
    if length > 64 << 20 {
        return Err(Error("request body over 64 MiB".into()));
    }
    let mut body = vec![0u8; length];
    r.read_exact(&mut body)?;
    Ok(Request { method, path, head, body })
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        409 => "Conflict",
        502 => "Bad Gateway",
        _ => "Error",
    }
}

pub fn respond_bytes(mut stream: &TcpStream, status: u16, content_type: &str, body: &[u8]) -> Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
        reason(status),
        body.len()
    )?;
    stream.write_all(body)?;
    Ok(())
}

pub fn respond(stream: &TcpStream, status: u16, body: &Value) -> Result<()> {
    let text = serde_json::to_vec_pretty(body).map_err(|e| Error(e.to_string()))?;
    respond_bytes(stream, status, "application/json", &text)
}

/// Passes a request on to `upstream` (`host:port`) unchanged and copies the answer back as it arrives (so long polls
/// and file downloads work).
pub fn forward(mut client: &TcpStream, upstream: &str, req: &Request) -> Result<()> {
    let mut up = TcpStream::connect(upstream).map_err(|e| Error(format!("the legacy server at {upstream} does not answer ({e})")))?;
    up.write_all(&req.head)?;
    up.write_all(&req.body)?;
    up.flush()?;
    std::io::copy(&mut up, &mut client)?;
    Ok(())
}

/// One request to `addr` (`host:port`); the JSON answer, or an error carrying the answer's message.
pub fn call(addr: &str, method: &str, path: &str, body: Option<&Value>) -> Result<Value> {
    let mut s = TcpStream::connect(addr).map_err(|e| Error(format!("no daemon at {addr} ({e}); start it with `h3-sycl start`")))?;
    s.set_read_timeout(Some(Duration::from_secs(60)))?;
    let text = body.map(|b| serde_json::to_vec(b).unwrap()).unwrap_or_default();
    write!(s, "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", text.len())?;
    s.write_all(&text)?;
    let mut all = Vec::new();
    s.read_to_end(&mut all)?;
    let all = String::from_utf8_lossy(&all);
    let (head, body) = all.split_once("\r\n\r\n").ok_or("a malformed answer")?;
    let status: u16 = head.split_whitespace().nth(1).and_then(|c| c.parse().ok()).ok_or("a malformed status line")?;
    let v: Value = if body.trim().is_empty() { Value::Null } else { serde_json::from_str(body).map_err(|e| Error(format!("the answer is not JSON: {e}")))? };
    if status == 200 {
        Ok(v)
    } else {
        Err(Error(v.get("error").and_then(|e| e.as_str()).unwrap_or("request failed").to_string()))
    }
}

/// `http://host:port/path` -> (`host:port`, `/path`); plain http only (the daemon talks to local services).
pub fn split_url(url: &str) -> Result<(String, String)> {
    let rest = url.strip_prefix("http://").ok_or_else(|| Error(format!("{url}: only http:// URLs")))?;
    let (host, path) = rest.split_once('/').map_or((rest, "/".to_string()), |(h, p)| (h, format!("/{p}")));
    let host = if host.contains(':') { host.to_string() } else { format!("{host}:80") };
    Ok((host, path))
}

/// The content type of a front-end file, by extension.
pub fn content_type(path: &str) -> &'static str {
    match path.rsplit('.').next().unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" | "map" => "application/json",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "webp" => "image/webp",
        "woff2" => "font/woff2",
        _ => "application/octet-stream",
    }
}
