//! Just enough HTTP/1.1, one request per connection, over TCP or a Unix socket - shared by the daemon (`h3d`, which
//! answers on a Unix socket) and the host command line (`sycl-h3`, which calls it, and whose `serve` answers the web
//! front end on TCP). JSON bodies for the API, files for the front end, and a byte-for-byte pass-through.
//! Standard library and serde_json only.

use std::fmt;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;

use serde_json::Value;

/// An error with a message for a person.
#[derive(Debug)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error(e.to_string())
    }
}
impl From<&str> for Error {
    fn from(e: &str) -> Self {
        Error(e.to_string())
    }
}
impl From<String> for Error {
    fn from(e: String) -> Self {
        Error(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Where a server is: `host:port`, or a Unix socket.
#[derive(Clone, Debug)]
pub enum Target {
    Tcp(String),
    Unix(PathBuf),
}

impl fmt::Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Target::Tcp(a) => f.write_str(a),
            Target::Unix(p) => write!(f, "{}", p.display()),
        }
    }
}

/// A connection of either kind.
pub enum Conn {
    Tcp(TcpStream),
    Unix(UnixStream),
}

impl Conn {
    pub fn connect(t: &Target) -> std::io::Result<Conn> {
        Ok(match t {
            Target::Tcp(a) => Conn::Tcp(TcpStream::connect(a)?),
            Target::Unix(p) => Conn::Unix(UnixStream::connect(p)?),
        })
    }
    pub fn set_read_timeout(&self, d: Option<Duration>) -> std::io::Result<()> {
        match self {
            Conn::Tcp(s) => s.set_read_timeout(d),
            Conn::Unix(s) => s.set_read_timeout(d),
        }
    }
}

impl Read for Conn {
    fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Conn::Tcp(s) => s.read(b),
            Conn::Unix(s) => s.read(b),
        }
    }
}

impl Write for Conn {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        match self {
            Conn::Tcp(s) => s.write(b),
            Conn::Unix(s) => s.write(b),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Conn::Tcp(s) => s.flush(),
            Conn::Unix(s) => s.flush(),
        }
    }
}

pub struct Request {
    pub method: String,
    /// the path without the query
    pub path: String,
    /// the path with the query, as received
    pub target: String,
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
pub fn read_request(stream: impl Read) -> Result<Request> {
    let mut r = BufReader::new(stream);
    let mut head = Vec::new();
    let mut line = String::new();
    r.read_line(&mut line)?;
    head.extend_from_slice(line.as_bytes());
    let mut parts = line.split_whitespace();
    let method = parts.next().ok_or("empty request")?.to_string();
    let target = parts.next().ok_or("no path in the request")?.to_string();
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
    Ok(Request { method, path, target, head, body })
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        409 => "Conflict",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        _ => "Error",
    }
}

pub fn respond_bytes(mut w: impl Write, status: u16, content_type: &str, body: &[u8]) -> Result<()> {
    write!(
        w,
        "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
        reason(status),
        body.len()
    )?;
    w.write_all(body)?;
    w.flush()?;
    Ok(())
}

pub fn respond(w: impl Write, status: u16, body: &Value) -> Result<()> {
    let text = serde_json::to_vec_pretty(body).map_err(|e| Error(e.to_string()))?;
    respond_bytes(w, status, "application/json", &text)
}

/// Passes a request on to `upstream` unchanged and copies the answer back as it arrives (long polls and file
/// downloads work).
pub fn forward(mut client: impl Write, upstream: &Target, req: &Request) -> Result<()> {
    let mut up = Conn::connect(upstream).map_err(|e| Error(format!("{upstream} does not answer ({e})")))?;
    up.write_all(&req.head)?;
    up.write_all(&req.body)?;
    up.flush()?;
    std::io::copy(&mut up, &mut client)?;
    Ok(())
}

/// One request; the JSON answer, or an error carrying the answer's message. `None` for the status when the server is
/// not there at all.
pub fn call(t: &Target, method: &str, path: &str, body: Option<&Value>) -> Result<Value> {
    let mut s = Conn::connect(t).map_err(|e| Error(format!("nothing answers at {t} ({e})")))?;
    s.set_read_timeout(Some(Duration::from_secs(60)))?;
    let text = body.map(|b| serde_json::to_vec(b).unwrap()).unwrap_or_default();
    write!(s, "{method} {path} HTTP/1.1\r\nHost: h3\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", text.len())?;
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

/// `http://host:port/path` -> (`host:port`, `/path`); plain http only.
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
