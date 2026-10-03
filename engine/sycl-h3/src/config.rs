//! Settings: the environment, then `sycl-h3.conf` beside the repository (one level above dist/), then
//! `~/.config/sycl-h3.conf`; the first that has a name wins. The files are `NAME=value` lines (`#` comments, quotes
//! around the value optional) - the same names as the environment variables.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub struct Config {
    files: BTreeMap<String, String>,
    /// dist/: where this program and everything it starts live
    pub dist: PathBuf,
}

fn parse(text: &str, into: &mut BTreeMap<String, String>) {
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        if let Some((k, v)) = line.split_once('=') {
            let v = v.trim();
            let v = v.strip_prefix('"').and_then(|v| v.strip_suffix('"')).or_else(|| v.strip_prefix('\'').and_then(|v| v.strip_suffix('\''))).unwrap_or(v);
            into.entry(k.trim().to_string()).or_insert_with(|| v.to_string());
        }
    }
}

impl Config {
    pub fn load() -> Config {
        let exe = std::env::current_exe().ok().and_then(|p| p.canonicalize().ok()).unwrap_or_default();
        let dist = std::env::var("H3_DIST").map(PathBuf::from).unwrap_or_else(|_| exe.parent().map(Path::to_path_buf).unwrap_or_default());
        let mut files = BTreeMap::new();
        let mut candidates = vec![dist.join("../sycl-h3.conf")];
        if let Ok(home) = std::env::var("HOME") {
            candidates.push(PathBuf::from(home).join(".config/sycl-h3.conf"));
        }
        for f in candidates {
            if let Ok(t) = std::fs::read_to_string(&f) {
                parse(&t, &mut files);
            }
        }
        Config { files, dist }
    }

    pub fn get(&self, name: &str) -> Option<String> {
        std::env::var(name).ok().filter(|v| !v.is_empty()).or_else(|| self.files.get(name).cloned().filter(|v| !v.is_empty()))
    }

    pub fn or(&self, name: &str, default: &str) -> String {
        self.get(name).unwrap_or_else(|| default.to_string())
    }

    /// The directory holding the daemon's socket, on the host.
    pub fn socket_dir(&self) -> PathBuf {
        if let Some(d) = self.get("H3_SOCKET_DIR") {
            return PathBuf::from(d);
        }
        let base = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".into());
        PathBuf::from(base).join("sycl-h3")
    }

    pub fn socket(&self) -> PathBuf {
        self.socket_dir().join("h3d.sock")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conf_lines() {
        let mut m = BTreeMap::new();
        parse("# a comment\nH3_MODELS=/m\nexport H3_PORT=\"9000\"\n  H3_IDLE = '60'\nH3_MODELS=/later\n", &mut m);
        assert_eq!(m["H3_MODELS"], "/m");
        assert_eq!(m["H3_PORT"], "9000");
        assert_eq!(m["H3_IDLE"], "60");
    }
}
