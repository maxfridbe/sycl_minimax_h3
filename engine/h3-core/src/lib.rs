//! The MiniMax H3 engine's host side. The arithmetic runs in `libh3sycl` (SYCL, `kernels/`); this crate owns
//! everything around it: the device and its memory, reading checkpoints, and (as the port proceeds) the model graph
//! and the sampler.

pub mod audio;
pub mod denoiser;
pub mod device;
pub mod dit;
pub mod dtype;
pub mod layout;
pub mod noise;
pub mod load;
pub mod ops;
pub mod reference;
pub mod rng;
pub mod safetensors;
pub mod upscale;
pub mod vae;

pub type Result<T> = std::result::Result<T, Error>;

/// One error type for the engine: a message, with the context added on the way up.
#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error(e.to_string())
    }
}
impl From<String> for Error {
    fn from(e: String) -> Self {
        Error(e)
    }
}
impl From<&str> for Error {
    fn from(e: &str) -> Self {
        Error(e.to_string())
    }
}

/// Adds what was being done to an error: `read(..).ctx("reading the header")?`.
pub trait Ctx<T> {
    fn ctx(self, what: impl std::fmt::Display) -> Result<T>;
}
impl<T, E: std::fmt::Display> Ctx<T> for std::result::Result<T, E> {
    fn ctx(self, what: impl std::fmt::Display) -> Result<T> {
        self.map_err(|e| Error(format!("{what}: {e}")))
    }
}
