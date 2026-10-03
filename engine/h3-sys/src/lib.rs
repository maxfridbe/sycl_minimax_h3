//! Raw bindings to `libh3sycl.so`: one function pointer per declaration in `kernels/h3sycl.h`.
//!
//! The library is opened at run time (`dlopen`), not linked: the Rust side then builds anywhere, without the SYCL
//! toolchain, and the kernels can be rebuilt without relinking the engine. Nothing here is safe to call directly;
//! `h3-core` wraps it.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};

/// Element types of floating-point tensors (`H3S_F32` ...).
pub const F32: c_int = 0;
pub const F16: c_int = 1;
pub const BF16: c_int = 2;

extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}
const RTLD_NOW: c_int = 2;
const RTLD_GLOBAL: c_int = 0x100;

/// The functions of `h3sycl.h`, resolved once.
#[allow(non_snake_case)]
pub struct Api {
    pub last_error: unsafe extern "C" fn() -> *const c_char,
    pub open: unsafe extern "C" fn() -> *mut c_void,
    pub gpu_count: unsafe extern "C" fn() -> c_int,
    pub gpu_info: unsafe extern "C" fn(c_int, *mut c_char, c_int, *mut u64, *mut c_char, c_int) -> c_int,
    pub open_gpu: unsafe extern "C" fn(c_int) -> *mut c_void,
    pub destroy: unsafe extern "C" fn(*mut c_void),
    pub device_name: unsafe extern "C" fn(*mut c_void) -> *const c_char,
    pub alloc: unsafe extern "C" fn(*mut c_void, u64) -> *mut c_void,
    pub free: unsafe extern "C" fn(*mut c_void, *mut c_void),
    pub mem_used: unsafe extern "C" fn(*mut c_void) -> u64,
    pub mem_cap: unsafe extern "C" fn(*mut c_void) -> u64,
    pub mem_free: unsafe extern "C" fn(*mut c_void) -> u64,
    pub write: unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_void, u64) -> c_int,
    pub read: unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_void, u64) -> c_int,
    pub wait: unsafe extern "C" fn(*mut c_void) -> c_int,
    pub copy: unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_void, u64) -> c_int,
    #[allow(clippy::type_complexity)]
    pub int8_linear: unsafe extern "C" fn(
        *mut c_void,    // ctx
        *const c_void,  // x
        c_int,          // x_dt
        i64,            // M
        i64,            // K
        *const i8,      // w
        i64,            // N
        *const f32,     // wscale
        i64,            // n_wscale
        *const f32,     // bias
        *mut c_void,    // out
        c_int,          // out_dt
        c_int,          // group
    ) -> c_int,
    #[allow(clippy::type_complexity)]
    pub linear: unsafe extern "C" fn(
        *mut c_void,    // ctx
        *const c_void,  // x
        c_int,          // dt
        i64,            // M
        i64,            // K
        *const c_void,  // w
        i64,            // N
        *const f32,     // bias
        *mut c_void,    // out
        c_int,          // out_dt
    ) -> c_int,
    #[allow(clippy::type_complexity)]
    pub conv1d: unsafe extern "C" fn(*mut c_void, *const f32, i64, i64, i64, *const f32, i64, i64, *const f32, i64, i64, i64, *mut f32, i64) -> c_int,
    pub conv_transpose1d: unsafe extern "C" fn(*mut c_void, *const f32, i64, i64, i64, *const f32, i64, i64, *const f32, i64, i64, *mut f32, i64) -> c_int,
    pub aa_snake: unsafe extern "C" fn(*mut c_void, *const f32, i64, i64, i64, *const f32, *const f32, *const f32, *const f32, *mut f32) -> c_int,
    pub scale: unsafe extern "C" fn(*mut c_void, *mut f32, i64, f32) -> c_int,
    pub layer_norm: unsafe extern "C" fn(*mut c_void, *const c_void, c_int, i64, i64, *const f32, *const f32, f32, *mut c_void, c_int) -> c_int,
    pub rms_norm_mod: unsafe extern "C" fn(
        *mut c_void,    // ctx
        *const c_void,  // x
        c_int,          // x_dt
        i64,            // M
        i64,            // C
        *const f32,     // weight
        f32,            // eps
        *const i32,     // rows
        *const f32,     // scale
        *const f32,     // shift
        *mut c_void,    // out
        c_int,          // out_dt
    ) -> c_int,
    #[allow(clippy::type_complexity)]
    pub rms_rope: unsafe extern "C" fn(
        *mut c_void,    // ctx
        *mut c_void,    // x
        c_int,          // x_dt
        i64,            // M
        i64,            // H
        i64,            // D
        i64,            // stride
        *const f32,     // weight
        f32,            // eps
        *const f32,     // cs
        c_int,          // rot_dim
    ) -> c_int,
    pub swiglu: unsafe extern "C" fn(*mut c_void, *const c_void, c_int, i64, i64, *mut c_void, c_int) -> c_int,
    #[allow(clippy::type_complexity)]
    pub gate_add: unsafe extern "C" fn(
        *mut c_void,    // ctx
        *mut c_void,    // x
        c_int,          // x_dt
        i64,            // M
        i64,            // C
        *const c_void,  // other
        c_int,          // other_dt
        *const i32,     // rows
        *const f32,     // gate
    ) -> c_int,
    #[allow(clippy::type_complexity)]
    pub attention: unsafe extern "C" fn(
        *mut c_void,    // ctx
        *const c_void,  // q
        *const c_void,  // k
        *const c_void,  // v
        c_int,          // dt
        i64,            // S
        i64,            // H
        i64,            // D
        i64,            // stride
        *mut c_void,    // out
        c_int,          // out_dt
    ) -> c_int,
}

fn dl_error() -> String {
    // SAFETY: dlerror returns NULL or a NUL-terminated string owned by libc.
    unsafe {
        let e = dlerror();
        if e.is_null() {
            "unknown dlopen error".into()
        } else {
            CStr::from_ptr(e).to_string_lossy().into_owned()
        }
    }
}

/// Where the library is looked for: `$H3SYCL_LIB`, then beside the executable, then the loader's search path.
pub fn candidates() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Ok(p) = std::env::var("H3SYCL_LIB") {
        v.push(PathBuf::from(p));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            v.push(dir.join("libh3sycl.so"));
        }
    }
    v.push(PathBuf::from("libh3sycl.so"));
    v
}

impl Api {
    /// Opens the library and resolves every function. The handle is never closed: the kernels live as long as the
    /// process.
    pub fn load() -> Result<Api, String> {
        let mut tried = Vec::new();
        for path in candidates() {
            if path.components().count() > 1 && !path.exists() {
                tried.push(format!("{} (no such file)", path.display()));
                continue;
            }
            match Self::load_from(&path) {
                Ok(api) => return Ok(api),
                Err(e) => tried.push(format!("{}: {e}", path.display())),
            }
        }
        Err(format!("libh3sycl.so could not be loaded:\n  {}", tried.join("\n  ")))
    }

    // each field's declared type is the annotation: the transmute target is inferred from it
    #[allow(clippy::missing_transmute_annotations)]
    pub fn load_from(path: &Path) -> Result<Api, String> {
        let c = CString::new(path.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
        // SAFETY: dlopen with a valid C string; the handle is checked.
        let h = unsafe { dlopen(c.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
        if h.is_null() {
            return Err(dl_error());
        }
        macro_rules! sym {
            ($name:literal) => {{
                let n = CString::new($name).unwrap();
                // SAFETY: `h` is a live handle; the symbol's type is the one declared in h3sycl.h.
                let p = unsafe { dlsym(h, n.as_ptr()) };
                if p.is_null() {
                    return Err(format!("{} is missing from the library ({})", $name, dl_error()));
                }
                unsafe { std::mem::transmute::<*mut c_void, _>(p) }
            }};
        }
        Ok(Api {
            last_error: sym!("h3s_last_error"),
            open: sym!("h3s_open"),
            gpu_count: sym!("h3s_gpu_count"),
            gpu_info: sym!("h3s_gpu_info"),
            open_gpu: sym!("h3s_open_gpu"),
            destroy: sym!("h3s_destroy"),
            device_name: sym!("h3s_device_name"),
            alloc: sym!("h3s_alloc"),
            free: sym!("h3s_free"),
            mem_used: sym!("h3s_mem_used"),
            mem_cap: sym!("h3s_mem_cap"),
            mem_free: sym!("h3s_mem_free"),
            write: sym!("h3s_write"),
            read: sym!("h3s_read"),
            wait: sym!("h3s_wait"),
            copy: sym!("h3s_copy"),
            int8_linear: sym!("h3s_int8_linear"),
            linear: sym!("h3s_linear"),
            conv1d: sym!("h3s_conv1d"),
            conv_transpose1d: sym!("h3s_conv_transpose1d"),
            aa_snake: sym!("h3s_aa_snake"),
            scale: sym!("h3s_scale"),
            layer_norm: sym!("h3s_layer_norm"),
            rms_norm_mod: sym!("h3s_rms_norm_mod"),
            rms_rope: sym!("h3s_rms_rope"),
            swiglu: sym!("h3s_swiglu"),
            gate_add: sym!("h3s_gate_add"),
            attention: sym!("h3s_attention"),
        })
    }

    /// The library's last error on this thread.
    pub fn error(&self) -> String {
        // SAFETY: h3s_last_error returns a NUL-terminated string that lives until the thread's next failing call.
        unsafe { CStr::from_ptr((self.last_error)()).to_string_lossy().into_owned() }
    }
}
