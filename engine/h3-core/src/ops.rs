//! The kernels, with types and shapes checked on the way in. Each call queues work on the device and returns; the
//! result is there for the next call that uses it (the queue is in order).

use crate::device::Tensor;
use crate::dtype::DType;
use crate::{Error, Result};

/// A linear layer with int8 weights (one layer of an `int8_convrot` checkpoint), on the device.
pub struct Int8Linear {
    /// int8 [N, K]
    pub weight: Tensor,
    /// float32, 1 or N values
    pub scale: Tensor,
    /// float32 [N]
    pub bias: Option<Tensor>,
    /// The rotation group size, when the weights were stored rotated.
    pub group: Option<usize>,
}

impl Int8Linear {
    pub fn inputs(&self) -> usize {
        self.weight.shape[1]
    }
    pub fn outputs(&self) -> usize {
        self.weight.shape[0]
    }

    /// `out = linear(x)`: x [M, K] in a float type, out [M, N] in `out`'s type.
    pub fn forward(&self, x: &Tensor, out: &Tensor) -> Result<()> {
        let (n, k) = (self.outputs(), self.inputs());
        let m = x.elements() / k;
        if self.weight.dtype != DType::I8 || self.scale.dtype != DType::F32 {
            return Err(Error("int8_linear: the weight must be int8 and its scale float32".into()));
        }
        if x.elements() != m * k || out.elements() != m * n {
            return Err(Error(format!("int8_linear: x {:?} and out {:?} do not fit a [{n}, {k}] weight", x.shape, out.shape)));
        }
        let ns = self.scale.elements();
        if ns != 1 && ns != n {
            return Err(Error(format!("int8_linear: {ns} weight scales for {n} outputs")));
        }
        if let Some(b) = &self.bias {
            if b.dtype != DType::F32 || b.elements() != n {
                return Err(Error("int8_linear: the bias must be float32 [N]".into()));
            }
        }
        let dev = x.buf.device();
        // SAFETY: every pointer is device memory of this device, with the sizes checked above.
        let rc = unsafe {
            (dev.api.int8_linear)(
                dev.ctx,
                x.buf.ptr(),
                x.dtype.kernel_code()?,
                m as i64,
                k as i64,
                self.weight.buf.ptr().cast(),
                n as i64,
                self.scale.buf.ptr().cast(),
                ns as i64,
                self.bias.as_ref().map_or(std::ptr::null(), |b| b.buf.ptr().cast()),
                out.buf.ptr(),
                out.dtype.kernel_code()?,
                self.group.unwrap_or(0) as i32,
            )
        };
        dev.check(rc)
    }
}
