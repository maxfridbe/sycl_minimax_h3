//! The audio decoder: latents [32, 2, T] (40 frames a second) -> stereo sound at 32 kHz, 800 samples per frame.
//!
//! BigVGAN, the two stereo channels as a batch of two mono signals: a 1x1 projection to 2048 channels, a 7-tap
//! convolution to 1024, then seven upsampling stages (x5, x5, x2 ... x2; the channels halve each time), each a
//! transposed convolution followed by the average of three residual stacks (kernels 3, 7, 11, dilations 1, 3, 5)
//! whose activations are anti-aliased Snake functions; a last activation, a 7-tap convolution to one channel, and a
//! clamp to [-1, 1]. Float32 throughout, as the checkpoint is stored (ComfyUI `ldm/minimax/audio_vae.py`).

use std::sync::Arc;

use crate::device::{Device, Tensor};
use crate::dit::host;
use crate::dtype::DType;
use crate::safetensors::Checkpoint;
use crate::{Error, Result};

const UP_RATES: [usize; 7] = [5, 5, 2, 2, 2, 2, 2];
const UP_KERNELS: [usize; 7] = [9, 9, 4, 4, 4, 4, 4];
const RES_KERNELS: [usize; 3] = [3, 7, 11];
const DILATIONS: [usize; 3] = [1, 3, 5];
pub const SAMPLE_RATE: u32 = 32_000;

fn upload(dev: &Arc<Device>, ck: &Checkpoint, name: &str) -> Result<Tensor> {
    let v = host(ck, name)?;
    let e = ck.get(name)?;
    Tensor::from_bytes(dev, DType::F32, &e.shape, &v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())
}

struct Conv {
    w: Tensor,
    b: Option<Tensor>,
    /// [Co, Ci, K] for a convolution, [Ci, Co, K] for a transposed one
    co: usize,
    ci: usize,
    k: usize,
}

impl Conv {
    fn load(dev: &Arc<Device>, ck: &Checkpoint, p: &str, transposed: bool) -> Result<Conv> {
        let w = upload(dev, ck, &format!("{p}.weight"))?;
        let b = match ck.entries.contains_key(&format!("{p}.bias")) {
            true => Some(upload(dev, ck, &format!("{p}.bias"))?),
            false => None,
        };
        let (a, c, k) = (w.shape[0], w.shape[1], w.shape[2]);
        let (co, ci) = if transposed { (c, a) } else { (a, c) };
        Ok(Conv { w, b, co, ci, k })
    }
}

/// One anti-aliased Snake activation: its per-channel (log) alpha and beta, its two 12-tap filters.
struct Act {
    alpha: Tensor,
    beta: Tensor,
    up: Tensor,
    down: Tensor,
}

impl Act {
    fn load(dev: &Arc<Device>, ck: &Checkpoint, p: &str) -> Result<Act> {
        Ok(Act {
            alpha: upload(dev, ck, &format!("{p}.act.alpha"))?,
            beta: upload(dev, ck, &format!("{p}.act.beta"))?,
            up: upload(dev, ck, &format!("{p}.upsample.filter"))?,
            down: upload(dev, ck, &format!("{p}.downsample.lowpass.filter"))?,
        })
    }
}

struct ResStack {
    k: usize,
    convs1: Vec<Conv>,
    convs2: Vec<Conv>,
    acts: Vec<Act>,
}

pub struct AudioDecoder {
    dev: Arc<Device>,
    in_proj: Conv,
    conv_pre: Conv,
    ups: Vec<Conv>,
    res: Vec<ResStack>,
    act_post: Act,
    conv_post: Conv,
    mean: Vec<f32>,
    std: Vec<f32>,
}

/// A signal [B, C, L] on the device, float32.
struct Sig {
    t: Tensor,
    b: usize,
    c: usize,
    l: usize,
}

impl AudioDecoder {
    pub fn load(dev: &Arc<Device>, ck: &Checkpoint) -> Result<AudioDecoder> {
        let mut res = Vec::new();
        for i in 0..UP_RATES.len() * RES_KERNELS.len() {
            let p = format!("decoder.resblocks.{i}");
            res.push(ResStack {
                k: RES_KERNELS[i % RES_KERNELS.len()],
                convs1: (0..3).map(|j| Conv::load(dev, ck, &format!("{p}.convs1.{j}"), false)).collect::<Result<_>>()?,
                convs2: (0..3).map(|j| Conv::load(dev, ck, &format!("{p}.convs2.{j}"), false)).collect::<Result<_>>()?,
                acts: (0..6).map(|j| Act::load(dev, ck, &format!("{p}.activations.{j}"))).collect::<Result<_>>()?,
            });
        }
        let ups: Vec<Conv> = (0..UP_RATES.len()).map(|i| Conv::load(dev, ck, &format!("decoder.ups.{i}.0"), true)).collect::<Result<_>>()?;
        if ups.iter().map(|u| u.k).ne(UP_KERNELS) {
            return Err(Error("the audio decoder's upsampling kernels are not BigVGAN's 32 kHz configuration".into()));
        }
        Ok(AudioDecoder {
            dev: dev.clone(),
            in_proj: Conv::load(dev, ck, "dec_in_proj", false)?,
            conv_pre: Conv::load(dev, ck, "decoder.conv_pre", false)?,
            ups,
            res,
            act_post: Act::load(dev, ck, "decoder.activation_post")?,
            conv_post: Conv::load(dev, ck, "decoder.conv_post", false)?,
            mean: host(ck, "latents_mean")?,
            std: host(ck, "latents_std")?,
        })
    }

    fn new_sig(&self, b: usize, c: usize, l: usize) -> Result<Sig> {
        Ok(Sig { t: Tensor::new(&self.dev, DType::F32, &[b, c, l])?, b, c, l })
    }

    fn conv(&self, x: &Sig, cv: &Conv, dil: usize, pad: usize) -> Result<Sig> {
        if x.c != cv.ci {
            return Err(Error(format!("audio conv: {} channels into a {}-channel convolution", x.c, cv.ci)));
        }
        let lo = x.l + 2 * pad - dil * (cv.k - 1);
        let y = self.new_sig(x.b, cv.co, lo)?;
        let d = &self.dev;
        // SAFETY: float32 device buffers of this device, sized as declared above.
        let rc = unsafe {
            (d.api.conv1d)(d.ctx, x.t.buf.ptr().cast(), x.b as i64, x.c as i64, x.l as i64, cv.w.buf.ptr().cast(), cv.co as i64, cv.k as i64,
                           cv.b.as_ref().map_or(std::ptr::null(), |b| b.buf.ptr().cast_const().cast()), 1, dil as i64, pad as i64, y.t.buf.ptr().cast(), lo as i64)
        };
        d.check(rc)?;
        Ok(y)
    }

    fn conv_t(&self, x: &Sig, cv: &Conv, stride: usize) -> Result<Sig> {
        let pad = (cv.k - stride) / 2;
        let lo = (x.l - 1) * stride + cv.k - 2 * pad;
        let y = self.new_sig(x.b, cv.co, lo)?;
        let d = &self.dev;
        // SAFETY: as in `conv`.
        let rc = unsafe {
            (d.api.conv_transpose1d)(d.ctx, x.t.buf.ptr().cast(), x.b as i64, x.c as i64, x.l as i64, cv.w.buf.ptr().cast(), cv.co as i64, cv.k as i64,
                                     cv.b.as_ref().map_or(std::ptr::null(), |b| b.buf.ptr().cast_const().cast()), stride as i64, pad as i64, y.t.buf.ptr().cast(), lo as i64)
        };
        d.check(rc)?;
        Ok(y)
    }

    fn act(&self, x: &Sig, a: &Act) -> Result<Sig> {
        let y = self.new_sig(x.b, x.c, x.l)?;
        let d = &self.dev;
        // SAFETY: as in `conv`; alpha and beta have x.c values, the filters 12.
        let rc = unsafe {
            (d.api.aa_snake)(d.ctx, x.t.buf.ptr().cast(), x.b as i64, x.c as i64, x.l as i64, a.alpha.buf.ptr().cast(), a.beta.buf.ptr().cast(),
                             a.up.buf.ptr().cast(), a.down.buf.ptr().cast(), y.t.buf.ptr().cast())
        };
        d.check(rc)?;
        Ok(y)
    }

    /// One residual stack (BigVGAN's AMPBlock1) on x; x itself is left as it is.
    fn stack(&self, x: &Sig, r: &ResStack) -> Result<Sig> {
        let mut cur = self.new_sig(x.b, x.c, x.l)?;
        cur.t.copy_rows(0, &x.t, 0, x.b)?;
        for (j, d) in DILATIONS.iter().enumerate() {
            let t = self.act(&cur, &r.acts[2 * j])?;
            let t = self.conv(&t, &r.convs1[j], *d, (r.k * d - d) / 2)?;
            let t = self.act(&t, &r.acts[2 * j + 1])?;
            let t = self.conv(&t, &r.convs2[j], 1, (r.k - 1) / 2)?;
            crate::ops::add(&t.t, &cur.t)?;
            cur = t;
        }
        Ok(cur)
    }

    /// Normalized latents [32, 2, T] -> one buffer of samples per stereo channel (800 T each), in [-1, 1].
    pub fn decode(&self, z: &[f32], t: usize, tick: &mut dyn FnMut() -> Result<()>) -> Result<Vec<Vec<f32>>> {
        let c = self.mean.len();
        if z.len() != c * 2 * t {
            return Err(Error(format!("audio latents: {} values for {c} x 2 x {t}", z.len())));
        }
        // [32, 2, T] -> [2, 32, T], denormalized
        let mut zz = vec![0f32; z.len()];
        for s in 0..2 {
            for ci in 0..c {
                for ti in 0..t {
                    zz[(s * c + ci) * t + ti] = z[(ci * 2 + s) * t + ti] * self.std[ci] + self.mean[ci];
                }
            }
        }
        let x = Sig { t: Tensor::from_bytes(&self.dev, DType::F32, &[2, c, t], &zz.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())?, b: 2, c, l: t };
        let x = self.conv(&x, &self.in_proj, 1, 0)?;
        let mut x = self.conv(&x, &self.conv_pre, 1, 3)?;
        let n = RES_KERNELS.len();
        for (i, up) in self.ups.iter().enumerate() {
            tick()?;
            x = self.conv_t(&x, up, UP_RATES[i])?;
            let xs = self.stack(&x, &self.res[i * n])?;
            for j in 1..n {
                let y = self.stack(&x, &self.res[i * n + j])?;
                crate::ops::add(&xs.t, &y.t)?;
            }
            // SAFETY: float32 device buffer of xs.t's size.
            let rc = unsafe { (self.dev.api.scale)(self.dev.ctx, xs.t.buf.ptr().cast(), xs.t.elements() as i64, 1.0 / n as f32) };
            self.dev.check(rc)?;
            x = xs;
        }
        let x = self.act(&x, &self.act_post)?;
        let y = self.conv(&x, &self.conv_post, 1, 3)?;
        let v = y.t.to_f32()?;
        let l = y.l;
        Ok((0..2).map(|s| v[s * l..(s + 1) * l].iter().map(|x| x.clamp(-1.0, 1.0)).collect()).collect())
    }
}
