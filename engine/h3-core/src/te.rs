//! The text encoder: Qwen3-VL 32B's language model cut to its first 50 layers (ComfyUI `text_encoders/minimax.py`),
//! from the llama.cpp-quantized file the reference uses (Q4_K / Q6_K weights). The conditioning is the hidden state
//! after layer 50 - no final norm - for every prompt token: [tokens, 5120].
//!
//! One layer, as the reference computes it (in IEEE half):
//!
//! ```text
//!   h = rmsnorm(x)                          q = h Wq [64 heads x 128], k = h Wk, v = h Wv [8 heads x 128]
//!   q, k: per-head rmsnorm + rotary positions (theta 5e6, all 128 features)
//!   x += attention(q, k, v) Wo              causal; 8 query heads share each key/value head
//!   h = rmsnorm(x)
//!   x += (silu(h Wgate) * (h Wup)) Wdown
//! ```
//!
//! 16.5 GB of weights for a step that runs once per prompt, beside a denoiser that holds 18 GB of the card: the
//! layers are streamed instead - read from the file by a thread one layer ahead, sent to the card, expanded to half
//! precision there (one layer's matrices at a time), used and dropped.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::mpsc;
use std::sync::Arc;

use crate::device::{Device, Tensor};
use crate::dtype::{bf16_to_f32, f32_to_f16, DType};
use crate::gguf::{GEntry, GType, Gguf};
use crate::ops::{self, Rows};
use crate::{Ctx, Error, Result};

const EPS: f32 = 1e-6;
const THETA: f32 = 5_000_000.0;
const HEAD_DIM: usize = 128;
const MATS: [&str; 7] = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"];
const NORMS: [&str; 4] = ["input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"];

pub struct TextEncoder {
    file: Gguf,
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub ffn: usize,
}

/// One layer as read from the file: the seven matrices' raw blocks and the four norms (float32).
struct HostLayer {
    mats: Vec<(GEntry, Vec<u8>)>,
    norms: Vec<Vec<f32>>,
}

/// Reads byte ranges of a file with `threads` readers in 32 MiB pieces.
fn read_ranges(path: &Path, ranges: &[(u64, usize)], threads: usize) -> Result<Vec<Vec<u8>>> {
    const PIECE: usize = 32 << 20;
    let mut bufs: Vec<Vec<u8>> = ranges.iter().map(|(_, n)| vec![0u8; *n]).collect();
    let mut jobs: Vec<(usize, usize, u64, usize)> = Vec::new(); // (buffer, offset in it, file offset, bytes)
    for (i, (off, n)) in ranges.iter().enumerate() {
        let mut o = 0;
        while o < *n {
            let k = PIECE.min(n - o);
            jobs.push((i, o, off + o as u64, k));
            o += k;
        }
    }
    // hand each reader disjoint pieces: split every buffer into its pieces up front
    let mut pieces: Vec<(&mut [u8], u64)> = Vec::new();
    let mut by_buf: Vec<Vec<(usize, u64, usize)>> = vec![Vec::new(); bufs.len()];
    for (b, o, fo, k) in jobs {
        by_buf[b].push((o, fo, k));
    }
    for (buf, js) in bufs.iter_mut().zip(&by_buf) {
        let mut rest: &mut [u8] = buf;
        for (_, fo, k) in js {
            let (head, tail) = rest.split_at_mut(*k);
            pieces.push((head, *fo));
            rest = tail;
        }
    }
    let n = pieces.len();
    let per = n.div_ceil(threads.max(1));
    let mut failed = None;
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        let mut it = pieces.into_iter();
        loop {
            let chunk: Vec<_> = it.by_ref().take(per.max(1)).collect();
            if chunk.is_empty() {
                break;
            }
            handles.push(s.spawn(move || -> Result<()> {
                let f = File::open(path)?;
                for (dst, fo) in chunk {
                    f.read_exact_at(dst, fo).ctx("reading the text encoder")?;
                }
                Ok(())
            }));
        }
        for h in handles {
            if let Err(e) = h.join().unwrap_or_else(|_| Err(Error("a reader panicked".into()))) {
                failed.get_or_insert(e);
            }
        }
    });
    match failed {
        Some(e) => Err(e),
        None => Ok(bufs),
    }
}

fn f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f32_to_f16(*f).to_le_bytes()).collect()
}

fn f32_tensor(dev: &Arc<Device>, v: &[f32]) -> Result<Tensor> {
    Tensor::from_bytes(dev, DType::F32, &[v.len()], &v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())
}

impl TextEncoder {
    pub fn open(path: &Path) -> Result<TextEncoder> {
        let file = Gguf::open(path)?;
        let layers = file.entries.keys().filter_map(|k| k.strip_prefix("model.layers.")?.split('.').next()?.parse::<usize>().ok()).max().map_or(0, |m| m + 1);
        let q = file.get("model.layers.0.self_attn.q_proj.weight")?.shape.clone(); // [heads * 128, hidden]
        let k = file.get("model.layers.0.self_attn.k_proj.weight")?.shape.clone();
        let ffn = file.get("model.layers.0.mlp.gate_proj.weight")?.shape[0];
        Ok(TextEncoder { layers, hidden: q[1], heads: q[0] / HEAD_DIM, kv_heads: k[0] / HEAD_DIM, ffn, file })
    }

    fn read_layer(&self, i: usize, threads: usize) -> Result<HostLayer> {
        let mats: Vec<GEntry> = MATS.iter().map(|m| self.file.get(&format!("model.layers.{i}.{m}.weight")).cloned()).collect::<Result<_>>()?;
        let norms: Vec<GEntry> = NORMS.iter().map(|m| self.file.get(&format!("model.layers.{i}.{m}.weight")).cloned()).collect::<Result<_>>()?;
        let ranges: Vec<(u64, usize)> = mats.iter().chain(&norms).map(|e| (e.offset, e.bytes)).collect();
        let mut bufs = read_ranges(&self.file.path, &ranges, threads)?;
        let nb = bufs.split_off(mats.len());
        let norms = nb
            .iter()
            .zip(&norms)
            .map(|(b, e)| match e.ty {
                GType::BF16 => Ok(b.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()),
                GType::F32 => Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
                t => Err(Error(format!("{}: a norm of type {t:?}", e.name))),
            })
            .collect::<Result<Vec<Vec<f32>>>>()?;
        Ok(HostLayer { mats: mats.into_iter().zip(bufs).collect(), norms })
    }

    /// The conditioning for `tokens`: [tokens, hidden] float32. `tick(layer)` before each layer.
    pub fn encode(&self, dev: &Arc<Device>, tokens: &[u32], threads: usize, tick: &mut dyn FnMut(usize) -> Result<()>) -> Result<Vec<f32>> {
        let l = tokens.len();
        let (c, hq, hkv, ffn) = (self.hidden, self.heads, self.kv_heads, self.ffn);
        let (wq, wkv) = (hq * HEAD_DIM, hkv * HEAD_DIM);
        // embeddings: the tokens' rows, read straight from the file
        let emb = self.file.get("model.embed_tokens.weight")?;
        if emb.ty != GType::BF16 || emb.shape[1] != c {
            return Err(Error("the token embedding is expected as bfloat16 [vocab, hidden]".into()));
        }
        let f = File::open(&self.file.path)?;
        let mut x0 = Vec::with_capacity(l * c);
        let mut row = vec![0u8; c * 2];
        for t in tokens {
            if *t as usize >= emb.shape[0] {
                return Err(Error(format!("token id {t} is outside the vocabulary")));
            }
            f.read_exact_at(&mut row, emb.offset + (*t as u64) * (c as u64 * 2)).ctx("reading the token embedding")?;
            x0.extend(row.chunks_exact(2).map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]]))));
        }
        let dt = DType::F16;
        let x = Tensor::from_bytes(dev, dt, &[l, c], &f16_bytes(&x0))?;
        // rotary table: position p, pair j -> angle p / theta^(2j / 128)
        let inv: Vec<f32> = (0..HEAD_DIM / 2).map(|j| 1.0 / THETA.powf((2 * j) as f32 / HEAD_DIM as f32)).collect();
        let cs: Vec<f32> = (0..l).flat_map(|p| inv.iter().flat_map(move |f| [(p as f32 * f).cos(), (p as f32 * f).sin()])).collect();
        let cs = Tensor::from_bytes(dev, DType::F32, &[l, HEAD_DIM / 2, 2], &cs.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;

        let h = Tensor::new(dev, dt, &[l, c])?;
        let q = Tensor::new(dev, dt, &[l, wq])?;
        let k = Tensor::new(dev, dt, &[l, wkv])?;
        let v = Tensor::new(dev, dt, &[l, wkv])?;
        let att = Tensor::new(dev, dt, &[l, wq])?;
        let o = Tensor::new(dev, dt, &[l, c])?;
        let gu = Tensor::new(dev, dt, &[l, 2 * ffn])?;
        let act = Tensor::new(dev, dt, &[l, ffn])?;
        // one layer's largest matrix (gate and up together), expanded to half precision
        let w = Tensor::new(dev, dt, &[2 * ffn * c])?;

        let d = dev.clone();
        let lin = |input: &Tensor, n: usize, kk: usize, out: &Tensor| -> Result<()> {
            // SAFETY: device buffers of this device; input [l, kk], w holds at least [n, kk], out [l, n].
            let rc = unsafe { (d.api.linear)(d.ctx, input.buf.ptr(), dt.kernel_code()?, l as i64, kk as i64, w.buf.ptr(), n as i64, std::ptr::null(), out.buf.ptr(), dt.kernel_code()?) };
            d.check(rc)
        };
        let expand = |blob: &Tensor, e: &GEntry, at: usize| -> Result<()> {
            let code = e.ty.quant_code().ok_or_else(|| Error(format!("{}: type {:?} is not a k-quant", e.name, e.ty)))?;
            // SAFETY: blob holds e.bytes of k-quant blocks; w has room for `at + elements` half values.
            let rc = unsafe {
                (d.api.dequant)(d.ctx, blob.buf.ptr(), code, e.elements() as i64, w.buf.ptr().cast::<u8>().add(at * 2).cast(), dt.kernel_code()?)
            };
            d.check(rc)
        };

        // the reader thread, one layer ahead
        let (tx, rx) = mpsc::sync_channel::<Result<HostLayer>>(1);
        std::thread::scope(|s| -> Result<Vec<f32>> {
            s.spawn(|| {
                for i in 0..self.layers {
                    if tx.send(self.read_layer(i, threads)).is_err() {
                        return; // the consumer stopped
                    }
                }
            });
            for i in 0..self.layers {
                tick(i)?;
                let hl = rx.recv().map_err(|_| Error("the text encoder's reader stopped".into()))??;
                let blobs: Vec<Tensor> = hl.mats.iter().map(|(e, b)| Tensor::from_bytes(dev, DType::U8, &[e.bytes], b)).collect::<Result<_>>()?;
                let [ln1, ln2, qn, kn] = [0, 1, 2, 3].map(|j| f32_tensor(dev, &hl.norms[j]));
                let (ln1, ln2, qn, kn) = (ln1?, ln2?, qn?, kn?);
                let e = |j: usize| &hl.mats[j].0;
                // attention half
                ops::rms_norm_mod(&x, &ln1, EPS, None, &h)?;
                expand(&blobs[0], e(0), 0)?;
                lin(&h, wq, c, &q)?;
                expand(&blobs[1], e(1), 0)?;
                lin(&h, wkv, c, &k)?;
                expand(&blobs[2], e(2), 0)?;
                lin(&h, wkv, c, &v)?;
                ops::rms_rope(Rows { t: &q, offset: 0, stride: wq, tokens: l, heads: hq, dim: HEAD_DIM }, &qn, EPS, &cs, HEAD_DIM)?;
                ops::rms_rope(Rows { t: &k, offset: 0, stride: wkv, tokens: l, heads: hkv, dim: HEAD_DIM }, &kn, EPS, &cs, HEAD_DIM)?;
                // SAFETY: q [l, wq], k and v [l, wkv], att [l, wq], all half on this device.
                let rc = unsafe {
                    (d.api.attention_causal)(d.ctx, q.buf.ptr(), k.buf.ptr(), v.buf.ptr(), dt.kernel_code()?, l as i64, hq as i64, hkv as i64, HEAD_DIM as i64,
                                             wq as i64, wkv as i64, att.buf.ptr())
                };
                d.check(rc)?;
                expand(&blobs[3], e(3), 0)?;
                lin(&att, c, wq, &o)?;
                ops::add(&x, &o)?;
                // MLP half: gate and up expanded one after the other, multiplied at once ([gate | up] per row)
                ops::rms_norm_mod(&x, &ln2, EPS, None, &h)?;
                expand(&blobs[4], e(4), 0)?;
                expand(&blobs[5], e(5), ffn * c)?;
                lin(&h, 2 * ffn, c, &gu)?;
                ops::swiglu(&gu, &act)?;
                expand(&blobs[6], e(6), 0)?;
                lin(&act, c, ffn, &o)?;
                ops::add(&x, &o)?;
                dev.wait()?; // the layer's blobs go next
            }
            drop(rx);
            x.to_f32()
        })
    }
}
