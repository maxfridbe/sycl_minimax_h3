/* h3sycl.h - the C ABI of libh3sycl: SYCL kernels for MiniMax H3 on Intel Arc (Xe2).
 *
 * The Rust engine (engine/h3-sys) binds exactly this file. Every data pointer is a USM device pointer of the
 * context's queue. Kernels are asynchronous: a call returns once its work is queued, and the queue is in order, so
 * a later call sees an earlier call's result; h3s_read and h3s_wait are the points that wait.
 *
 * Functions returning int give 0, or -1 with the reason in h3s_last_error() (per thread). Functions returning a
 * pointer give NULL on failure, same.
 */
#ifndef H3SYCL_H
#define H3SYCL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* element types of floating-point tensors */
enum { H3S_F32 = 0, H3S_F16 = 1, H3S_BF16 = 2 };

const char* h3s_last_error(void);

/* ---- context -------------------------------------------------------------------------------------------------- */

/* A context on the first GPU, with its own in-order queue. The Rust engine's way in. */
void* h3s_open(void);
/* The GPUs the runtime sees (ONEAPI_DEVICE_SELECTOR=level_zero:* for every card), and a context on one of them.
 * h3s_gpu_info fills the name, the memory size and the PCI address ("0000:0b:00.0", empty when unknown). */
int h3s_gpu_count(void);
int h3s_gpu_info(int index, char* name, int name_len, uint64_t* mem_bytes, char* pci, int pci_len);
void* h3s_open_gpu(int index);
/* A context around an existing `sycl::queue*` (PyTorch: torch.xpu.current_stream().sycl_queue); the queue handle is
 * copied. The reference pipeline's way in: its tensors' device pointers are then valid here. */
void* h3s_create(void* sycl_queue);
/* Waits for the queue, frees the scratch buffers and everything still allocated with h3s_alloc. */
void h3s_destroy(void* ctx);

const char* h3s_device_name(void* ctx);

/* ---- device memory -------------------------------------------------------------------------------------------- */

/* The xe driver has no out-of-memory error: an over-commit stalls the whole machine. So allocations are counted, and
 * one that would pass the cap (0.94 of the card; H3S_MEM_FRACTION overrides) is refused here, with an error. The
 * kernels' own scratch buffers count against the same cap: a kernel call that needs more fails the same way. */
void* h3s_alloc(void* ctx, uint64_t bytes);
void h3s_free(void* ctx, void* p);              /* waits for the queue first */
uint64_t h3s_mem_used(void* ctx);
uint64_t h3s_mem_cap(void* ctx);
/* What the whole card has free right now, every process counted (0 when the driver cannot tell: it needs
 * ZES_ENABLE_SYSMAN=1, which the image sets). For deciding whether to load beside whatever else holds the card. */
uint64_t h3s_mem_free(void* ctx);

/* Host <-> device copies; both wait for the copy (the host buffer may be released on return). Thread-safe. */
int h3s_write(void* ctx, void* dst, const void* src_host, uint64_t bytes);
int h3s_read(void* ctx, void* dst_host, const void* src, uint64_t bytes);
/* Device to device, queued like a kernel (it does not wait). */
int h3s_copy(void* ctx, void* dst, const void* src, uint64_t bytes);
/* Waits until everything queued has run; reports an asynchronous error if there was one. */
int h3s_wait(void* ctx);

/* ---- kernels -------------------------------------------------------------------------------------------------- */

/* A linear layer with int8 weights and activations quantized on the fly (comfy-kitchen's int8_linear):
 *
 *   x_rot = x . H           per group of `group` features, H the normalized regular Hadamard matrix ("ConvRot")
 *   s_r   = max|x_rot[r]| / 127                          per row, at least 1e-30
 *   q     = clamp(round(x_rot / s_r), -128, 127)         int8
 *   acc   = q . W^T                                      int32 - the card's matrix engine
 *   out   = acc * s_r * wscale + bias                    in out_dt
 *
 * x [M, K] in x_dt; w int8 [N, K]; wscale float32, n_wscale = 1 (one scale) or N (one per output); bias float32 [N]
 * or NULL; out [M, N] in out_dt. group = 0: no rotation, else a power of 4, at most 256, dividing K. */
int h3s_int8_linear(void* ctx, const void* x, int x_dt, int64_t M, int64_t K, const int8_t* w, int64_t N,
                    const float* wscale, int64_t n_wscale, const float* bias, void* out, int out_dt, int group);

/* A plain linear layer: out = x . w^T + bias. x [M, K] and w [N, K] in the same type dt (float32, half or
 * bfloat16); bias float32 [N] or NULL; out [M, N] in out_dt. */
int h3s_linear(void* ctx, const void* x, int dt, int64_t M, int64_t K, const void* w, int64_t N, const float* bias,
               void* out, int out_dt);

/* Row-wise RMS norm, optionally followed by a per-row scale and shift picked from a table:
 *
 *   n   = x[r] / sqrt(mean(x[r]^2) + eps) * weight
 *   out = n * (1 + scale[rows[r]]) + shift[rows[r]]        (or out = n when rows, scale or shift is NULL)
 *
 * x, out [M, C]; weight float32 [C]; rows int32 [M]; scale, shift float32 [R, C]. out may be x. */
int h3s_rms_norm_mod(void* ctx, const void* x, int x_dt, int64_t M, int64_t C, const float* weight, float eps,
                     const int32_t* rows, const float* scale, const float* shift, void* out, int out_dt);

/* Per-head RMS norm, then the rotary position rotation, in place. x holds M token rows of H heads x D features;
 * row m starts at element m * stride (stride >= H * D: the rows may sit inside a wider buffer, as q and k do inside
 * the qkv linear's output). Each head's D features are normalized (weight float32 [D]); then feature pairs
 * (i, rot_dim/2 + i), i < rot_dim/2, are rotated by the token's angle:
 *
 *   a' = a * cos - b * sin,   b' = b * cos + a * sin
 *
 * cs float32 [M, rot_dim/2, 2] holds (cos, sin) per token and pair. Features from rot_dim on are only normalized. */
int h3s_rms_rope(void* ctx, void* x, int x_dt, int64_t M, int64_t H, int64_t D, int64_t stride, const float* weight,
                 float eps, const float* cs, int rot_dim);

/* The gated activation between an MLP's two linears: out[r, i] = silu(x[r, i]) * x[r, C + i].
 * x [M, 2 * C], out [M, C]. */
int h3s_swiglu(void* ctx, const void* x, int x_dt, int64_t M, int64_t C, void* out, int out_dt);

/* x[r] += other[r] * gate[rows[r]], in place. x, other [M, C]; rows int32 [M]; gate float32 [R, C].
 * gate NULL: a plain add. */
int h3s_gate_add(void* ctx, void* x, int x_dt, int64_t M, int64_t C, const void* other, int other_dt,
                 const int32_t* rows, const float* gate);

/* Attention: out = softmax(q . k^T / sqrt(D)) . v, per head. q, k, v: S token rows of H heads x D features, row s at
 * element s * stride (as in h3s_rms_rope), in type dt; out [S, H * D] in out_dt. The S x S scores are never
 * held whole: the work is done a block of query rows at a time, sized so that a block's score table is at most
 * 1.5 GiB (H3S_ATTN_TABLE_MB in the environment overrides; with oneDNN's fused kernel no table is written at all, and
 * the bound is what is at stake if oneDNN silently falls back - see h3sycl.cpp). */
int h3s_attention(void* ctx, const void* q, const void* k, const void* v, int dt, int64_t S, int64_t H, int64_t D,
                  int64_t stride, void* out, int out_dt);

#ifdef __cplusplus
}
#endif
#endif
