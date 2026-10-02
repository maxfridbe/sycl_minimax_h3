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
/* A context around an existing `sycl::queue*` (PyTorch: torch.xpu.current_stream().sycl_queue); the queue handle is
 * copied. The reference pipeline's way in: its tensors' device pointers are then valid here. */
void* h3s_create(void* sycl_queue);
/* Waits for the queue, frees the scratch buffers and everything still allocated with h3s_alloc. */
void h3s_destroy(void* ctx);

const char* h3s_device_name(void* ctx);

/* ---- device memory -------------------------------------------------------------------------------------------- */

/* The xe driver has no out-of-memory error: an over-commit stalls the whole machine. So allocations are counted, and
 * one that would pass the cap (0.92 of the card; H3S_MEM_FRACTION overrides) is refused here, with an error. */
void* h3s_alloc(void* ctx, uint64_t bytes);
void h3s_free(void* ctx, void* p);              /* waits for the queue first */
uint64_t h3s_mem_used(void* ctx);
uint64_t h3s_mem_cap(void* ctx);

/* Host <-> device copies; both wait for the copy (the host buffer may be released on return). Thread-safe. */
int h3s_write(void* ctx, void* dst, const void* src_host, uint64_t bytes);
int h3s_read(void* ctx, void* dst_host, const void* src, uint64_t bytes);
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

#ifdef __cplusplus
}
#endif
#endif
