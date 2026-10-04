// libh3sage.so - SageAttention (v1) for libh3sycl: the attention kernel of Intel's ARK (intel/auto-round,
// auto_round_extension/ark, Apache 2.0 - the kernel SageAttention's Intel GPU path calls), on sycl-tla.
//
// Optional: libh3sycl loads this library only when H3S_ATTN=sage asks for it, and attention stays on oneDNN's fused
// kernel when the library is absent or refuses. It is its own library because sycl-tla wants its own compiler flags
// (C++17, SPIR-V extensions for 2-D block loads and the matrix engine, the device named at link time) and a long
// compile that the rest of the kernels should not pay.
//
// What it computes, per head: softmax(q k^T * scale) v, with q and k already int8 (libh3sycl quantizes them, one
// scale per head per `block` rows, k after subtracting its mean over the sequence - which leaves the softmax
// unchanged), v and the result IEEE half. All four [H, S, D], packed. Queued on the caller's queue; no wait.

#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>

#include <sycl/sycl.hpp>
#include "sycl_tla_sdpa.hpp"

namespace {
const sycl::queue* g_default = nullptr;    // the queue last made sycl-tla's default (setting it waits on the old one)
std::string g_err;
}

extern "C" {

const char* h3sage_error() { return g_err.c_str(); }

// 0 on success, else -1 with the reason in h3sage_error(). D must be 64 or 128; block a multiple of 64.
int h3sage_attention(void* queue, const int8_t* q, const int8_t* k, const void* v, void* out, const float* qscale,
                     const float* kscale, int block, int64_t S, int64_t H, int64_t D, float scale) try {
    if (D != 64 && D != 128) { g_err = "h3sage: head size " + std::to_string(D) + " (64 or 128 only)"; return -1; }
    if (block <= 0 || block % 64 != 0) { g_err = "h3sage: the scale block must be a multiple of 64"; return -1; }
    if (S <= 0 || H <= 0 || S > INT32_MAX / D / H) { g_err = "h3sage: sequence out of range"; return -1; }
    auto* qu = static_cast<sycl::queue*>(queue);
    if (g_default != qu) {
        compat::set_default_queue(*qu);
        g_default = qu;
    }
    ark::detail::Options o;
    o.q = q; o.k = k; o.v = v; o.o = out;
    o.scale_block_size = block;
    o.qscale = qscale; o.kscale = kscale;
    o.batch = 1;
    o.num_heads_q = o.num_heads_kv = (int) H;
    o.seq_len_qo = o.seq_len_kv = (int) S;
    o.head_size_qk = o.head_size_vo = (int) D;
    o.softmax_scale = scale;
    o.is_causal = false;
    const int rc = D == 128 ? ark::detail::launch_sage_prefill_kernel_128<cute::int8_t, cute::int8_t, cute::half_t>(o)
                            : ark::detail::launch_sage_prefill_kernel_64<cute::int8_t, cute::int8_t, cute::half_t>(o);
    if (rc != 0) { g_err = "h3sage: the kernel refused the problem"; return -1; }
    return 0;
} catch (const std::exception& e) {
    g_err = std::string("h3sage: ") + e.what();
    return -1;
}

}  // extern "C"
