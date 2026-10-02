// h3sycl.cpp - SYCL kernels for MiniMax H3 on Intel Arc (Xe2). The C ABI is h3sycl.h; two hosts use it:
//
//   the Rust engine (engine/)        h3s_open(): the library owns the queue and the device memory (h3s_alloc ...)
//   the PyTorch reference pipeline   h3s_create(queue): PyTorch's own in-order queue and its tensors' device pointers,
//                                    so calls are ordered with PyTorch's operations and nothing is copied
//
// Every data pointer is a USM device pointer of the context's queue.
//
// int8_linear, as comfy-kitchen's eager backend defines it:
//   x_rot = x . H           per group of `group` features, H the normalized regular Hadamard matrix (ConvRot)
//   s_r   = max|x_rot[r]| / 127                                  (per row, at least 1e-30)
//   q     = clamp(round(x_rot / s_r), -128, 127)                 int8
//   acc   = q . W^T                                              int32, W int8 [N, K]   <- the matrix engine, oneDNN
//   y     = cast(acc * (s_r * w_scale)) + bias                   in the output type
#include "h3sycl.h"

#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {

enum Dt : int { F32 = 0, F16 = 1, BF16 = 2 };

thread_local std::string g_err;

// bf16 <-> f32 on bits (SYCL has no bfloat16 type): widen = shift; narrow = round to nearest even
inline float bf16_to_f32(uint16_t b) { return sycl::bit_cast<float>((uint32_t) b << 16); }
inline uint16_t f32_to_bf16(float f) {
    uint32_t u = sycl::bit_cast<uint32_t>(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t) (u >> 16);          // inf / nan: truncate
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
inline float load(const void* p, int dt, size_t i) {
    switch (dt) {
        case F16: return (float) ((const sycl::half*) p)[i];
        case BF16: return bf16_to_f32(((const uint16_t*) p)[i]);
        default: return ((const float*) p)[i];
    }
}
inline void store(void* p, int dt, size_t i, float v) {
    switch (dt) {
        case F16: ((sycl::half*) p)[i] = (sycl::half) v; break;
        case BF16: ((uint16_t*) p)[i] = f32_to_bf16(v); break;
        default: ((float*) p)[i] = v;
    }
}
constexpr int kMaxGroup = 256;

// exp(x) for x <= 0, to about 2e-5 relative (the result is stored in a 16-bit float): 2^(x log2 e) with the
// fraction's power from a 6th-order polynomial and the integer part put straight into the exponent bits. The
// library's exp is several times slower, and attention needs one per query-key pair.
inline float exp_neg(float x) {
    const float t = sycl::fmax(x * 1.44269504f, -126.0f);
    const float n = sycl::floor(t);
    const float f = t - n;
    const float p = 1.0f + f * (0.693147182f + f * (0.240226507f + f * (0.0555041087f + f * (0.00961812911f + f * (0.00133335581f + f * 0.000154035304f)))));
    return p * sycl::bit_cast<float>((uint32_t) ((int32_t) n + 127) << 23);
}

struct Ctx {
    sycl::queue q;
    dnnl::engine eng;
    dnnl::stream strm;
    int8_t* xq = nullptr; size_t xq_cap = 0;       // quantized activations [M, K]
    float* rs = nullptr; size_t rs_cap = 0;        // per-row scale times the scalar weight scale [M]
    float* gmax = nullptr; size_t gmax_cap = 0;    // per-(row, group) absmax
    float* xr = nullptr; size_t xr_cap = 0;        // rotated activations [M, K], float32 (see pass 1)
    int32_t* acc = nullptr; size_t acc_cap = 0;    // fallback path: one chunk of int32 products [C, N]
    float* inv = nullptr; size_t inv_cap = 0;      // 1 / rms per row (norms)
    // attention, all in IEEE half: a chunk of q by head [H, rows, D]; k transposed [H, D, S]; v by head [H, S, D];
    // one chunk of scores [H, rows, S]; one chunk of results [H, rows, D]
    uint16_t* hq = nullptr; size_t hq_cap = 0;
    uint16_t* hk = nullptr; size_t hk_cap = 0;
    uint16_t* hv = nullptr; size_t hv_cap = 0;
    uint16_t* sc = nullptr; size_t sc_cap = 0;
    float* ao = nullptr; size_t ao_cap = 0;        // float32: the sums are over un-normalized weights
    float* part = nullptr; size_t part_cap = 0;    // per (row, lane): the max of that lane's share of the row
    float* rowv = nullptr; size_t rowv_cap = 0;    // per row: its max
    bool profile = false;                          // H3S_PROFILE: attention waits per phase and reports its timings
    struct Attn { dnnl::matmul qk; dnnl::matmul pv; };
    std::map<std::tuple<int64_t, int64_t, int64_t, int64_t>, Attn> attn;                   // (H, rows, S, D)
    void* had[3] = {nullptr, nullptr, nullptr};    // the normalized 256 x 256 Hadamard matrix per Dt
    std::map<std::tuple<int64_t, int64_t, int64_t, int, int, int>, dnnl::matmul> gemm;   // (M, K, N, out, bias, per_n)
    std::map<std::tuple<int64_t, int, int>, dnnl::matmul> rot;                            // (rows, group, dt)
    bool fused_ok = true;                          // oneDNN took the post-op form (else: int32 + our rescale kernel)
    bool rot_f32 = true;                           // oneDNN took a float32 result for the 16-bit rotation
    // h3s_alloc's book: the xe driver has no out-of-memory error (an over-commit stalls the whole machine), so the
    // library refuses an allocation that would pass the cap instead of asking the driver
    std::mutex mem_mu;
    std::unordered_map<void*, size_t> mem;
    size_t mem_used = 0, mem_cap = 0;
    std::string name;

    // A scratch buffer of at least n elements; counted against the same cap as h3s_alloc. NULL (with the reason in
    // g_err) when it would pass the cap or the device refuses.
    template <typename T> T* grow(T*& p, size_t& cap, size_t n) {
        if (n <= cap) return p;
        const size_t want = n + (n >> 3);
        std::lock_guard<std::mutex> l(mem_mu);
        if (mem_used - cap * sizeof(T) + want * sizeof(T) > mem_cap) {
            g_err = "a scratch buffer of " + std::to_string((want * sizeof(T)) >> 20) + " MiB would pass the device memory cap (" +
                    std::to_string(mem_used >> 20) + " of " + std::to_string(mem_cap >> 20) + " MiB in use)";
            return nullptr;
        }
        if (p) { q.wait(); sycl::free(p, q); }
        mem_used -= cap * sizeof(T);
        p = sycl::malloc_device<T>(want, q);
        cap = p ? want : 0;
        mem_used += cap * sizeof(T);
        if (!p) g_err = "the device refused a scratch buffer of " + std::to_string((want * sizeof(T)) >> 20) + " MiB";
        return p;
    }
};

inline dnnl::memory::data_type ddt(int dt) {
    return dt == F16 ? dnnl::memory::data_type::f16 : dt == BF16 ? dnnl::memory::data_type::bf16
                                                                 : dnnl::memory::data_type::f32;
}
inline dnnl::memory usm(const dnnl::memory::desc& md, const dnnl::engine& eng, const void* p) {
    return dnnl::sycl_interop::make_memory(md, eng, dnnl::sycl_interop::memory_kind::usm, const_cast<void*>(p));
}

// the normalized regular Hadamard matrix of size g (h4 (x) h4 (x) ...), on the device in type dt
void* hadamard_matrix(Ctx& c, int g, int dt) {
    if (c.had[dt]) return c.had[dt];
    std::vector<float> h((size_t) g * g);
    static const int h4[4][4] = {{1, 1, 1, -1}, {1, 1, -1, 1}, {1, -1, 1, 1}, {-1, 1, 1, 1}};
    float norm = 1.0f;
    for (int s = 1; s < g; s *= 4) norm *= 0.5f;
    for (int i = 0; i < g; ++i)
        for (int j = 0; j < g; ++j) {
            int sign = 1;
            for (int a = i, b = j, s = 1; s < g; s *= 4, a /= 4, b /= 4) sign *= h4[a % 4][b % 4];
            h[(size_t) i * g + j] = (float) sign * norm;
        }
    const size_t n = (size_t) g * g;
    if (dt == F32) {
        float* d = sycl::malloc_device<float>(n, c.q);
        c.q.memcpy(d, h.data(), n * 4).wait();
        c.had[dt] = d;
    } else {
        std::vector<uint16_t> hb(n);
        for (size_t i = 0; i < n; ++i) {
            if (dt == BF16) hb[i] = f32_to_bf16(h[i]);
            else { sycl::half v = (sycl::half) h[i]; std::memcpy(&hb[i], &v, 2); }
        }
        uint16_t* d = sycl::malloc_device<uint16_t>(n, c.q);
        c.q.memcpy(d, hb.data(), n * 2).wait();
        c.had[dt] = d;
    }
    return c.had[dt];
}

// what both ways of making a context share: the oneDNN engine on the queue, the device's name, the memory cap
void init(Ctx& c) {
    c.eng = dnnl::sycl_interop::make_engine(c.q.get_device(), c.q.get_context());
    c.strm = dnnl::sycl_interop::make_stream(c.eng, c.q);
    c.name = c.q.get_device().get_info<sycl::info::device::name>();
    const uint64_t total = c.q.get_device().get_info<sycl::info::device::global_mem_size>();
    double frac = 0.94;                            // of the card, for h3s_alloc and the kernels' scratch together
    if (const char* e = std::getenv("H3S_MEM_FRACTION")) frac = std::min(0.97, std::max(0.05, std::atof(e)));
    c.mem_cap = (size_t) ((double) total * frac);
    c.profile = std::getenv("H3S_PROFILE") != nullptr;
}

}  // namespace

extern "C" {

const char* h3s_last_error(void) { return g_err.c_str(); }

void* h3s_create(void* sycl_queue) try {
    auto* c = new Ctx{*static_cast<sycl::queue*>(sycl_queue)};
    init(*c);
    return c;
} catch (const std::exception& e) { g_err = e.what(); return nullptr; }

void* h3s_open(void) try {
    auto* c = new Ctx{sycl::queue(sycl::gpu_selector_v, sycl::property::queue::in_order())};
    init(*c);
    return c;
} catch (const std::exception& e) { g_err = e.what(); return nullptr; }

const char* h3s_device_name(void* ctx) { return static_cast<Ctx*>(ctx)->name.c_str(); }
uint64_t h3s_mem_cap(void* ctx) { return static_cast<Ctx*>(ctx)->mem_cap; }
uint64_t h3s_mem_used(void* ctx) {
    auto& c = *static_cast<Ctx*>(ctx);
    std::lock_guard<std::mutex> l(c.mem_mu);
    return c.mem_used;
}

void* h3s_alloc(void* ctx, uint64_t bytes) try {
    auto& c = *static_cast<Ctx*>(ctx);
    std::lock_guard<std::mutex> l(c.mem_mu);
    if (c.mem_used + bytes > c.mem_cap) {
        g_err = "h3s_alloc: " + std::to_string(bytes >> 20) + " MiB would pass the device memory cap (" +
                std::to_string(c.mem_used >> 20) + " of " + std::to_string(c.mem_cap >> 20) + " MiB in use)";
        return nullptr;
    }
    void* p = sycl::malloc_device(bytes ? bytes : 1, c.q);
    if (!p) { g_err = "h3s_alloc: the device refused " + std::to_string(bytes >> 20) + " MiB"; return nullptr; }
    c.mem.emplace(p, bytes);
    c.mem_used += bytes;
    return p;
} catch (const std::exception& e) { g_err = e.what(); return nullptr; }

void h3s_free(void* ctx, void* p) {
    auto& c = *static_cast<Ctx*>(ctx);
    if (!p) return;
    c.q.wait();                                    // nothing queued may still read or write it
    std::lock_guard<std::mutex> l(c.mem_mu);
    auto it = c.mem.find(p);
    if (it == c.mem.end()) return;
    c.mem_used -= it->second;
    c.mem.erase(it);
    sycl::free(p, c.q);
}

int h3s_write(void* ctx, void* dst, const void* src_host, uint64_t bytes) try {
    static_cast<Ctx*>(ctx)->q.memcpy(dst, src_host, bytes).wait();
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

int h3s_read(void* ctx, void* dst_host, const void* src, uint64_t bytes) try {
    static_cast<Ctx*>(ctx)->q.memcpy(dst_host, src, bytes).wait();
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

int h3s_wait(void* ctx) try {
    static_cast<Ctx*>(ctx)->q.wait_and_throw();
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

void h3s_destroy(void* ctx) {
    auto* c = static_cast<Ctx*>(ctx);
    if (!c) return;
    c->q.wait();
    for (void* p : {(void*) c->xq, (void*) c->rs, (void*) c->gmax, (void*) c->xr, (void*) c->acc, c->had[0], c->had[1], c->had[2],
                    (void*) c->inv, (void*) c->hq, (void*) c->hk, (void*) c->hv, (void*) c->sc, (void*) c->ao, (void*) c->part, (void*) c->rowv})
        if (p) sycl::free(p, c->q);
    for (auto& [p, n] : c->mem) sycl::free(p, c->q);
    delete c;
}

// The passes (the contract is in h3sycl.h):
//   1. rotation     x_rot = x . H per group: ONE GEMM [M * K / g, g] x [g, g] on the matrix engine (it is ~0.3 ms;
//                   a scalar butterfly kernel was 5 ms a pass)
//   2. row scales   absmax per (row, group), then per row
//   3. quantize     one value per work-item
//   4. int8 GEMM    oneDNN, the rescale (per-row scale, optional per-column weight scale) and the bias as post-ops,
//                   written straight into the output in its own type - no int32 table, no second pass over it
int h3s_int8_linear(void* ctx, const void* x, int x_dt, int64_t M, int64_t K, const int8_t* w, int64_t N,
                    const float* wscale, int64_t n_wscale, const float* bias, void* out, int out_dt, int group) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (M <= 0 || K <= 0 || N <= 0) return 0;
    const bool rot = group > 0;
    if (rot && (group > kMaxGroup || K % group != 0 || (group & (group - 1)) != 0)) {
        g_err = "h3s_int8_linear: unsupported ConvRot group size"; return -1;
    }
    const int g = rot ? group : 256;
    const int64_t G = (K + g - 1) / g;
    int8_t* xq = c.grow(c.xq, c.xq_cap, (size_t) M * K);
    float* rs = c.grow(c.rs, c.rs_cap, (size_t) M);
    float* gmax = c.grow(c.gmax, c.gmax_cap, (size_t) M * G);
    if (!xq || !rs || !gmax) return -1;
    sycl::queue& q = c.q;
    using dnnl::memory;

    // ---- 1. rotation on the matrix engine
    const void* xs = x;          // what the quantizer reads
    int xs_dt = x_dt;
    if (rot) {
        float* xr = c.grow(c.xr, c.xr_cap, (size_t) M * K);
        if (!xr) return -1;
        // The rotated copy is float32. In bfloat16 (8 significant bits) its rounding moves ~5% of the values to the
        // neighbouring int8 level in pass 3: 0.7% error on the layer's output, measured against exact arithmetic,
        // next to the 0.9% the 8-bit quantization costs by itself.
        int r_dt = c.rot_f32 ? F32 : x_dt;
        const int64_t rows = M * (K / g);
        memory::desc smd({rows, g}, ddt(x_dt), memory::format_tag::ab);
        memory::desc hmd({g, g}, ddt(x_dt), memory::format_tag::ab);
        auto key = std::make_tuple(rows, g, x_dt);
        auto it = c.rot.find(key);
        if (it == c.rot.end()) {
            try {
                memory::desc d32({rows, g}, ddt(r_dt), memory::format_tag::ab);
                it = c.rot.emplace(key, dnnl::matmul(dnnl::matmul::primitive_desc(c.eng, smd, hmd, d32))).first;
            } catch (const dnnl::error& e) {
                if (r_dt == x_dt) throw;
                std::fprintf(stderr, "h3sycl: oneDNN refused a float32 rotation result (%s); keeping the input's type\n", e.what());
                c.rot_f32 = false; r_dt = x_dt;
                memory::desc d16({rows, g}, ddt(r_dt), memory::format_tag::ab);
                it = c.rot.emplace(key, dnnl::matmul(dnnl::matmul::primitive_desc(c.eng, smd, hmd, d16))).first;
            }
        }
        memory::desc dmd({rows, g}, ddt(r_dt), memory::format_tag::ab);
        it->second.execute(c.strm, {{DNNL_ARG_SRC, usm(smd, c.eng, x)}, {DNNL_ARG_WEIGHTS, usm(hmd, c.eng, hadamard_matrix(c, g, x_dt))},
                                    {DNNL_ARG_DST, usm(dmd, c.eng, xr)}});
        xs = xr; xs_dt = r_dt;
    }
    // ---- 2. row scales
    q.parallel_for(sycl::range<2>((size_t) M, (size_t) G), [=](sycl::id<2> id) {
        const size_t r = id[0], gi = id[1];
        const int n = (int) std::min<int64_t>(g, K - (int64_t) gi * g);
        float m = 0.0f;
        for (int i = 0; i < n; ++i) m = sycl::fmax(m, sycl::fabs(load(xs, xs_dt, r * K + gi * g + i)));
        gmax[r * G + gi] = m;
    });
    const bool per_n = n_wscale > 1;
    q.parallel_for(sycl::range<1>((size_t) M), [=](sycl::id<1> r) {
        float m = 0.0f;
        for (int64_t gi = 0; gi < G; ++gi) m = sycl::fmax(m, gmax[r[0] * G + gi]);
        // the row's quantization step; times the scalar weight scale it is the whole rescale factor
        rs[r[0]] = sycl::fmax(m / 127.0f, 1e-30f);
    });
    // ---- 3. quantize
    q.parallel_for(sycl::range<2>((size_t) M, (size_t) K), [=](sycl::id<2> id) {
        const size_t i = id[0] * K + id[1];
        xq[i] = (int8_t) sycl::clamp(sycl::rint(load(xs, xs_dt, i) / rs[id[0]]), -128.0f, 127.0f);
    });
    if (!per_n) {
        const float ws = 0.0f;   // read on the device below
        (void) ws;
        q.parallel_for(sycl::range<1>((size_t) M), [=](sycl::id<1> r) { rs[r[0]] *= wscale[0]; });
    }
    // ---- 4. int8 GEMM with the rescale and the bias as post-ops
    memory::desc smd({M, K}, memory::data_type::s8, memory::format_tag::ab);
    memory::desc wmd({K, N}, memory::data_type::s8, memory::format_tag::ba);     // the [N, K] buffer, read transposed
    memory::desc row_md({M, 1}, memory::data_type::f32, memory::format_tag::ab);
    memory::desc col_md({1, N}, memory::data_type::f32, memory::format_tag::ab);
    if (c.fused_ok) {
        try {
            memory::desc dmd({M, N}, ddt(out_dt), memory::format_tag::ab);
            auto key = std::make_tuple(M, K, N, out_dt, bias ? 1 : 0, per_n ? 1 : 0);
            auto it = c.gemm.find(key);
            if (it == c.gemm.end()) {
                dnnl::post_ops po;
                po.append_binary(dnnl::algorithm::binary_mul, row_md);
                if (per_n) po.append_binary(dnnl::algorithm::binary_mul, col_md);
                if (bias) po.append_binary(dnnl::algorithm::binary_add, col_md);
                dnnl::primitive_attr attr;
                attr.set_post_ops(po);
                it = c.gemm.emplace(key, dnnl::matmul(dnnl::matmul::primitive_desc(c.eng, smd, wmd, dmd, attr))).first;
            }
            std::unordered_map<int, memory> args{{DNNL_ARG_SRC, usm(smd, c.eng, xq)}, {DNNL_ARG_WEIGHTS, usm(wmd, c.eng, w)},
                                                 {DNNL_ARG_DST, usm(dmd, c.eng, out)}};
            int po_i = 0;
            args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(po_i++) | DNNL_ARG_SRC_1, usm(row_md, c.eng, rs)});
            if (per_n) args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(po_i++) | DNNL_ARG_SRC_1, usm(col_md, c.eng, wscale)});
            if (bias) args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(po_i++) | DNNL_ARG_SRC_1, usm(col_md, c.eng, bias)});
            it->second.execute(c.strm, args);
            return 0;
        } catch (const dnnl::error& e) {
            std::fprintf(stderr, "h3sycl: oneDNN refused the fused int8 GEMM (%s); using the int32 + rescale path\n", e.what());
            c.fused_ok = false;
        }
    }
    // ---- fallback: int32 products in row chunks, rescaled by a kernel of ours
    const int64_t C = std::max<int64_t>(1, std::min<int64_t>(M, (int64_t) (384ull << 20) / (N * 4)));
    int32_t* acc = c.grow(c.acc, c.acc_cap, (size_t) C * N);
    if (!acc) return -1;
    for (int64_t r0 = 0; r0 < M; r0 += C) {
        const int64_t rows = std::min(C, M - r0);
        memory::desc s2({rows, K}, memory::data_type::s8, memory::format_tag::ab);
        memory::desc d2({rows, N}, memory::data_type::s32, memory::format_tag::ab);
        auto key = std::make_tuple(rows, K, N, -1, 0, 0);
        auto it = c.gemm.find(key);
        if (it == c.gemm.end()) it = c.gemm.emplace(key, dnnl::matmul(dnnl::matmul::primitive_desc(c.eng, s2, wmd, d2))).first;
        it->second.execute(c.strm, {{DNNL_ARG_SRC, usm(s2, c.eng, xq + r0 * K)}, {DNNL_ARG_WEIGHTS, usm(wmd, c.eng, w)},
                                    {DNNL_ARG_DST, usm(d2, c.eng, acc)}});
        q.parallel_for(sycl::range<2>((size_t) rows, (size_t) N), [=](sycl::id<2> id) {
            const size_t r = id[0], n = id[1];
            float v = (float) acc[r * N + n] * rs[r0 + r] * (per_n ? wscale[n] : 1.0f);
            if (bias) v += bias[n];
            store(out, out_dt, (size_t) (r0 + r) * N + n, v);
        });
    }
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

// 1 / sqrt(mean(x[r]^2) + eps) per row of x [M, C] into c.inv. Summed in float32, in a fixed order per row.
static float* row_inv_rms(Ctx& c, const void* x, int x_dt, int64_t M, int64_t C, float eps) {
    float* inv = c.grow(c.inv, c.inv_cap, (size_t) M);
    if (!inv) return nullptr;
    c.q.parallel_for(sycl::range<1>((size_t) M), [=](sycl::id<1> r) {
        float s = 0.0f;
        for (int64_t i = 0; i < C; ++i) {
            const float v = load(x, x_dt, r[0] * C + i);
            s += v * v;
        }
        inv[r[0]] = sycl::rsqrt(s / (float) C + eps);
    });
    return inv;
}

int h3s_rms_norm_mod(void* ctx, const void* x, int x_dt, int64_t M, int64_t C, const float* weight, float eps,
                     const int32_t* rows, const float* scale, const float* shift, void* out, int out_dt) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (M <= 0 || C <= 0) return 0;
    const float* inv = row_inv_rms(c, x, x_dt, M, C, eps);
    if (!inv) return -1;
    const bool mod = rows && scale && shift;
    c.q.parallel_for(sycl::range<2>((size_t) M, (size_t) C), [=](sycl::id<2> id) {
        const size_t r = id[0], i = id[1];
        float v = load(x, x_dt, r * C + i) * inv[r] * weight[i];
        if (mod) {
            const size_t m = (size_t) rows[r] * C + i;
            v = v * (1.0f + scale[m]) + shift[m];
        }
        store(out, out_dt, r * C + i, v);
    });
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

int h3s_rms_rope(void* ctx, void* x, int x_dt, int64_t M, int64_t H, int64_t D, int64_t stride, const float* weight,
                 float eps, const float* cs, int rot_dim) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (M <= 0 || H <= 0 || D <= 0) return 0;
    if (rot_dim < 0 || rot_dim > D || rot_dim % 2 != 0 || (D - rot_dim) % 2 != 0) {
        g_err = "h3s_rms_rope: rot_dim must be even, at most D, and leave an even remainder"; return -1;
    }
    if (stride < H * D) { g_err = "h3s_rms_rope: the row stride is shorter than a row"; return -1; }
    float* inv = c.grow(c.inv, c.inv_cap, (size_t) (M * H));
    if (!inv) return -1;
    c.q.parallel_for(sycl::range<1>((size_t) (M * H)), [=](sycl::id<1> mh) {
        const size_t base = (mh[0] / H) * stride + (mh[0] % H) * D;
        float s = 0.0f;
        for (int64_t i = 0; i < D; ++i) {
            const float v = load(x, x_dt, base + i);
            s += v * v;
        }
        inv[mh[0]] = sycl::rsqrt(s / (float) D + eps);
    });
    const int64_t half = rot_dim / 2;                  // rotated pairs: (i, half + i)
    const int64_t slots = half + (D - rot_dim) / 2;    // then pairs of the features passed through
    // one work-item per pair: it owns its two values, so the update is in place
    c.q.parallel_for(sycl::range<2>((size_t) (M * H), (size_t) slots), [=](sycl::id<2> id) {
        const size_t mh = id[0], j = id[1], base = (mh / H) * stride + (mh % H) * D;
        const float s = inv[mh];
        if ((int64_t) j < half) {
            const size_t ia = base + j, ib = base + half + j;
            const float a = load(x, x_dt, ia) * s * weight[j], b = load(x, x_dt, ib) * s * weight[half + j];
            const size_t t = (mh / H) * half + j;      // the token's angle for this pair
            const float co = cs[2 * t], si = cs[2 * t + 1];
            store(x, x_dt, ia, a * co - b * si);
            store(x, x_dt, ib, b * co + a * si);
        } else {
            const size_t i0 = rot_dim + 2 * (j - half);
            store(x, x_dt, base + i0, load(x, x_dt, base + i0) * s * weight[i0]);
            store(x, x_dt, base + i0 + 1, load(x, x_dt, base + i0 + 1) * s * weight[i0 + 1]);
        }
    });
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

int h3s_swiglu(void* ctx, const void* x, int x_dt, int64_t M, int64_t C, void* out, int out_dt) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (M <= 0 || C <= 0) return 0;
    c.q.parallel_for(sycl::range<2>((size_t) M, (size_t) C), [=](sycl::id<2> id) {
        const size_t r = id[0], i = id[1];
        const float g = load(x, x_dt, r * 2 * C + i), u = load(x, x_dt, r * 2 * C + C + i);
        store(out, out_dt, r * C + i, g / (1.0f + sycl::exp(-g)) * u);
    });
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

int h3s_gate_add(void* ctx, void* x, int x_dt, int64_t M, int64_t C, const void* other, int other_dt,
                 const int32_t* rows, const float* gate) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (M <= 0 || C <= 0) return 0;
    c.q.parallel_for(sycl::range<2>((size_t) M, (size_t) C), [=](sycl::id<2> id) {
        const size_t r = id[0], i = id[1], p = r * C + i;
        const float g = gate ? gate[(size_t) rows[r] * C + i] : 1.0f;
        store(x, x_dt, p, load(x, x_dt, p) + load(other, other_dt, p) * g);
    });
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

// Attention with a bounded score table. A chunk of query rows at a time, so the scores held at once are
// heads x rows x S <= ~1.5 GiB whatever S is:
//
//   scores   q . k^T               oneDNN, on the matrix engine          -> half [H, rows, S]
//   weights  exp(score - row max)  ours, in place, one score per work-item (the row's max from a read-only pass)
//   values   weights . [v | 1]     oneDNN, on the matrix engine          -> float32 [H, rows, D + 1]
//   out      values / row sum      ours, folded into the copy back to token order
//
// The column of ones beside v makes the matrix engine deliver each row's sum of weights with the values, so no pass
// of ours has to add them up. q, k, v are stored by token; the multiplies want them by head, so k and v are copied
// once and q a chunk at a time, in IEEE half (11 significant bits where bfloat16 has 8); 1 / sqrt(D) rides on q.
//
// Two things oneDNN offers for this and why they are not used:
//  - its stand-alone softmax: 1.0 s of a 1.2 s call at 16.5k tokens (it is slow on rows that long);
//  - its fused attention through the graph interface (MatMul -> Divide -> SoftMax -> MatMul): on this card oneDNN
//    3.11 runs every form of that pattern as separate steps with the WHOLE S x S table in device memory
//    (kernels/sdpa_probe.cpp), and nothing tells the caller. At 16.5k tokens that table is 30 GiB; the xe driver has
//    no out-of-memory error, the card spilled into host RAM and the machine went down (2026-10-02).
//
// H3S_PROFILE=1 in the environment: wait after every phase and report where a call's time went (slower).
int h3s_attention(void* ctx, const void* q, const void* k, const void* v, int dt, int64_t S, int64_t H, int64_t D,
                  int64_t stride, void* out, int out_dt) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (S <= 0 || H <= 0 || D <= 0) return 0;
    if (stride < H * D) { g_err = "h3s_attention: the row stride is shorter than a row"; return -1; }
    using dnnl::memory;
    sycl::queue& qu = c.q;
    const int64_t D1 = D + 1;                           // v with its column of ones
    // rows per chunk: the scores of a chunk take at most ~1.5 GiB
    const int64_t rows_max = std::max<int64_t>(1, std::min<int64_t>(S, (int64_t) (1536ull << 20) / (H * S * 2)));
    // the max pass: work-item j of a row takes scores j, j + kLanes, j + 2 kLanes ..., so neighbouring work-items
    // (which the GPU runs as the lanes of one vector) read neighbouring scores
    constexpr int64_t kLanes = 256;
    const int64_t G = kLanes;
    uint16_t* hq = c.grow(c.hq, c.hq_cap, (size_t) H * rows_max * D);
    uint16_t* hk = c.grow(c.hk, c.hk_cap, (size_t) S * H * D);
    uint16_t* hv = c.grow(c.hv, c.hv_cap, (size_t) S * H * D1);
    uint16_t* sc = c.grow(c.sc, c.sc_cap, (size_t) H * rows_max * S);
    float* ao = c.grow(c.ao, c.ao_cap, (size_t) H * rows_max * D1);
    float* part = c.grow(c.part, c.part_cap, (size_t) (H * rows_max * G));
    float* rowv = c.grow(c.rowv, c.rowv_cap, (size_t) (H * rows_max));
    if (!hq || !hk || !hv || !sc || !ao || !part || !rowv) return -1;

    const bool prof = c.profile;
    double t_ph[6] = {0, 0, 0, 0, 0, 0};
    auto clock = std::chrono::steady_clock::now();
    auto lap = [&](int i) {
        if (!prof) return;
        qu.wait();
        const auto now = std::chrono::steady_clock::now();
        t_ph[i] += std::chrono::duration<double, std::milli>(now - clock).count();
        clock = now;
    };
    lap(0);
    qu.parallel_for(sycl::range<3>((size_t) H, (size_t) S, (size_t) D1), [=](sycl::id<3> id) {
        const size_t h = id[0], t = id[1], d = id[2];
        if ((int64_t) d < D) {
            const size_t src = t * stride + h * D + d;
            ((sycl::half*) hk)[(h * D + d) * S + t] = (sycl::half) load(k, dt, src);
            ((sycl::half*) hv)[(h * S + t) * D1 + d] = (sycl::half) load(v, dt, src);
        } else {
            ((sycl::half*) hv)[(h * S + t) * D1 + d] = (sycl::half) 1.0f;
        }
    });
    const float scale = 1.0f / std::sqrt((float) D);
    const auto f16 = memory::data_type::f16;
    for (int64_t r0 = 0; r0 < S; r0 += rows_max) {
        const int64_t rows = std::min(rows_max, S - r0);
        const size_t R = (size_t) (H * rows);           // rows of the score table of this chunk
        qu.parallel_for(sycl::range<3>((size_t) H, (size_t) rows, (size_t) D), [=](sycl::id<3> id) {
            ((sycl::half*) hq)[(id[0] * rows + id[1]) * D + id[2]] =
                    (sycl::half) (load(q, dt, (r0 + id[1]) * stride + id[0] * D + id[2]) * scale);
        });
        memory::desc q_md({H, rows, D}, f16, memory::format_tag::abc);
        memory::desc k_md({H, D, S}, f16, memory::format_tag::abc);
        memory::desc s_md({H, rows, S}, f16, memory::format_tag::abc);
        memory::desc v_md({H, S, D1}, f16, memory::format_tag::abc);
        memory::desc o_md({H, rows, D1}, memory::data_type::f32, memory::format_tag::abc);
        auto key = std::make_tuple(H, rows, S, D);
        auto it = c.attn.find(key);
        if (it == c.attn.end()) {
            Ctx::Attn a{dnnl::matmul(dnnl::matmul::primitive_desc(c.eng, q_md, k_md, s_md)),
                        dnnl::matmul(dnnl::matmul::primitive_desc(c.eng, s_md, v_md, o_md))};
            it = c.attn.emplace(key, std::move(a)).first;
        }
        auto s_mem = usm(s_md, c.eng, sc);
        lap(0);
        it->second.qk.execute(c.strm, {{DNNL_ARG_SRC, usm(q_md, c.eng, hq)}, {DNNL_ARG_WEIGHTS, usm(k_md, c.eng, hk)}, {DNNL_ARG_DST, s_mem}});
        lap(1);
        // the row's max (read only)
        sycl::half* sh = (sycl::half*) sc;
        qu.parallel_for(sycl::range<2>(R, (size_t) G), [=](sycl::id<2> id) {
            const size_t base = id[0] * S;
            float m = -3.0e38f;
            for (size_t i = id[1]; i < (size_t) S; i += kLanes) m = sycl::fmax(m, (float) sh[base + i]);
            part[id[0] * G + id[1]] = m;
        });
        qu.parallel_for(sycl::range<1>(R), [=](sycl::id<1> r) {
            float m = -3.0e38f;
            for (int64_t g = 0; g < G; ++g) m = sycl::fmax(m, part[r[0] * G + g]);
            rowv[r[0]] = m;
        });
        lap(2);
        // weights in place, each at most 1
        qu.parallel_for(sycl::range<2>(R, (size_t) S), [=](sycl::id<2> id) {
            const size_t i = id[0] * S + id[1];
            sh[i] = (sycl::half) exp_neg((float) sh[i] - rowv[id[0]]);
        });
        lap(3);
        it->second.pv.execute(c.strm, {{DNNL_ARG_SRC, s_mem}, {DNNL_ARG_WEIGHTS, usm(v_md, c.eng, hv)}, {DNNL_ARG_DST, usm(o_md, c.eng, ao)}});
        lap(4);
        // divide by the row's sum of weights (the last column), and back to token order: out [S, H * D]
        qu.parallel_for(sycl::range<3>((size_t) rows, (size_t) H, (size_t) D), [=](sycl::id<3> id) {
            const size_t hr = (id[1] * rows + id[0]) * D1;
            store(out, out_dt, ((r0 + id[0]) * H + id[1]) * D + id[2], ao[hr + id[2]] / ao[hr + D]);
        });
        lap(5);
    }
    if (prof)
        std::fprintf(stderr, "h3sycl: attention S=%lld: copies %.1f ms, scores %.1f, row max %.1f, exp %.1f, values %.1f, out %.1f\n",
                     (long long) S, t_ph[0], t_ph[1], t_ph[2], t_ph[3], t_ph[4], t_ph[5]);
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

}  // extern "C"
