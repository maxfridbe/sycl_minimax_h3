// h3sycl.cpp - SYCL kernels for MiniMax H3 on Intel Arc (Xe2), loaded into the PyTorch-XPU process.
//
// A plain C ABI (ctypes-friendly). Every pointer is a USM device pointer from the SAME SYCL context PyTorch uses,
// and the queue is PyTorch's own in-order queue, so calls here are ordered with PyTorch's operations and nothing
// is copied or synchronized by hand.
//
//   h3s_create(queue)                 a context around PyTorch's sycl::queue (scratch buffers, oneDNN engine)
//   h3s_int8_linear(...)              comfy-kitchen's int8_linear: [rotate] -> quantize rows -> int8 GEMM -> rescale
//
// int8_linear, as comfy-kitchen's eager backend defines it:
//   x_rot = x . H           per group of `group` features, H the normalized regular Hadamard matrix (ConvRot)
//   s_r   = max|x_rot[r]| / 127                                  (per row, at least 1e-30)
//   q     = clamp(round(x_rot / s_r), -128, 127)                 int8
//   acc   = q . W^T                                              int32, W int8 [N, K]   <- the matrix engine, oneDNN
//   y     = cast(acc * (s_r * w_scale)) + bias                   in the output type
#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
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
// a value rounded through the output type (what "cast, then add the bias in that type" does)
inline float through(int dt, float v) {
    switch (dt) {
        case F16: return (float) (sycl::half) v;
        case BF16: return bf16_to_f32(f32_to_bf16(v));
        default: return v;
    }
}

constexpr int kMaxGroup = 256;

// v[0..g) *= regular Hadamard (h4 (x) h4 (x) ...), normalized by 1/sqrt(g). g a power of 4, <= 256.
inline void hadamard(float* v, int g) {
    for (int s = 1; s < g; s *= 4)
        for (int b = 0; b < g; b += 4 * s)
            for (int i = b; i < b + s; ++i) {
                const float a = v[i], bb = v[i + s], c = v[i + 2 * s], d = v[i + 3 * s];
                v[i] = a + bb + c - d;
                v[i + s] = a + bb - c + d;
                v[i + 2 * s] = a - bb + c + d;
                v[i + 3 * s] = -a + bb + c + d;
            }
    float norm = 1.0f;
    for (int s = 1; s < g; s *= 4) norm *= 0.5f;       // 1/sqrt(g) = 2^-(log4 g)
    for (int i = 0; i < g; ++i) v[i] *= norm;
}

struct Ctx {
    sycl::queue q;
    dnnl::engine eng;
    dnnl::stream strm;
    int8_t* xq = nullptr; size_t xq_cap = 0;       // quantized activations [M, K]
    float* rs = nullptr; size_t rs_cap = 0;        // per-row scale times the scalar weight scale [M]
    float* gmax = nullptr; size_t gmax_cap = 0;    // per-(row, group) absmax
    uint16_t* xr = nullptr; size_t xr_cap = 0;     // rotated activations [M, K], in x's 16-bit type (f32 x: f16)
    int32_t* acc = nullptr; size_t acc_cap = 0;    // fallback path: one chunk of int32 products [C, N]
    void* had[3] = {nullptr, nullptr, nullptr};    // the normalized 256 x 256 Hadamard matrix per Dt
    std::map<std::tuple<int64_t, int64_t, int64_t, int, int, int>, dnnl::matmul> gemm;   // (M, K, N, out, bias, per_n)
    std::map<std::tuple<int64_t, int, int>, dnnl::matmul> rot;                            // (rows, group, dt)
    bool fused_ok = true;                          // oneDNN took the post-op form (else: int32 + our rescale kernel)

    template <typename T> T* grow(T*& p, size_t& cap, size_t n) {
        if (n > cap) {
            if (p) { q.wait(); sycl::free(p, q); }
            p = sycl::malloc_device<T>(n + (n >> 3), q);
            cap = p ? n + (n >> 3) : 0;
        }
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

}  // namespace

extern "C" {

const char* h3s_last_error() { return g_err.c_str(); }

// `sycl_queue`: a `sycl::queue*` (PyTorch: torch.xpu.current_stream().sycl_queue). The queue is copied (a handle).
void* h3s_create(void* sycl_queue) try {
    auto* c = new Ctx{*static_cast<sycl::queue*>(sycl_queue), {}, {}};
    c->eng = dnnl::sycl_interop::make_engine(c->q.get_device(), c->q.get_context());
    c->strm = dnnl::sycl_interop::make_stream(c->eng, c->q);
    return c;
} catch (const std::exception& e) { g_err = e.what(); return nullptr; }

void h3s_destroy(void* ctx) {
    auto* c = static_cast<Ctx*>(ctx);
    if (!c) return;
    c->q.wait();
    for (void* p : {(void*) c->xq, (void*) c->rs, (void*) c->gmax, (void*) c->xr, (void*) c->acc, c->had[0], c->had[1], c->had[2]})
        if (p) sycl::free(p, c->q);
    delete c;
}

// x [M, K] (x_dt), w int8 [N, K], wscale float32 [1 or N], bias float32 [N] or null, out [M, N] (out_dt).
// group = 0: no rotation; else the ConvRot group size (a power of 4, <= 256, dividing K).
// Returns 0, or -1 with h3s_last_error(). Asynchronous: the result is complete when the queue reaches it.
//
// The passes:
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
    if (!xq || !rs || !gmax) { g_err = "h3s_int8_linear: device allocation failed"; return -1; }
    sycl::queue& q = c.q;
    using dnnl::memory;

    // ---- 1. rotation on the matrix engine
    const void* xs = x;          // what the quantizer reads
    int xs_dt = x_dt;
    if (rot) {
        uint16_t* xr = c.grow(c.xr, c.xr_cap, (size_t) M * K);
        if (!xr) { g_err = "h3s_int8_linear: device allocation failed"; return -1; }
        const int r_dt = x_dt == F32 ? F16 : x_dt;                       // the rotated copy is 16-bit
        const int64_t rows = M * (K / g);
        memory::desc smd({rows, g}, ddt(x_dt), memory::format_tag::ab);
        memory::desc hmd({g, g}, ddt(x_dt), memory::format_tag::ab);
        memory::desc dmd({rows, g}, ddt(r_dt), memory::format_tag::ab);
        auto key = std::make_tuple(rows, g, x_dt);
        auto it = c.rot.find(key);
        if (it == c.rot.end()) it = c.rot.emplace(key, dnnl::matmul(dnnl::matmul::primitive_desc(c.eng, smd, hmd, dmd))).first;
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
    if (!acc) { g_err = "h3s_int8_linear: device allocation failed"; return -1; }
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

}  // extern "C"
