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
    float* xs = nullptr; size_t xs_cap = 0;        // per-row scales [M]
    float* gmax = nullptr; size_t gmax_cap = 0;    // per-(row, group) absmax
    int32_t* acc = nullptr; size_t acc_cap = 0;    // one chunk of int32 products [C, N]
    std::map<std::tuple<int64_t, int64_t, int64_t>, dnnl::matmul> prims;

    template <typename T> T* grow(T*& p, size_t& cap, size_t n) {
        if (n > cap) {
            if (p) { q.wait(); sycl::free(p, q); }
            p = sycl::malloc_device<T>(n + (n >> 3), q);
            cap = p ? n + (n >> 3) : 0;
        }
        return p;
    }
};

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
    for (void* p : {(void*) c->xq, (void*) c->xs, (void*) c->gmax, (void*) c->acc}) if (p) sycl::free(p, c->q);
    delete c;
}

// x [M, K] (x_dt), w int8 [N, K], wscale float32 [1 or N], bias [N] in out_dt or null, out [M, N] (out_dt).
// group = 0: no rotation; else the ConvRot group size (a power of 4, <= 256, dividing K).
// Returns 0, or -1 with h3s_last_error(). Asynchronous: the result is complete when the queue reaches it.
int h3s_int8_linear(void* ctx, const void* x, int x_dt, int64_t M, int64_t K, const int8_t* w, int64_t N,
                    const float* wscale, int64_t n_wscale, const void* bias, void* out, int out_dt, int group) try {
    auto& c = *static_cast<Ctx*>(ctx);
    if (M <= 0 || K <= 0 || N <= 0) return 0;
    const int g = group > 0 ? group : 256;                 // without rotation: 256-wide pieces of the row (or less)
    const bool rot = group > 0;
    if (rot && (group > kMaxGroup || K % group != 0 || (group & (group - 1)) != 0)) {
        g_err = "h3s_int8_linear: unsupported ConvRot group size"; return -1;
    }
    const int64_t G = (K + g - 1) / g;
    int8_t* xq = c.grow(c.xq, c.xq_cap, (size_t) M * K);
    float* xs = c.grow(c.xs, c.xs_cap, (size_t) M);
    float* gmax = c.grow(c.gmax, c.gmax_cap, (size_t) M * G);
    if (!xq || !xs || !gmax) { g_err = "h3s_int8_linear: device allocation failed"; return -1; }
    sycl::queue& q = c.q;

    // ---- 1. per (row, group): rotate, absmax
    q.parallel_for(sycl::range<2>((size_t) M, (size_t) G), [=](sycl::id<2> id) {
        const size_t r = id[0], gi = id[1];
        const int n = (int) std::min<int64_t>(g, K - (int64_t) gi * g);
        float v[kMaxGroup];
        for (int i = 0; i < n; ++i) v[i] = load(x, x_dt, r * K + gi * g + i);
        if (rot) hadamard(v, g);
        float m = 0.0f;
        for (int i = 0; i < n; ++i) m = sycl::fmax(m, sycl::fabs(v[i]));
        gmax[r * G + gi] = m;
    });
    // ---- 2. per row: the scale
    q.parallel_for(sycl::range<1>((size_t) M), [=](sycl::id<1> r) {
        float m = 0.0f;
        for (int64_t gi = 0; gi < G; ++gi) m = sycl::fmax(m, gmax[r[0] * G + gi]);
        xs[r[0]] = sycl::fmax(m / 127.0f, 1e-30f);
    });
    // ---- 3. per (row, group): rotate again, quantize
    q.parallel_for(sycl::range<2>((size_t) M, (size_t) G), [=](sycl::id<2> id) {
        const size_t r = id[0], gi = id[1];
        const int n = (int) std::min<int64_t>(g, K - (int64_t) gi * g);
        float v[kMaxGroup];
        for (int i = 0; i < n; ++i) v[i] = load(x, x_dt, r * K + gi * g + i);
        if (rot) hadamard(v, g);
        const float inv = 1.0f / xs[r];
        for (int i = 0; i < n; ++i)
            xq[r * K + gi * g + i] = (int8_t) sycl::clamp(sycl::rint(v[i] * inv), -128.0f, 127.0f);
    });
    // ---- 4. int8 GEMM in row chunks (the int32 products of a whole [M, N] would be gigabytes), 5. rescale
    const int64_t C = std::max<int64_t>(1, std::min<int64_t>(M, (int64_t) (384ull << 20) / (N * 4)));
    int32_t* acc = c.grow(c.acc, c.acc_cap, (size_t) C * N);
    if (!acc) { g_err = "h3s_int8_linear: device allocation failed"; return -1; }
    using dnnl::memory;
    for (int64_t r0 = 0; r0 < M; r0 += C) {
        const int64_t rows = std::min(C, M - r0);
        memory::desc smd({rows, K}, memory::data_type::s8, memory::format_tag::ab);
        memory::desc wmd({K, N}, memory::data_type::s8, memory::format_tag::ba);     // the [N, K] buffer, read transposed
        memory::desc dmd({rows, N}, memory::data_type::s32, memory::format_tag::ab);
        auto key = std::make_tuple(rows, K, N);
        auto it = c.prims.find(key);
        if (it == c.prims.end())
            it = c.prims.emplace(key, dnnl::matmul(dnnl::matmul::primitive_desc(c.eng, smd, wmd, dmd))).first;
        memory src = dnnl::sycl_interop::make_memory(smd, c.eng, dnnl::sycl_interop::memory_kind::usm, xq + r0 * K);
        memory wei = dnnl::sycl_interop::make_memory(wmd, c.eng, dnnl::sycl_interop::memory_kind::usm, (void*) w);
        memory dst = dnnl::sycl_interop::make_memory(dmd, c.eng, dnnl::sycl_interop::memory_kind::usm, acc);
        it->second.execute(c.strm, {{DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, wei}, {DNNL_ARG_DST, dst}});
        const bool per_n = n_wscale > 1;
        q.parallel_for(sycl::range<2>((size_t) rows, (size_t) N), [=](sycl::id<2> id) {
            const size_t r = id[0], n = id[1];
            float v = through(out_dt, (float) acc[r * N + n] * (xs[r0 + r] * wscale[per_n ? n : 0]));
            if (bias) v += load(bias, out_dt, n);
            store(out, out_dt, (size_t) (r0 + r) * N + n, v);
        });
    }
    return 0;
} catch (const std::exception& e) { g_err = e.what(); return -1; }

}  // extern "C"
