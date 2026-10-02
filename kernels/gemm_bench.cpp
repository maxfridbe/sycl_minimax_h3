// gemm_bench.cpp - how fast does the B70's matrix engine multiply at H3's shapes, int8 against 16-bit float?
//
//   build (oneAPI image):  icpx -O2 -fsycl gemm_bench.cpp -ldnnl -o gemm_bench
//   run:                   ./gemm_bench [reps]
//
// Uses oneDNN's matmul primitive on the GPU engine - the same library PyTorch-XPU calls - so the f16 line is today's
// floor and the s8 line is what an int8 path could reach without writing a kernel of our own.
//
// Two families of shapes:
//   "linear"    tokens x in_features  times  in_features x out_features      (the DiT's four big Linear layers)
//   "scores"    56 heads x [T x 128] times [128 x T]                          (attention's Q.K^T, one T x T tile per head)
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace dnnl;
using dt = memory::data_type;

static void fill(memory& m, dt t) {
    void* p = m.map_data<void>();
    const size_t bytes = m.get_desc().get_size();
    if (t == dt::s8 || t == dt::u8) {
        auto* b = static_cast<uint8_t*>(p);
        for (size_t i = 0; i < bytes; ++i) b[i] = (uint8_t) ((i * 2654435761u >> 24) % 61);   // small values: no overflow talk
    } else {
        std::memset(p, 0, bytes);   // zeros are fine for timing a dense GEMM (no data-dependent work)
    }
    m.unmap_data(p);
}

static const char* name(dt t) {
    switch (t) { case dt::f16: return "f16"; case dt::bf16: return "bf16"; case dt::f32: return "f32";
                 case dt::s8: return "s8"; case dt::u8: return "u8"; case dt::s32: return "s32"; default: return "?"; }
}

// one timed matmul: src dims a, weights dims b, dst dims c
static void run(engine& eng, stream& s, const char* what, memory::dims a, memory::dims b, memory::dims c,
                dt ta, dt tb, dt tc, int reps) {
    const auto tag = a.size() == 2 ? memory::format_tag::ab : memory::format_tag::abc;
    try {
        memory::desc amd(a, ta, tag), bmd(b, tb, tag), cmd(c, tc, tag);
        matmul::primitive_desc pd(eng, amd, bmd, cmd);
        matmul prim(pd);
        memory A(amd, eng), B(bmd, eng), C(cmd, eng);
        fill(A, ta); fill(B, tb);
        std::unordered_map<int, memory> args{{DNNL_ARG_SRC, A}, {DNNL_ARG_WEIGHTS, B}, {DNNL_ARG_DST, C}};
        prim.execute(s, args); s.wait();   // warm-up: kernel generation
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) prim.execute(s, args);
        s.wait();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
        // multiply-adds: M * K * N per batch entry, times 2 for the usual "FLOP" count
        const double batch = a.size() == 3 ? (double) a[0] : 1.0;
        const double M = (double) a[a.size() - 2], K = (double) a[a.size() - 1], N = (double) b[b.size() - 1];
        const double tops = 2.0 * batch * M * K * N / (ms * 1e-3) / 1e12;
        std::printf("%-34s %4s x %4s -> %4s  %9.2f ms  %7.1f T-ops/s   (%s)\n", what, name(ta), name(tb), name(tc), ms, tops,
                    pd.impl_info_str());
    } catch (const dnnl::error& e) {
        std::printf("%-34s %4s x %4s -> %4s  not supported: %s\n", what, name(ta), name(tb), name(tc), e.what());
    }
    std::fflush(stdout);
}

// softmax over the last axis of [heads, T, T] scores: attention's "turn scores into weights" step (one exp per score)
static void run_softmax(engine& eng, stream& s, long T, dt t, int reps) {
    try {
        memory::desc md({56, T, T}, t, memory::format_tag::abc);
        softmax_forward::primitive_desc pd(eng, prop_kind::forward_inference, algorithm::softmax_accurate, md, md, 2);
        softmax_forward prim(pd);
        memory A(md, eng), C(md, eng);
        fill(A, t);
        std::unordered_map<int, memory> args{{DNNL_ARG_SRC, A}, {DNNL_ARG_DST, C}};
        prim.execute(s, args); s.wait();
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) prim.execute(s, args);
        s.wait();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
        const double scores = 56.0 * T * T;
        std::printf("softmax 56 heads, tile %-10ld %4s                 %9.2f ms  %7.2f G scores/s  (%s)\n", T, name(t), ms,
                    scores / (ms * 1e-3) / 1e9, pd.impl_info_str());
    } catch (const dnnl::error& e) {
        std::printf("softmax tile %ld %s not supported: %s\n", T, name(t), e.what());
    }
    std::fflush(stdout);
}

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 5;
    engine eng(engine::kind::gpu, 0);
    stream s(eng);
    const long M = 16384;   // tokens in the linear tests (a step has 16k-52k; the rate does not depend on it)
    struct L { const char* n; long k, o; } lin[] = {
        {"linear qkv   5376 -> 21504", 5376, 21504}, {"linear fc1   5376 -> 28672", 5376, 28672},
        {"linear fc2  14336 ->  5376", 14336, 5376}, {"linear out   7168 ->  5376", 7168, 5376}};
    const bool attn_only = argc > 2 && std::string(argv[2]) == "attn";
    for (const auto& l : lin) {
        if (attn_only) break;
        run(eng, s, l.n, {M, l.k}, {l.k, l.o}, {M, l.o}, dt::f16, dt::f16, dt::f16, reps);
        run(eng, s, l.n, {M, l.k}, {l.k, l.o}, {M, l.o}, dt::bf16, dt::bf16, dt::bf16, reps);
        run(eng, s, l.n, {M, l.k}, {l.k, l.o}, {M, l.o}, dt::s8, dt::s8, dt::s32, reps);
        run(eng, s, l.n, {M, l.k}, {l.k, l.o}, {M, l.o}, dt::u8, dt::s8, dt::s32, reps);
        run(eng, s, l.n, {M, l.k}, {l.k, l.o}, {M, l.o}, dt::s8, dt::s8, dt::f16, reps);
    }
    for (long T : {1024L, 2048L, 4096L}) {
        const std::string w = "scores 56 heads, tile " + std::to_string(T);
        run(eng, s, w.c_str(), {56, T, 128}, {56, 128, T}, {56, T, T}, dt::f16, dt::f16, dt::f16, reps);
        run(eng, s, w.c_str(), {56, T, 128}, {56, 128, T}, {56, T, T}, dt::s8, dt::s8, dt::s32, reps);
        run(eng, s, w.c_str(), {56, T, 128}, {56, 128, T}, {56, T, T}, dt::s8, dt::s8, dt::f16, reps);
    }
    // attention's second half: probabilities [T x T] times values [T x 128]
    for (long T : {2048L, 4096L}) {
        const std::string w = "P.V   56 heads, tile " + std::to_string(T);
        run(eng, s, w.c_str(), {56, T, T}, {56, T, 128}, {56, T, 128}, dt::f16, dt::f16, dt::f16, reps);
    }
    for (long T : {2048L, 4096L}) { run_softmax(eng, s, T, dt::f16, reps); run_softmax(eng, s, T, dt::f32, reps); }
    // for scale: the same tiles' two matmuls, as scores per second (a score = one query-key pair of one head)
    std::printf("(a score costs 256 multiply-adds in Q.K^T and 256 in P.V: at 90 and 136 T-ops/s that is 0.35 and 0.53 G scores/s)\n");
    return 0;
}
