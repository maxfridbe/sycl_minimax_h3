// sdpa_probe.cpp - which form of the attention pattern does oneDNN's graph interface run as its FUSED kernel?
//
// oneDNN has a fused attention kernel (scores, softmax and the weighted sum over tiles, the score table never written
// to memory), reached only through the graph interface, and only when the graph matches what it expects. When it
// does not match, the same call runs four separate steps with the whole [heads, rows, keys] score table in device
// memory - and nothing reports which happened. This probe builds the pattern in several variants at a SMALL size,
// times each, and prints the rate in scores per second: the fused kernel is several times faster than the fallback.
// Run it with ONEDNN_VERBOSE=1 to also see the implementation names.
//
//   sdpa_probe [rows=1024] [keys=2048] [reps=3] [nofallback]     the fallback's score table: 56 * rows * keys * 2 bytes
//
// Keep rows * keys small (the default is 0.23 GiB). Never run this at a real sequence length: see h3s_attention.
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_graph.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace dnnl::graph;
using lt = logical_tensor;

struct Variant {
    const char* name;
    lt::data_type dt;
    bool multiply;        // Multiply by 1/sqrt(D) instead of Divide by sqrt(D)
    bool const_scale;     // the scale marked as a constant
    bool mode;            // SoftMax mode = "inf_as_zero"
    bool mask;            // an additive mask [1, 1, 1, keys] of zeros before the softmax
    bool host_scalar;     // the scale as a float32 scalar held on the host (how PyTorch passes it)
    bool f32_inter;       // the tensors between the two multiplies declared float32 (q, k, v stay 16-bit)
};

static void run(const Variant& va, dnnl::engine& eng, dnnl::stream& strm, long H, long R, long S, long D, int reps) {
    try {
        const auto strided = lt::layout_type::strided;
        size_t id = 0;
        lt q(id++, va.dt, lt::dims {1, H, R, D}, strided), k(id++, va.dt, lt::dims {1, H, S, D}, strided);
        lt scale = va.host_scalar
                ? lt(id++, lt::data_type::f32, lt::dims {}, strided, lt::property_type::host_scalar)
                : lt(id++, va.dt, lt::dims {1}, strided, va.const_scale ? lt::property_type::constant : lt::property_type::undef);
        lt v(id++, va.dt, lt::dims {1, H, S, D}, strided), mask(id++, va.dt, lt::dims {1, 1, 1, S}, strided);
        const auto it = va.f32_inter ? lt::data_type::f32 : va.dt;
        lt score(id++, it, lt::dims {1, H, R, S}, strided), scaled(id++, it, lt::dims {1, H, R, S}, strided);
        lt masked(id++, it, lt::dims {1, H, R, S}, strided), probs(id++, va.dt, lt::dims {1, H, R, S}, strided);   // the weights are 16-bit again
        lt out(id++, va.dt, lt::dims {1, H, R, D}, strided);
        op bmm1(id++, op::kind::MatMul, {q, k}, {score}, "scores");
        bmm1.set_attr<bool>(op::attr::transpose_b, true);
        op sc(id++, va.multiply ? op::kind::Multiply : op::kind::Divide, {score, scale}, {scaled}, "scale");
        op add(id++, op::kind::Add, {scaled, mask}, {masked}, "mask");
        op sm(id++, op::kind::SoftMax, {va.mask ? masked : scaled}, {probs}, "softmax");
        sm.set_attr<int64_t>(op::attr::axis, -1);
        if (va.mode) sm.set_attr<std::string>(op::attr::mode, "inf_as_zero");
        op bmm2(id++, op::kind::MatMul, {probs, v}, {out}, "values");
        graph g(dnnl::engine::kind::gpu);
        g.add_op(bmm1); g.add_op(sc);
        if (va.mask) g.add_op(add);
        g.add_op(sm); g.add_op(bmm2);
        g.finalize();
        auto parts = g.get_partitions();
        if (parts.size() != 1 || !parts[0].is_supported()) {
            std::printf("%-44s %zu partitions%s - not taken as one kernel\n", va.name, parts.size(),
                        parts.empty() || parts[0].is_supported() ? "" : ", unsupported");
            return;
        }
        const std::vector<lt> mine = {q, k, scale, v, mask};
        std::vector<lt> in;
        for (const auto& port : parts[0].get_input_ports())
            for (const auto& m : mine)
                if (m.get_id() == port.get_id()) in.push_back(m);
        auto cp = parts[0].compile(in, {out}, eng);
        std::vector<tensor> tin;
        float scale_value = va.multiply ? 1.0f / std::sqrt((float) D) : std::sqrt((float) D);
        for (const auto& l : in) {                                  // device memory, contents irrelevant (zeros or junk)
            if (va.host_scalar && l.get_id() == scale.get_id()) tin.push_back(tensor::make_scalar_tensor(l, &scale_value));
            else tin.emplace_back(l, eng);
        }
        tensor tout(cp.query_logical_tensor(out.get_id()), eng);
        cp.execute(strm, tin, {tout});
        strm.wait();
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) cp.execute(strm, tin, {tout});
        strm.wait();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
        std::printf("%-44s %8.2f ms  %7.1f G scores/s\n", va.name, ms, (double) H * R * S / (ms * 1e-3) / 1e9);
    } catch (const std::exception& e) {
        std::printf("%-44s failed: %s\n", va.name, e.what());
    }
    std::fflush(stdout);
}

int main(int argc, char** argv) {
    const long R = argc > 1 ? std::atol(argv[1]) : 1024, S = argc > 2 ? std::atol(argv[2]) : 2048;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 3;
    // a 4th argument: ask a patched oneDNN (container/onednn-sdpa-no-fallback.patch) to refuse instead of falling back
    if (argc > 4) setenv("_ONEDNN_GRAPH_SDPA_NO_FALLBACK", "1", 1);
    const long H = 56, D = 128;
    if ((double) H * R * S * 2 > 1.6e9) {
        std::printf("refusing: a score table of %.1f GiB (keep 56 * rows * keys * 2 bytes under 1.5 GiB)\n", (double) H * R * S * 2 / (1 << 30));
        return 2;
    }
    dnnl::engine eng(dnnl::engine::kind::gpu, 0);
    dnnl::stream strm(eng);
    std::printf("attention %ld heads x %ld query rows x %ld keys, head size %ld; a fallback's score table: %.2f GiB\n", H, R, S, D,
                (double) H * R * S * 2 / (1 << 30));
    const auto f16 = lt::data_type::f16, bf16 = lt::data_type::bf16;
    const Variant vs[] = {
        {"f16  divide", f16, false, false, false, false, false, false},
        {"f16  divide, softmax mode", f16, false, false, true, false, false, false},
        {"f16  divide, constant scale", f16, false, true, false, false, false, false},
        {"f16  divide, constant scale, mode", f16, false, true, true, false, false, false},
        {"f16  multiply", f16, true, false, false, false, false, false},
        {"f16  multiply, softmax mode", f16, true, false, true, false, false, false},
        {"f16  divide, mask", f16, false, false, false, true, false, false},
        {"f16  divide, mask, softmax mode", f16, false, false, true, true, false, false},
        {"bf16 divide", bf16, false, false, false, false, false, false},
        {"bf16 divide, softmax mode", bf16, false, false, true, false, false, false},
        {"bf16 divide, mask, softmax mode", bf16, false, false, true, true, false, false},
        {"f16  multiply, host scalar", f16, true, false, false, false, true, false},
        {"f16  multiply, host scalar, softmax mode", f16, true, false, true, false, true, false},
        {"f16  divide, host scalar, softmax mode", f16, false, false, true, false, true, false},
        {"bf16 multiply, host scalar, softmax mode", bf16, true, false, true, false, true, false},
        {"f16  multiply, host scalar, mode, f32 between", f16, true, false, true, false, true, true},
        {"bf16 multiply, host scalar, mode, f32 between", bf16, true, false, true, false, true, true},
        {"f16  divide, mode, f32 between", f16, false, false, true, false, false, true},
        {"f16  multiply, mode, f32 between", f16, true, false, true, false, false, true},
        {"f16  multiply, f32 between (no mode)", f16, true, false, false, false, true, true},
    };
    for (const auto& va : vs) run(va, eng, strm, H, R, S, D, reps);
    return 0;
}
