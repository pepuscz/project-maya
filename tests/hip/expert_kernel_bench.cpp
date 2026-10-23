// Decode-only, model-free expert benchmark. Run from the repository root:
// HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/rocm/lib build-multi/hip_expert_kernel_bench
// --parity-only also checks sparse plans, CPU contributions and shared experts.
// --sweep exposes rejected tuning variants; --shared times the fused Q6_K shared down.
// --fixture GU,DN (repeatable) times that gate/up and down type pair instead of the default 16,22 and 16,18 (Maya-M:
// 22,18); --host puts the expert pools in pinned, mapped host memory, as an integrated GPU's expert helper reads them;
// --exact fails any comparison that is not bit-identical (the default allows rel_L2 1e-6 / max_abs 1e-4); --k K1,K2,..
// times each route width (experts a call, <= 8; default 8) on its own bytes, over enough pools that one graph rotates
// at least 320 MiB of expert weights.
#include "strata/kernels/glm_expert_bench.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace gf = strata::kernels::glmf;
namespace bench = gf::hip_expert_bench;
using V = bench::Variant;
namespace {
constexpr int NE = 8, EMB = 4096, FF = 2048, POOLS = 5, CALLS = 60, ROUNDS = 7;
constexpr size_t ROTATE_MIN = size_t(320) << 20;   // --k: the expert bytes one timing graph rotates through, at least
bool g_exact = false;                              // --exact
void check(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s (no processes will be stopped)\n", what, hipGetErrorString(e));
        std::exit(2);
    }
}
struct Buffer {
    void* p = nullptr;      // what kernels and copies use (a mapped host buffer's device pointer)
    void* base = nullptr;   // a host buffer's own pointer, for hipHostFree
    bool host = false;
    explicit Buffer(size_t n, bool pinned_host = false) : host(pinned_host) {
        if (!host) { check(hipMalloc(&p, n), "hipMalloc"); return; }
        void* h = nullptr;
        check(hipHostMalloc(&h, n, hipHostMallocMapped | hipHostMallocPortable), "hipHostMalloc");
        check(hipHostGetDevicePointer(&p, h, 0), "host device pointer");
        base = h;
    }
    ~Buffer() { if (host) check(hipHostFree(base), "hipHostFree"); else if (p) check(hipFree(p), "hipFree"); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    template<class T> T* as() const { return static_cast<T*>(p); }
    void put(const void* src, size_t n) { check(hipMemcpy(p, src, n, hipMemcpyHostToDevice), "H2D"); }
    void zero(size_t n) { check(hipMemset(p, 0, n), "memset"); }
    std::vector<uint8_t> get(size_t n) const {
        std::vector<uint8_t> v(n);
        check(hipMemcpy(v.data(), p, n, hipMemcpyDeviceToHost), "D2H");
        return v;
    }
};
uint16_t half_bits(float f) {
    return std::bit_cast<uint16_t>(__float2half(f));
}
float half_float(uint16_t b) {
    return __half2float(std::bit_cast<__half>(b));
}
int block_bytes(int t) {
    switch (t) {
        case 14: return sizeof(block_q6_K);
        case 16: return sizeof(block_iq2_xxs);
        case 18: return sizeof(block_iq3_xxs);
        case 22: return sizeof(block_iq2_s);
        default: std::abort();
    }
}
void weights(uint8_t* dst, size_t bytes, int t, std::mt19937& rng) {
    const int bs = block_bytes(t);
    for (size_t i = 0; i < bytes; i += bs) {
        for (int j = 0; j < bs; ++j) dst[i + j] = rng() & 255;
        // Finite, signed scales; all other block bits are valid grid/scale/sign indices.
        const float magnitude = (1 + (rng() % 1024)) / 65536.f;
        const uint16_t d = half_bits((rng() & 1) ? magnitude : -magnitude);
        std::memcpy(dst + i + (t == 14 ? bs - 2 : 0), &d, 2);
    }
}
std::vector<uint8_t> activation(int n, std::mt19937& rng, bool zeros = false) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint8_t> v(size_t(n / 32) * sizeof(block_q8_1));
    for (int b = 0; b < n / 32; ++b) {
        float x[32], amax = 0, sum = 0;
        for (auto& f : x) { f = zeros ? 0.f : nd(rng); amax = std::max(amax, std::fabs(f)); sum += f; }
        const float d = amax / 127.f;
        uint8_t* p = v.data() + size_t(b) * sizeof(block_q8_1);
        const uint16_t dh = half_bits(d), sh = half_bits(sum);
        std::memcpy(p, &dh, 2); std::memcpy(p + 2, &sh, 2);
        for (int i = 0; i < 32; ++i) p[4 + i] = uint8_t(int8_t(d == 0 ? 0 : std::round(x[i] / d)));
    }
    return v;
}
std::vector<float> dequant(const std::vector<uint8_t>& v) {
    std::vector<float> out(v.size() / sizeof(block_q8_1) * 32);
    for (size_t b = 0; b < out.size() / 32; ++b) {
        uint16_t dh;
        std::memcpy(&dh, v.data() + b * sizeof(block_q8_1), 2);
        const float d = half_float(dh);
        for (int j = 0; j < 32; ++j) out[32 * b + j] = d * int8_t(v[b * sizeof(block_q8_1) + 4 + j]);
    }
    return out;
}
bool compare(const char* label, const std::vector<float>& a, const std::vector<float>& b, bool exact) {
    double max_abs = 0, num = 0, den = 0;
    bool finite = true;
    for (size_t i = 0; i < a.size(); ++i) {
        finite &= std::isfinite(a[i]) && std::isfinite(b[i]);
        const double diff = double(a[i]) - b[i];
        max_abs = std::max(max_abs, std::abs(diff)); num += diff * diff; den += double(a[i]) * a[i];
    }
    const double rel = std::sqrt(num / std::max(den, 1e-30));
    const bool ok = finite && (exact || (!g_exact && rel <= 1e-6 && max_abs <= 1e-4));
    std::printf("parity %-28s max_abs=%.3e rel_L2=%.3e bit_identical=%s %s\n",
                label, max_abs, rel, exact ? "yes" : "no", ok ? "PASS" : "FAIL");
    return ok;
}
const char* name(V v) {
    switch (v) {
        case V::baseline: return "baseline";
        case V::production: return "production";
        case V::signs: return "signs";
        case V::waves8: return "waves8";
        case V::rows2: return "rows2";
        case V::rows8: return "rows8";
        case V::global_grid: return "global_grid";
        case V::global_plain: return "global_plain";
        case V::global_waves8: return "global_waves8";
        case V::global_rows2: return "global_rows2";
        case V::global_rows8: return "global_rows8";
        case V::split_shared: return "split_shared";
        case V::direct_signs: return "direct_signs";
        case V::vector_load: return "vector_load";
        case V::occupancy8: return "occupancy8";
        case V::direct_lds: return "direct_lds";
        case V::direct_rows4: return "direct_rows4";
        case V::direct_lds_rows4: return "direct_lds_rows4";
    }
    std::abort();
}
struct Graph {
    hipGraph_t graph{};
    hipGraphExec_t exec{};
    Graph(hipStream_t s, const std::function<void(int)>& launch, int pools) {
        check(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal), "capture begin");
        for (int i = 0; i < CALLS; ++i) launch(i % pools);
        check(hipStreamEndCapture(s, &graph), "capture end");
        check(hipGraphInstantiateWithFlags(&exec, graph, 0), "graph instantiate");
    }
    ~Graph() { check(hipGraphExecDestroy(exec), "graph exec destroy"); check(hipGraphDestroy(graph), "graph destroy"); }
};
// Alternate the two graphs each round so clock/server-load drift affects both.
void timing(const char* kernel, V variant, size_t bytes, hipStream_t s, int pools,
            const std::function<void(int)>& old_launch, const std::function<void(int)>& new_launch) {
    Graph old(s, old_launch, pools), fresh(s, new_launch, pools);
    hipEvent_t start, end;
    check(hipEventCreate(&start), "event create"); check(hipEventCreate(&end), "event create");
    for (int j = 0; j < 3; ++j) {
        check(hipGraphLaunch(old.exec, s), "warmup old"); check(hipGraphLaunch(fresh.exec, s), "warmup new");
    }
    std::vector<double> a, b;
    for (int r = 0; r < ROUNDS; ++r) for (int j = 0; j < 2; ++j) {
        const bool baseline = (r + j) % 2 == 0;
        check(hipEventRecord(start, s), "event start");
        check(hipGraphLaunch(baseline ? old.exec : fresh.exec, s), "timed graph");
        check(hipEventRecord(end, s), "event end"); check(hipEventSynchronize(end), "event wait");
        float ms;
        check(hipEventElapsedTime(&ms, start, end), "elapsed");
        (baseline ? a : b).push_back(ms * 1000. / CALLS);
    }
    std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
    const double before = a[ROUNDS / 2], after = b[ROUNDS / 2];
    std::printf("bench %-11s %-12s old=%.3f us %.2f GB/s new=%.3f us %.2f GB/s speedup=%.3fx latency_reduction=%.1f%%\n",
                kernel, name(variant), before, bytes / before / 1000., after, bytes / after / 1000.,
                before / after, 100 * (1 - after / before));
    check(hipEventDestroy(start), "event destroy"); check(hipEventDestroy(end), "event destroy");
}
bool run(int gu_type, int dn_type, bool parity_only, bool shared_mode, bool sweep, hipStream_t s, bool host = false,
         const std::vector<int>& ks = {}) {
    const size_t gu_bytes = size_t(2) * FF * (EMB / 256) * block_bytes(gu_type);
    const size_t dn_bytes = size_t(EMB) * (FF / 256) * block_bytes(dn_type);
    const size_t blob_bytes = gu_bytes + dn_bytes;
    const std::vector<int> widths = ks.empty() ? std::vector<int>{NE} : ks;
    int pools = parity_only ? 1 : POOLS;
    if (!parity_only && !ks.empty()) {   // the narrowest route width still rotates ROTATE_MIN bytes
        const size_t per_pool = size_t(*std::min_element(widths.begin(), widths.end())) * blob_bytes;
        pools = std::max(POOLS, int((ROTATE_MIN + per_pool - 1) / per_pool));
    }
    const size_t hbytes = size_t(NE) * (FF / 32) * sizeof(block_q8_1);
    std::printf("fixture gu=%d down=%d experts=%d n_embd=%d n_ff=%d pools=%d weights=%.1f MiB shared=%s experts in %s\n",
                gu_type, dn_type, NE, EMB, FF, pools, pools * NE * blob_bytes / 1048576., shared_mode ? "Q6_K 4096x2048" : "off",
                host ? "pinned host memory" : "device memory");
    Buffer blobs(pools * NE * blob_bytes, host), ptrs(pools * NE * 8), w(NE * 4), flag(4), cpu(EMB * 4);
    Buffer xq(EMB / 32 * sizeof(block_q8_1)), hq(hbytes), old_hq(hbytes), out(EMB * 4), old_out(EMB * 4);
    Buffer shared(EMB * 4), sh_weights(size_t(EMB) * (FF / 256) * sizeof(block_q6_K));
    Buffer sh_hq(FF / 32 * sizeof(block_q8_1)), sh_result(EMB * 4), sh_old(EMB * 4);
    std::mt19937 rng(0x5eed + dn_type);
    std::vector<uint8_t> raw(blob_bytes);
    std::vector<unsigned long long> pointers(pools * NE);
    for (int p = 0; p < pools * NE; ++p) {
        weights(raw.data(), gu_bytes, gu_type, rng); weights(raw.data() + gu_bytes, dn_bytes, dn_type, rng);
        void* dst = blobs.as<uint8_t>() + p * blob_bytes;
        check(hipMemcpy(dst, raw.data(), raw.size(), hipMemcpyHostToDevice), "expert upload");
        pointers[p] = reinterpret_cast<unsigned long long>(dst);
    }
    ptrs.put(pointers.data(), pointers.size() * 8);
    float weights_host[NE];
    for (int i = 0; i < NE; ++i) weights_host[i] = float(i + 1) / 36.f;
    w.put(weights_host, sizeof(weights_host)); flag.zero(4);
    auto x = activation(EMB, rng); xq.put(x.data(), x.size());
    auto sh_x = activation(FF, rng); sh_hq.put(sh_x.data(), sh_x.size());
    std::vector<uint8_t> sw(size_t(EMB) * (FF / 256) * sizeof(block_q6_K));
    weights(sw.data(), sw.size(), 14, rng); sh_weights.put(sw.data(), sw.size());
    std::vector<float> extra(EMB);
    for (int i = 0; i < EMB; ++i) extra[i] = float((i % 31) - 15) / 32.f;
    cpu.put(extra.data(), EMB * 4); shared.put(extra.data(), EMB * 4);
    std::vector<gf::MoeDev> plans(pools);
    for (int p = 0; p < pools; ++p) {
        plans[p].plan_ptr = ptrs.as<unsigned long long>() + p * NE;
        plans[p].plan_w = w.as<float>(); plans[p].cpu_flag = flag.as<int>(); plans[p].cpu_part = cpu.as<float>();
    }
    const std::vector<V> gu_variants = sweep ? std::vector<V>{
        V::signs, V::waves8, V::global_grid, V::global_plain, V::global_waves8, V::split_shared,
        V::direct_signs, V::vector_load, V::occupancy8, V::direct_lds, V::production} : std::vector<V>{V::production};
    const std::vector<V> dn_variants = sweep ? std::vector<V>{
        V::signs, V::rows2, V::rows8, V::global_grid, V::global_plain, V::global_rows2, V::global_rows8,
        V::direct_signs, V::vector_load, V::direct_lds, V::direct_rows4, V::direct_lds_rows4,
        V::production} : std::vector<V>{V::production};
    bool ok = true;
    // Include k<8, absent experts, CPU/shared branches, zero activations and unclamped SwiGLU.
    const int cases = parity_only ? 6 : 1;
    for (int c = 0; c < cases; ++c) {
        const int k = c == 1 ? 1 : c == 2 ? 3 : NE;
        const bool use_shared = c == 3 || shared_mode;
        const int cpu_flag = c == 3;
        flag.put(&cpu_flag, 4);
        auto plan = pointers;
        if (c == 2 || c == 3) { plan[1] = 0; plan[6] = 0; }
        ptrs.put(plan.data(), plan.size() * 8);
        x = activation(EMB, rng, c == 4); xq.put(x.data(), x.size());
        const float limit = c == 5 ? 1e9f : 10.f;
        auto gu = [&](V v, int pool, void* dst, float* sh_dst) {
            bench::gate_up(v, gu_type, plans[pool], k, EMB, FF, limit, xq.p, dst,
                          use_shared ? sh_weights.p : nullptr, 14, sh_hq.p, FF, sh_dst, s);
        };
        auto dn = [&](V v, int pool, const void* h, float* dst) {
            bench::down(v, dn_type, plans[pool], k, EMB, FF, gu_bytes, h,
                        use_shared ? shared.as<float>() : nullptr, dst, s);
        };
        old_hq.zero(hbytes); gu(V::baseline, 0, old_hq.p, sh_old.as<float>());
        check(hipStreamSynchronize(s), "baseline gu sync");
        auto ref_h = old_hq.get(hbytes);
        auto ref_sh = use_shared ? sh_old.get(EMB * 4) : std::vector<uint8_t>{};
        for (V v : gu_variants) {
            hq.zero(hbytes); gu(v, 0, hq.p, sh_result.as<float>()); check(hipStreamSynchronize(s), "gu sync");
            auto got = hq.get(hbytes);
            const std::string label = "gu" + std::to_string(gu_type) + " " + name(v) + " case" + std::to_string(c);
            ok &= compare(label.c_str(), dequant(ref_h), dequant(got), ref_h == got);
            if (use_shared) {
                const auto got_sh = sh_result.get(EMB * 4);
                std::vector<float> a(EMB), b(EMB);
                std::memcpy(a.data(), ref_sh.data(), EMB * 4); std::memcpy(b.data(), got_sh.data(), EMB * 4);
                ok &= compare((label + " shared down").c_str(), a, b, ref_sh == got_sh);   // (the variant and case it follows)
            }
        }
        // Down uses the identical baseline hq so gate/up differences cannot mask errors.
        dn(V::baseline, 0, old_hq.p, old_out.as<float>()); check(hipStreamSynchronize(s), "baseline dn sync");
        auto ref_d = old_out.get(EMB * 4);
        for (V v : dn_variants) {
            dn(v, 0, old_hq.p, out.as<float>()); check(hipStreamSynchronize(s), "dn sync");
            auto got = out.get(EMB * 4);
            std::vector<float> a(EMB), b(EMB);
            std::memcpy(a.data(), ref_d.data(), EMB * 4); std::memcpy(b.data(), got.data(), EMB * 4);
            const std::string label = "down" + std::to_string(dn_type) + " " + name(v) + " case" + std::to_string(c);
            ok &= compare(label.c_str(), a, b, ref_d == got);
        }
        if (!parity_only) {
            // Independent random h activations in all pools; bytes read, not FLOPs, define GB/s.
            auto h = activation(NE * FF, rng); old_hq.put(h.data(), h.size());
            for (int kt : widths) {
                const std::string kk = ks.empty() ? "" : " k" + std::to_string(kt);
                const std::string glabel = "gate_up<" + std::to_string(gu_type) + ">" + kk;
                const std::string dlabel = "down<" + std::to_string(dn_type) + ">" + kk;
                if (!ks.empty())
                    std::printf("timing k=%d experts a call: %d pools, %.1f MiB of experts rotated a graph\n",
                                kt, pools, pools * kt * blob_bytes / 1048576.);
                auto gk = [&](V v, int p) {
                    bench::gate_up(v, gu_type, plans[p], kt, EMB, FF, limit, xq.p, hq.p,
                                   use_shared ? sh_weights.p : nullptr, 14, sh_hq.p, FF, sh_result.as<float>(), s);
                };
                auto dk = [&](V v, int p) {
                    bench::down(v, dn_type, plans[p], kt, EMB, FF, gu_bytes, old_hq.p,
                                use_shared ? shared.as<float>() : nullptr, out.as<float>(), s);
                };
                // this width's own parity: every variant against the baseline at k = kt (--exact: bit for bit)
                hq.zero(hbytes); gk(V::baseline, 0); check(hipStreamSynchronize(s), "k gu sync");
                const auto ref_k = hq.get(hbytes);
                dk(V::baseline, 0); check(hipStreamSynchronize(s), "k dn sync");
                const auto ref_kd = out.get(EMB * 4);
                for (V v : gu_variants) {
                    hq.zero(hbytes); gk(v, 0); check(hipStreamSynchronize(s), "k gu sync");
                    const auto got = hq.get(hbytes);
                    ok &= compare((glabel + " " + name(v)).c_str(), dequant(ref_k), dequant(got), ref_k == got);
                }
                for (V v : dn_variants) {
                    dk(v, 0); check(hipStreamSynchronize(s), "k dn sync");
                    const auto got = out.get(EMB * 4);
                    std::vector<float> a(EMB), b(EMB);
                    std::memcpy(a.data(), ref_kd.data(), EMB * 4); std::memcpy(b.data(), got.data(), EMB * 4);
                    ok &= compare((dlabel + " " + name(v)).c_str(), a, b, ref_kd == got);
                }
                for (V v : gu_variants)
                    timing(glabel.c_str(), v, kt * gu_bytes + (use_shared ? sw.size() : 0), s, pools,
                           [&](int p) { gk(V::baseline, p); }, [&](int p) { gk(v, p); });
                for (V v : dn_variants)
                    timing(dlabel.c_str(), v, kt * dn_bytes, s, pools,
                           [&](int p) { dk(V::baseline, p); }, [&](int p) { dk(v, p); });
            }
        }
    }
    return ok;
}
}
int main(int argc, char** argv) {
    bool parity_only = false, shared_mode = false, sweep = false, host = false;
    std::vector<std::pair<int, int>> fixtures;
    std::vector<int> ks;
    auto parse_k = [&ks](const char* list) {
        std::string t(list);
        for (size_t at = 0; at <= t.size();) {
            const size_t comma = std::min(t.find(',', at), t.size());
            const std::string one = t.substr(at, comma - at);
            if (one.empty() || one.find_first_not_of("0123456789") != std::string::npos) return false;
            const int k = std::atoi(one.c_str());
            if (k < 1 || k > NE) return false;
            ks.push_back(k);
            at = comma + 1;
        }
        return !ks.empty();
    };
    auto known = [](int t) { return t == 16 || t == 18 || t == 22; };
    for (int i = 1; i < argc; ++i) {
        int gu = 0, dn = 0;
        if (std::string(argv[i]) == "--parity-only") parity_only = true;
        else if (std::string(argv[i]) == "--shared") shared_mode = true;
        else if (std::string(argv[i]) == "--sweep") sweep = true;
        else if (std::string(argv[i]) == "--host") host = true;
        else if (std::string(argv[i]) == "--exact") g_exact = true;
        else if (std::string(argv[i]) == "--k" && i + 1 < argc && parse_k(argv[i + 1])) ++i;
        else if (std::string(argv[i]) == "--fixture" && i + 1 < argc && std::sscanf(argv[i + 1], "%d,%d", &gu, &dn) == 2 &&
                 known(gu) && known(dn)) { fixtures.emplace_back(gu, dn); ++i; }
        else {
            std::fprintf(stderr, "usage: %s [--parity-only] [--shared] [--sweep] [--host] [--exact] [--k K1,K2,..] "
                                 "[--fixture GU,DN]... (types 16, 18, 22; k 1..8)\n", argv[0]);
            return 2;
        }
    }
    check(hipSetDevice(0), "GPU 0");
    hipDeviceProp_t prop;
    check(hipGetDeviceProperties(&prop, 0), "device properties");
    size_t free_bytes, total_bytes;
    check(hipMemGetInfo(&free_bytes, &total_bytes), "VRAM info");
    std::printf("GPU 0: %s (%s), free=%.1f MiB, graph_calls=%d rounds=%d (median)\n",
                prop.name, prop.gcnArchName, free_bytes / 1048576., CALLS, ROUNDS);
    hipStream_t stream;
    check(hipStreamCreate(&stream), "stream create");
    bool ok = true;
    if (!fixtures.empty()) {
        for (auto [gu, dn] : fixtures) ok &= run(gu, dn, parity_only, shared_mode, sweep, stream, host, ks);
    } else {
        ok &= run(16, 22, parity_only, shared_mode, sweep, stream, host, ks);
        ok &= run(16, 18, parity_only, shared_mode, sweep, stream, host, ks);
        if (parity_only) { ok &= run(18, 16, true, shared_mode, sweep, stream, host); ok &= run(22, 22, true, shared_mode, sweep, stream, host); }
    }
    check(hipStreamDestroy(stream), "stream destroy");
    std::printf("hip_expert_kernel_bench: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
