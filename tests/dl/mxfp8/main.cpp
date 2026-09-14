// VX_MXFP8 acceptance test (plan P6-02 Q4), SimX-verified.
//
// 1. Host self-check of the shared E8M0 helper against a long-double
//    ground truth (sufficiency + minimality of the scale, plus the
//    ceil(log2(amax/448)) formula where the clamp does not engage).
// 2. Pack bit-exactness vs a host float mirror that calls the same shared
//    helpers in the kernel's op order (weights, activations, edge tensor
//    with zero groups / subnormals / exact 448 boundaries / K tail).
// 3. Dequant: bit-exact vs the host mirror, plus the honest per-group
//    absolute bound |dequant(w) - w| <= 16 * scale (E4M3 half-ULP at the
//    top binade [256,448]; the task's |err| <= scale is NOT achievable
//    for a 3-bit mantissa - documented in args.h and below).
// 4. GEMM vs a double reference of the quantized values, rel <= 5e-3.

#include <vortex2.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "mxfp8_args.h"

#define CHECK(expr) do { \
    if ((expr) != 0) { \
        fprintf(stderr, "FAILED at %s:%d: '%s'\n", __FILE__, __LINE__, #expr); \
        return 1; \
    } \
} while (0)

static unsigned int g_seed = 20260914;
static float frand() {
    g_seed = g_seed * 1103515245u + 12345u;
    return ((float)(g_seed >> 8 & 0xffffu) / 65536.0f - 0.5f) * 2.0f;
}

struct DevBuf {
    vx_buffer_h h = nullptr;
    uint64_t addr = 0;
};

static DevBuf make_buf(vx_device_h dev, size_t bytes) {
    DevBuf b;
    vx_buffer_create(dev, bytes, 0, &b.h);
    vx_buffer_address(b.h, &b.addr);
    return b;
}

static void upload(vx_queue_h q, const DevBuf& b, const void* src,
                   size_t bytes) {
    vx_event_h ev = nullptr;
    vx_enqueue_write(q, b.h, 0, src, bytes, 0, nullptr, &ev);
    vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE);
    vx_event_release(ev);
}

template <typename T>
static void download(vx_queue_h q, std::vector<T>& dst, const DevBuf& b) {
    vx_event_h ev = nullptr;
    vx_enqueue_read(q, dst.data(), b.h, 0, dst.size() * sizeof(T), 0, nullptr,
                    &ev);
    vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE);
    vx_event_release(ev);
}

// ---------------------------------------------------------------------------
// 1. E8M0 helper self-check vs long-double ground truth
// ---------------------------------------------------------------------------

static int check_e8m0_helper() {
    std::vector<float> cases;
    // exact boundaries 448*2^c and their nextafter neighbors, plus a
    // log-uniform random sweep over the full FP32 exponent range
    for (int c = -130; c <= 110; ++c) {
        float b = ldexpf(448.0f, c);
        if (b == 0.0f || std::isinf(b)) continue;
        cases.push_back(b);
        cases.push_back(nextafterf(b, 0.0f));
        cases.push_back(nextafterf(b, INFINITY));
    }
    for (int i = 0; i < 20000; ++i) {
        int e = -45 + (int)(frand() * 0.5f + 0.5f) * 83;  // [-45, 38]
        float m = 1.0f + frand() * 0.5f + 0.5f;
        cases.push_back(ldexpf(m, e));
    }
    float edges[] = {0.0f,  -0.0f, 1.0f,  0.875f, 1.75f,      3.5f,
                     448.0f, FLT_MIN, FLT_MAX, 1e-40f, 1e-30f, 1e-35f,
                     8.8e-37f, 1e30f, 1e38f, 0.5f};
    for (float e : edges) cases.push_back(e);

    uint32_t suff_bad = 0, minimal_bad = 0, formula_bad = 0;
    for (float a : cases) {
        uint8_t x = vx_mxfp8_e8m0(a);
        long double amax = fabsl((long double)a);
        if (amax == 0.0L) {
            if (x != 127) ++suff_bad;
            continue;
        }
        long double s = scalbnl(448.0L, (int)x - 127);
        if (!(s >= amax)) ++suff_bad;                 // amax <= 448*s
        if (x > 0) {
            long double s2 = scalbnl(448.0L, (int)x - 128);
            if (!(s2 < amax)) ++minimal_bad;          // minimality
        }
        int32_t want = 127 + (int32_t)ceill(log2l(amax / 448.0L));
        if (want > 254) want = 254;
        if (want < 0) want = 0;
        if ((int32_t)x != want) ++formula_bad;        // ceil(log2) formula
    }
    printf("e8m0_helper:  checked=%zu sufficiency_bad=%u minimality_bad=%u "
           "formula_bad=%u\n", cases.size(), suff_bad, minimal_bad,
           formula_bad);
    return (suff_bad || minimal_bad || formula_bad) ? 1 : 0;
}

// ---------------------------------------------------------------------------
// CPU mirror of the pack kernels (same shared helpers, same op order)
// ---------------------------------------------------------------------------

static void cpu_pack(const std::vector<float>& src, uint32_t rows,
                     uint32_t cols, uint32_t group,
                     std::vector<uint8_t>& codes, std::vector<uint8_t>& scales) {
    const uint32_t groups = (cols + group - 1) / group;
    codes.assign((size_t)rows * cols, 0);
    scales.assign((size_t)rows * groups, 0);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t g = 0; g < groups; ++g) {
            const uint32_t c0 = g * group;
            const uint32_t c1 = c0 + group < cols ? c0 + group : cols;
            float amax = 0.0f;
            for (uint32_t c = c0; c < c1; ++c) {
                float a = fabsf(src[(size_t)r * cols + c]);
                if (a > amax) amax = a;
            }
            scales[(size_t)r * groups + g] = vx_mxfp8_e8m0(amax);
        }
    }
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            codes[(size_t)r * cols + c] = vx_mxfp8_quant(
                src[(size_t)r * cols + c],
                scales[(size_t)r * groups + c / group]);
        }
    }
}

// Pack on device, download, compare bit-exact vs the mirror. Returns 0 on
// match. Prints the per-case line.
static int run_pack_case(const char* tag, vx_queue_h q, vx_device_h dev,
                         const std::vector<float>& src, uint32_t rows,
                         uint32_t cols, uint32_t group,
                         std::vector<uint8_t>& codes,
                         std::vector<uint8_t>& scales) {
    std::vector<uint8_t> ref_codes, ref_scales;
    cpu_pack(src, rows, cols, group, ref_codes, ref_scales);

    DevBuf bsrc = make_buf(dev, src.size() * 4);
    DevBuf bcodes = make_buf(dev, ref_codes.size());
    DevBuf bscales = make_buf(dev, ref_scales.size());
    upload(q, bsrc, src.data(), src.size() * 4);
    CHECK(vx_mxfp8_pack(q, bsrc.addr, bcodes.addr, bscales.addr, rows, cols,
                        group));
    CHECK(vx_queue_flush(q));
    codes.resize(ref_codes.size());
    scales.resize(ref_scales.size());
    download(q, codes, bcodes);
    download(q, scales, bscales);

    uint32_t code_bad = 0, scale_bad = 0;
    for (size_t i = 0; i < codes.size(); ++i)
        if (codes[i] != ref_codes[i]) ++code_bad;
    for (size_t i = 0; i < scales.size(); ++i)
        if (scales[i] != ref_scales[i]) ++scale_bad;
    printf("pack_%-7s code_mismatch=%u scale_mismatch=%u (of %zu/%zu)\n", tag,
           code_bad, scale_bad, codes.size(), scales.size());
    vx_buffer_release(bsrc.h);
    vx_buffer_release(bcodes.h);
    vx_buffer_release(bscales.h);
    return (code_bad || scale_bad) ? 1 : 0;
}

// Dequant on device: bit-exact vs mirror + |err| <= 16*scale vs the source.
static int run_dequant_case(const char* tag, vx_queue_h q, vx_device_h dev,
                            const std::vector<float>& src, uint32_t rows,
                            uint32_t cols, uint32_t group,
                            const std::vector<uint8_t>& codes,
                            const std::vector<uint8_t>& scales) {
    const uint32_t groups = (cols + group - 1) / group;
    DevBuf bcodes = make_buf(dev, codes.size());
    DevBuf bscales = make_buf(dev, scales.size());
    DevBuf bout = make_buf(dev, src.size() * 4);
    upload(q, bcodes, codes.data(), codes.size());
    upload(q, bscales, scales.data(), scales.size());
    CHECK(vx_mxfp8_unpack(q, bcodes.addr, bscales.addr, bout.addr, rows, cols,
                           group));
    CHECK(vx_queue_flush(q));
    std::vector<float> got(src.size());
    download(q, got, bout);

    uint32_t bit_bad = 0, abs_bad = 0;
    double max_ratio = 0.0, mean_abs = 0.0;
    for (uint32_t i = 0; i < rows * cols; ++i) {
        const uint32_t r = i / cols, c = i % cols;
        const uint8_t x = scales[(size_t)r * groups + c / group];
        const float expect = vx_mxfp8_dequant(codes[i], x);
        if (got[i] != expect) ++bit_bad;
        // Honest bound: |dequant(w)-w| <= 16*scale exactly (E4M3 RNE
        // half-ULP at the top binade [256,448]; pack guarantees |w|/s<=448).
        // A 3-bit mantissa cannot meet |err| <= 1*scale at the top of the
        // range - see args.h.
        const double s = scalbn(1.0, (int)x - 127);
        const double err = fabs((double)got[i] - (double)src[i]);
        mean_abs += err;
        if (err / s > max_ratio) max_ratio = err / s;
        if (err > 16.0 * s) ++abs_bad;
    }
    mean_abs /= (double)src.size();
    printf("dequant_%-4s bitexact_bad=%u abs_violations=%u "
           "max_err/s=%.2f mean_abs_err=%.2e\n", tag, bit_bad, abs_bad,
           max_ratio, mean_abs);
    vx_buffer_release(bcodes.h);
    vx_buffer_release(bscales.h);
    vx_buffer_release(bout.h);
    return (bit_bad || abs_bad) ? 1 : 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    const char* vxbin = (argc > 1) ? argv[1] : "mxfp8.vxbin";

    int failures = check_e8m0_helper();

    vx_device_h dev = nullptr;
    CHECK(vx_device_open(0, &dev));
    vx_queue_h q = nullptr;
    CHECK(vx_queue_create(dev, nullptr, &q));
    if (vx_mxfp8_init(dev, vxbin) != VX_MXFP8_OK) {
        fprintf(stderr, "FAILED: vx_mxfp8_init(%s)\n", vxbin);
        return 1;
    }

    const uint32_t M = 40, N = 33, K = 97, GROUP = 32;  // 97 = 3*32 + 1 tail
    const uint32_t groups = (K + GROUP - 1) / GROUP;

    // ---- tensors --------------------------------------------------------
    std::vector<float> w((size_t)N * K), act((size_t)M * K);
    for (auto& x : w) x = frand();
    for (auto& x : act) x = frand();
    for (uint32_t c = 0; c < 32; ++c) w[(size_t)5 * K + c] = 0.0f;  // zero group
    for (uint32_t c = 0; c < 32; ++c) act[(size_t)7 * K + 32 + c] = 0.0f;
    w[(size_t)3 * K + K - 1] = 0.0f;   // lone tail-group element (group of 1)

    // Edge tensor: exact 448 boundary, all-zero row w/ -0.0, subnormal and
    // deep-tiny amax (E8M0 clamp path), huge amax, random body, K tail.
    const uint32_t E = 3;
    std::vector<float> edge((size_t)E * K, 0.0f);
    for (uint32_t c = 0; c < K; ++c) edge[c] = (c & 1) ? -0.875f : 0.875f;
    edge[(size_t)1 * K + 3] = -0.0f;
    edge[(size_t)2 * K + 0] = 1e-40f;
    edge[(size_t)2 * K + 1] = -3.0e30f;
    edge[(size_t)2 * K + 2] = 8.8e-37f;
    edge[(size_t)2 * K + 3] = 1e-35f;
    edge[(size_t)2 * K + 4] = 4.5f;
    edge[(size_t)2 * K + 5] = nextafterf(0.875f, 1.0f);
    for (uint32_t c = 6; c < K - 1; ++c) edge[(size_t)2 * K + c] = frand();
    edge[(size_t)2 * K + K - 1] = 0.5f;  // tail group of one element

    // ---- pack (bit-exact) ------------------------------------------------
    std::vector<uint8_t> w_codes, w_scales, a_codes, a_scales, e_codes, e_scales;
    failures += run_pack_case("w", q, dev, w, N, K, GROUP, w_codes, w_scales);
    failures += run_pack_case("a", q, dev, act, M, K, GROUP, a_codes, a_scales);
    failures += run_pack_case("edge", q, dev, edge, E, K, GROUP, e_codes, e_scales);

    // ---- dequant (bit-exact + absolute bound) ----------------------------
    failures += run_dequant_case("w", q, dev, w, N, K, GROUP, w_codes, w_scales);
    failures += run_dequant_case("edge", q, dev, edge, E, K, GROUP, e_codes,
                                 e_scales);

    // ---- GEMM vs double reference of the quantized values ----------------
    std::vector<double> ref((size_t)M * N, 0.0);
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t nn = 0; nn < N; ++nn) {
            double acc = 0.0;
            for (uint32_t kk = 0; kk < K; ++kk) {
                const double av = (double)vx_mxfp8_dequant(
                    a_codes[(size_t)m * K + kk],
                    a_scales[(size_t)m * groups + kk / GROUP]);
                const double wv = (double)vx_mxfp8_dequant(
                    w_codes[(size_t)nn * K + kk],
                    w_scales[(size_t)nn * groups + kk / GROUP]);
                acc += av * wv;
            }
            ref[(size_t)m * N + nn] = acc;
        }
    }

    {
        DevBuf bac = make_buf(dev, a_codes.size());
        DevBuf bas = make_buf(dev, a_scales.size());
        DevBuf bwc = make_buf(dev, w_codes.size());
        DevBuf bws = make_buf(dev, w_scales.size());
        DevBuf bo = make_buf(dev, (size_t)M * N * 4);
        upload(q, bac, a_codes.data(), a_codes.size());
        upload(q, bas, a_scales.data(), a_scales.size());
        upload(q, bwc, w_codes.data(), w_codes.size());
        upload(q, bws, w_scales.data(), w_scales.size());
        CHECK(vx_mxfp8_gemm(q, bac.addr, bas.addr, bwc.addr, bws.addr,
                            bo.addr, M, N, K, GROUP));
        CHECK(vx_queue_flush(q));
        std::vector<float> got((size_t)M * N);
        download(q, got, bo);

        uint32_t bad = 0;
        double maxrel = 0.0;
        for (size_t i = 0; i < got.size(); ++i) {
            double rel = fabs((double)got[i] - ref[i]) / (fabs(ref[i]) + 1e-3);
            if (rel > maxrel) maxrel = rel;
            if (rel > 5e-3) ++bad;
        }
        printf("gemm_mxfp8:   max_rel=%.2e bad=%u (M=%u N=%u K=%u g=%u)\n",
               maxrel, bad, M, N, K, GROUP);
        if (bad) ++failures;
        vx_buffer_release(bac.h);
        vx_buffer_release(bas.h);
        vx_buffer_release(bwc.h);
        vx_buffer_release(bws.h);
        vx_buffer_release(bo.h);
    }

    vx_mxfp8_finalize();
    vx_queue_release(q);
    vx_device_release(dev);

    if (failures == 0) {
        printf("PASSED\n");
        return 0;
    }
    fprintf(stderr, "FAILED (%d)\n", failures);
    return 1;
}
