// NVFP4 software-path acceptance (plan P6-02 Q4): exhaustive bit-level
// converter verification vs brute-force RNE references, pack->unpack
// bit-exactness against a host mirror computed in float with the exact
// kernel op order, dequant absolute-error bound, and W4A16-style GEMM vs
// a double reference of the same quantized weights x quantized activations.
//
// Single translation unit: test.cpp includes host.cpp (the self-contained
// dispatch layer), which includes args.h (the shared format/conversions).

#include <vortex/nvfp4.h>
#include <vortex/dtypes.h>

#include <vortex/dtypes.h>  // fp16 storage conversions (read-only header)
#include "nvfp4_args.h"  // e2m1 <-> f32 bit-level conversions

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

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
static uint32_t urand() {
    g_seed = g_seed * 1103515245u + 12345u;
    return g_seed;
}

// ---------------------------------------------------------------------------
// Independent brute-force RNE references (do not share logic with args.h)
// ---------------------------------------------------------------------------

static const float kE2m1Val[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

static uint8_t brute_e2m1(float x) {
    uint32_t sx;
    memcpy(&sx, &x, 4);
    uint8_t sign = (uint8_t)((sx >> 28) & 0x8);
    if (std::isnan(x)) return (uint8_t)(sign | 0x7);  // documented mapping
    double a = fabs((double)x);
    // Nearest-over-a-finite-grid oracle: |x| at/above the max element (6)
    // rounds to it (exact for inf too; distances themselves would absorb).
    if (a >= 6.0) return (uint8_t)(sign | 0x7);
    int best = 0;
    double bd = 1e300;
    for (int i = 0; i < 8; ++i) {
        // Exact below 6: both operands are f32-exact and <= 6, so the
        // double difference (and tie equality) is exact.
        double d = fabs((double)kE2m1Val[i] - a);
        if (d < bd || (d == bd && ((i & 1) == 0))) { bd = d; best = i; }
    }
    return (uint8_t)(sign | best);
}

static float dec_e4m3_ref(int c) {  // arithmetic reference via ldexp
    int e = (c >> 3) & 0xf, m = c & 7;
    if (e == 0xf && m == 0x7) return NAN;
    float v = (float)(e == 0 ? ldexp((double)m, -9)
                             : ldexp(1.0 + m / 8.0, e - 7));
    return (c & 0x80) ? -v : v;
}

static uint8_t brute_e4m3(float x) {
    if (std::isnan(x)) return 0x7f;
    uint32_t sx;
    memcpy(&sx, &x, 4);
    uint8_t sign = (uint8_t)((sx >> 24) & 0x80);
    double a = fabs((double)x);
    // Nearest-over-a-finite-grid oracle: |x| at/above the max finite
    // element (448) saturates to it (exact for inf too).
    if (a >= 448.0) return (uint8_t)(sign | 0x7e);
    int best = 0;
    double bd = 1e300;
    for (int c = 0; c < 0x7f; ++c) {  // NaN code excluded
        // Exact below 448: f32-exact operands, difference fits in double.
        double d = fabs((double)dec_e4m3_ref(c) - a);
        if (d < bd || (d == bd && ((c & 1) == 0))) { bd = d; best = c; }
    }
    return (uint8_t)(sign | best);
}

// ---------------------------------------------------------------------------
// Host-side converter verification (bit level)
// ---------------------------------------------------------------------------

static int check_converters() {
    int failures = 0;
    uint32_t bad = 0, tot = 0;

    // e2m1 decode: exact, including signed zero
    for (int c = 0; c < 16; ++c) {
        float expect = (c & 8 ? -1.0f : 1.0f) * kE2m1Val[c & 7];
        float got = vx_e2m1_to_f32((uint8_t)c);
        uint32_t ue, ug;
        memcpy(&ue, &expect, 4);
        memcpy(&ug, &got, 4);
        ++tot;
        if (ue != ug) ++bad;
    }
    // e2m1 roundtrip: every code is a fixed point of encode(decode(.))
    for (int c = 0; c < 16; ++c) {
        ++tot;
        if (vx_f32_to_e2m1(vx_e2m1_to_f32((uint8_t)c)) != (uint8_t)c) ++bad;
    }
    // e2m1 encode: midpoints +- 1 ulp, saturators, inf, randoms
    for (int i = 0; i < 7; ++i) {
        double mid = 0.5 * ((double)kE2m1Val[i] + (double)kE2m1Val[i + 1]);
        float m = (float)mid;
        for (int s = 0; s < 2; ++s) {
            float x = s ? -m : m;
            float down = nextafterf(m, 0.0f), up = nextafterf(m, 6.0e3f);
            float xs[3] = {x, s ? -down : down, s ? -up : up};
            uint8_t want[3] = {brute_e2m1(x), brute_e2m1(xs[1]),
                               brute_e2m1(xs[2])};
            for (int j = 0; j < 3; ++j) {
                ++tot;
                if (vx_f32_to_e2m1(xs[j]) != want[j]) ++bad;
            }
        }
    }
    {
        float sp[] = {0.0f, -0.0f, 7.5f, -7.5f, 8.0f, -8.0f, 100.0f, -100.0f,
                      1e30f, -1e30f, INFINITY, -INFINITY, NAN};
        for (float x : sp) {
            ++tot;
            if (vx_f32_to_e2m1(x) != brute_e2m1(x)) ++bad;
        }
    }
    for (int i = 0; i < 200000; ++i) {
        float x = frand() * 16.0f;
        ++tot;
        if (vx_f32_to_e2m1(x) != brute_e2m1(x)) ++bad;
    }
    printf("conv_e2m1:  mismatches=%u/%u (decode+roundtrip+ties+sat+randoms)\n",
           bad, tot);
    if (bad) ++failures;

    // e4m3 decode vs arithmetic reference (all codes; NaN -> NaN)
    bad = tot = 0;
    for (int c = 0; c < 256; ++c) {
        float got = vx_nvfp4_e4m3_to_f32((uint8_t)c);
        float ref = dec_e4m3_ref(c);
        ++tot;
        if (std::isnan(ref) ? !std::isnan(got) : got != ref) ++bad;
    }
    // e4m3 roundtrip (NaN code has no fixed point)
    for (int c = 0; c < 256; ++c) {
        if (c == 0x7f || c == 0xff) continue;
        ++tot;
        if (vx_nvfp4_f32_to_e4m3(vx_nvfp4_e4m3_to_f32((uint8_t)c)) !=
            (uint8_t)c) ++bad;
    }
    // midpoints between adjacent positive codes +- 1 ulp (both signs)
    for (int c = 0; c < 0x7e; ++c) {
        double mid = 0.5 * ((double)dec_e4m3_ref(c) + (double)dec_e4m3_ref(c + 1));
        float m = (float)mid;
        float down = nextafterf(m, 0.0f), up = nextafterf(m, 1e30f);
        float xs[6] = {m, down, up, -m, -down, -up};
        for (int j = 0; j < 6; ++j) {
            ++tot;
            if (vx_nvfp4_f32_to_e4m3(xs[j]) != brute_e4m3(xs[j])) ++bad;
        }
    }
    // exhaustive sweep of the subnormal + carry region [0, 0.03)
    for (double x = 0.0; x < 0.03; x += 1.0 / 1048576.0) {
        float f = (float)x;
        ++tot;
        if (vx_nvfp4_f32_to_e4m3(f) != brute_e4m3(f)) ++bad;
    }
    // saturators and log-uniform randoms over magnitudes up to 1e6
    {
        float sp[] = {0.0f, -0.0f, 448.0f, 432.0f, 476.0f, 512.0f, 1e30f,
                      0.0078125f, 0.015625f};
        for (float x : sp) {
            ++tot;
            if (vx_nvfp4_f32_to_e4m3(x) != brute_e4m3(x)) ++bad;
        }
    }
    for (int i = 0; i < 300000; ++i) {
        double mag = ((double)(urand() % 1000003u) / 1000003.0);
        double scale = pow(2.0, (double)((int)(urand() % 53u) - 30));
        float x = (float)(mag * scale);
        if (urand() & 1) x = -x;
        ++tot;
        if (vx_nvfp4_f32_to_e4m3(x) != brute_e4m3(x)) ++bad;
    }
    printf("conv_e4m3:  mismatches=%u/%u (decode+roundtrip+ties+subnormals+sat+randoms)\n",
           bad, tot);
    if (bad) ++failures;
    return failures;
}

// ---------------------------------------------------------------------------
// Device harness
// ---------------------------------------------------------------------------

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

// CPU packing mirror — float ops in the exact kernel order:
// scales pass then nibble pass.
static void cpu_pack(const std::vector<float>& w, uint32_t n, uint32_t k,
                     uint32_t group, float tscale,
                     std::vector<uint8_t>& packed,
                     std::vector<uint8_t>& scales) {
    uint32_t groups = (k + group - 1) / group;
    scales.assign((size_t)n * groups, 0);
    packed.assign((size_t)n * ((k + 1) / 2), 0);
    for (uint32_t r = 0; r < n; ++r) {
        for (uint32_t g = 0; g < groups; ++g) {
            uint32_t c0 = g * group;
            uint32_t c1 = c0 + group < k ? c0 + group : k;
            float amax = 0.0f;
            for (uint32_t c = c0; c < c1; ++c) {
                float a = fabsf(w[(size_t)r * k + c]);
                if (a > amax) amax = a;
            }
            float s = amax > 0.0f ? amax / 6.0f : 1.0f;
            s = s / tscale;
            scales[(size_t)r * groups + g] = vx_nvfp4_f32_to_e4m3(s);
        }
    }
    for (uint32_t r = 0; r < n; ++r) {
        for (uint32_t c = 0; c < k; ++c) {
            float q = w[(size_t)r * k + c];
            q = q / tscale;
            float E = vx_nvfp4_e4m3_to_f32(
                scales[(size_t)r * groups + c / group]);
            q = q / E;
            uint8_t code = vx_f32_to_e2m1(q);
            uint8_t& byte = packed[(size_t)r * ((k + 1) / 2) + c / 2];
            if (c % 2 == 0) {
                byte = (uint8_t)((byte & 0xf0) | (code & 0x0f));
            } else {
                byte = (uint8_t)((byte & 0x0f) | ((code & 0x0f) << 4));
            }
        }
    }
}

// Nibble accessor (raw E2M1 code) in the documented packed layout.
static uint8_t mirror_code(const std::vector<uint8_t>& p, uint32_t r,
                           uint32_t c, uint32_t k) {
    uint8_t b = p[(size_t)r * ((k + 1) / 2) + c / 2];
    return (c % 2 == 0) ? (uint8_t)(b & 0x0f) : (uint8_t)(b >> 4);
}

// Element decode mirroring the device op order exactly:
// v = (e2m1(code) * E4M3_scale) * tscale, two f32 multiplies.
static float mirror_decode(const std::vector<uint8_t>& p,
                           const std::vector<uint8_t>& s, uint32_t r,
                           uint32_t c, uint32_t k, uint32_t group,
                           float tscale) {
    float E = vx_nvfp4_e4m3_to_f32(
        s[(size_t)r * ((k + group - 1) / group) + c / group]);
    float o = vx_e2m1_to_f32(mirror_code(p, r, c, k)) * E;
    return o * tscale;
}

int main(int argc, char** argv) {
    const char* vxbin = (argc > 1) ? argv[1] : "nvfp4.vxbin";

    int failures = check_converters();

    vx_device_h dev = nullptr;
    CHECK(vx_device_open(0, &dev));
    vx_queue_h q = nullptr;
    CHECK(vx_queue_create(dev, nullptr, &q));
    if (vx_nvfp4_init(dev, vxbin) != VX_NVFP4_OK) {
        fprintf(stderr, "FAILED: vx_nvfp4_init(%s)\n", vxbin);
        return 1;
    }

    const uint32_t M = 40, N = 33, K = 100, GROUP = 16;
    const uint32_t groups = (K + GROUP - 1) / GROUP;  // 7, last group = 4 cols

    // Weights with corner cases: an all-zero group (row 0, cols 0..15) and
    // a lone zero inside another group.
    std::vector<float> w((size_t)N * K);
    for (auto& x : w) x = frand();
    for (uint32_t c = 0; c < GROUP; ++c) w[c] = 0.0f;
    w[(size_t)1 * K + 5] = 0.0f;

    std::vector<uint16_t> act16((size_t)M * K);
    for (auto& x : act16) x = vx_f32_to_fp16(frand());

    DevBuf bw = make_buf(dev, w.size() * 4);
    DevBuf bp = make_buf(dev, (size_t)N * ((K + 1) / 2));
    DevBuf bs = make_buf(dev, (size_t)N * groups);
    DevBuf bdq = make_buf(dev, w.size() * 4);
    DevBuf ba = make_buf(dev, act16.size() * 2);
    DevBuf bo = make_buf(dev, (size_t)M * N * 4);
    upload(q, bw, w.data(), w.size() * 4);
    upload(q, ba, act16.data(), act16.size() * 2);

    const float tscales[2] = {1.0f, 0.015625f};  // 2^0 and 2^-6
    for (int ts = 0; ts < 2; ++ts) {
        const float tscale = tscales[ts];
        const char* tag = ts ? "ts=2^-6" : "ts=1   ";

        std::vector<uint8_t> ref_packed, ref_scales;
        cpu_pack(w, N, K, GROUP, tscale, ref_packed, ref_scales);

        // ---- pack (bit-exact vs the host mirror) ------------------------
        CHECK(vx_nvfp4_pack(q, bw.addr, bp.addr, bs.addr, tscale, N, K, GROUP));
        CHECK(vx_queue_flush(q));
        std::vector<uint8_t> got_packed(ref_packed.size());
        std::vector<uint8_t> got_scales(ref_scales.size());
        download(q, got_packed, bp);
        download(q, got_scales, bs);
        uint32_t code_bad = 0, sc_bad = 0;
        for (size_t i = 0; i < got_packed.size(); ++i)
            if (got_packed[i] != ref_packed[i]) ++code_bad;
        for (size_t i = 0; i < got_scales.size(); ++i)
            if (got_scales[i] != ref_scales[i]) ++sc_bad;
        printf("pack %s:  code_mismatch=%u scale_mismatch=%u (of %zu/%zu)\n",
               tag, code_bad, sc_bad, got_packed.size(), got_scales.size());
        if (code_bad || sc_bad) ++failures;

        if (ts == 0) {  // info: code histogram, exercises negatives + zero
            uint32_t hist[16] = {0};
            for (uint32_t r = 0; r < N; ++r)
                for (uint32_t c = 0; c < K; ++c)
                    ++hist[mirror_code(ref_packed, r, c, K)];
            printf("pack codes: ");
            for (int c = 0; c < 16; ++c) printf("%u ", hist[c]);
            printf("(codes 0..15)\n");
        }

        // ---- unpack (bit-exact mirror + absolute error bound) -----------
        CHECK(vx_nvfp4_unpack(q, bp.addr, bs.addr, bdq.addr, tscale, N, K,
                              GROUP));
        CHECK(vx_queue_flush(q));
        std::vector<float> got_dq(w.size());
        download(q, got_dq, bdq);
        uint32_t dq_bad = 0, dq_abs_bad = 0;
        double dq_mean_abs = 0.0;
        for (uint32_t i = 0; i < w.size(); ++i) {
            uint32_t r = i / K, c = i % K;
            float E = vx_nvfp4_e4m3_to_f32(
                ref_scales[(size_t)r * groups + c / GROUP]);
            float expect =
                vx_e2m1_to_f32(mirror_code(ref_packed, r, c, K)) * E;
            expect = expect * tscale;
            if (got_dq[i] != expect) ++dq_bad;
            // Honest bound: RNE over the e2m1 grid errs by at most one grid
            // step of the decoded scale (largest adjacent gap = 2 -> half
            // gap = 1), plus fp division noise (~1e-6 relative).
            double bound = (double)E * (double)tscale * 1.001;
            double abs_err = fabs((double)got_dq[i] - (double)w[i]);
            dq_mean_abs += abs_err;
            if (abs_err > bound) ++dq_abs_bad;
        }
        dq_mean_abs /= w.size();
        printf("unpack %s: bitexact_bad=%u abs_violations=%u "
               "mean_abs_err=%.2e\n", tag, dq_bad, dq_abs_bad, dq_mean_abs);
        if (dq_bad || dq_abs_bad) ++failures;

        // ---- GEMM vs double reference of the same quantized operands ----
        std::vector<double> gemm_ref((size_t)M * N, 0.0);
        for (uint32_t m = 0; m < M; ++m) {
            for (uint32_t nn = 0; nn < N; ++nn) {
                double s = 0.0;
                for (uint32_t kk = 0; kk < K; ++kk) {
                    float a = vx_fp16_to_f32(act16[(size_t)m * K + kk]);
                    float wv = mirror_decode(ref_packed, ref_scales, nn, kk, K,
                                             GROUP, tscale);
                    s += (double)a * (double)wv;  // same k order as device
                }
                gemm_ref[(size_t)m * N + nn] = s;
            }
        }
        CHECK(vx_nvfp4_gemm(q, ba.addr, bp.addr, bs.addr, bo.addr, tscale, M,
                            N, K, GROUP));
        CHECK(vx_queue_flush(q));
        std::vector<float> got_out((size_t)M * N);
        download(q, got_out, bo);
        uint32_t bad = 0;
        double maxrel = 0.0;
        for (size_t i = 0; i < got_out.size(); ++i) {
            double rel = fabs((double)got_out[i] - gemm_ref[i]) /
                         (fabs(gemm_ref[i]) + 1e-2);
            if (rel > maxrel) maxrel = rel;
            if (rel > 2e-2) ++bad;
        }
        printf("gemm %s:  max_rel=%.2e bad=%u (M=%u N=%u K=%u g=%u)\n", tag,
               maxrel, bad, M, N, K, GROUP);
        if (bad) ++failures;
    }

    for (DevBuf* b : {&bw, &bp, &bs, &bdq, &ba, &bo}) vx_buffer_release(b->h);
    vx_nvfp4_finalize();
    vx_queue_release(q);
    vx_device_release(dev);

    if (failures == 0) {
        printf("PASSED\n");
        return 0;
    }
    fprintf(stderr, "FAILED (%d)\n", failures);
    return 1;
}
