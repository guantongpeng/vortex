// Vortex DL quant acceptance (plan P6-01): W4A16 pack/unpack bit-exactness,
// dequant error, W4A16 and W8A8 GEMM end-to-end vs double references.
// The packing reference is computed in float with the exact operation
// order of the kernels so the bit-exact claims are real, not approximated.

#include <vortex/quant.h>
#include <vortex/dtypes.h>
#include <vortex2.h>

#include "quant_fp8.h"  // host side of the fp8 conversions (sw/dl/src)

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

// CPU packing reference mirroring the kernels exactly (float ops, same
// order): scales pass then nibble pass.
static void cpu_pack_w4(const std::vector<float>& w, uint32_t n, uint32_t k,
                        uint32_t group, std::vector<uint8_t>& packed,
                        std::vector<float>& scales) {
    uint32_t groups = (k + group - 1) / group;
    scales.assign((size_t)n * groups, 0.0f);
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
            scales[(size_t)r * groups + g] = amax > 0.0f ? amax / 7.0f : 1.0f;
        }
    }
    for (uint32_t r = 0; r < n; ++r) {
        for (uint32_t c = 0; c < k; ++c) {
            float s = scales[(size_t)r * groups + c / group];
            long q = lroundf(w[(size_t)r * k + c] / s);
            if (q > 7) q = 7;
            if (q < -8) q = -8;
            uint8_t& byte = packed[(size_t)r * ((k + 1) / 2) + c / 2];
            if (c % 2 == 0) {
                byte = (uint8_t)((byte & 0xf0) | (q & 0x0f));
            } else {
                byte = (uint8_t)((byte & 0x0f) | ((q & 0x0f) << 4));
            }
        }
    }
}

static float nib(const std::vector<uint8_t>& p, uint32_t r, uint32_t c,
                 uint32_t k) {
    uint8_t b = p[(size_t)r * ((k + 1) / 2) + c / 2];
    int q = (c % 2 == 0) ? (int)(b & 0x0f) : (int)(b >> 4);
    return (float)((q >= 8) ? q - 16 : q);
}

int main(int argc, char** argv) {
    const char* vxbin = (argc > 1) ? argv[1] : "quant.vxbin";

    vx_device_h dev = nullptr;
    CHECK(vx_device_open(0, &dev));
    vx_queue_h q = nullptr;
    CHECK(vx_queue_create(dev, nullptr, &q));
    if (vx_quant_init(dev, vxbin) != VX_QUANT_OK) {
        fprintf(stderr, "FAILED: vx_quant_init(%s)\n", vxbin);
        return 1;
    }

    int failures = 0;
    const uint32_t M = 40, N = 33, K = 100, GROUP = 32;
    const uint32_t groups = (K + GROUP - 1) / GROUP;

    // ---- W4A16 pack / unpack ------------------------------------------------
    std::vector<float> w((size_t)N * K);
    for (auto& x : w) x = frand();
    w[5] = 0.0f;  // zero-heavy group corner
    for (uint32_t c = 0; c < 10; ++c) w[c] = 0.0f;

    std::vector<uint8_t> ref_packed;
    std::vector<float> ref_scales;
    cpu_pack_w4(w, N, K, GROUP, ref_packed, ref_scales);

    DevBuf bw = make_buf(dev, w.size() * 4);
    DevBuf bp = make_buf(dev, ref_packed.size());
    DevBuf bs = make_buf(dev, ref_scales.size() * 4);
    DevBuf bdq = make_buf(dev, w.size() * 4);
    upload(q, bw, w.data(), w.size() * 4);

    CHECK(vx_quant_pack_w4(q, bw.addr, bp.addr, bs.addr, N, K, GROUP));
    CHECK(vx_queue_flush(q));

    std::vector<uint8_t> got_packed(ref_packed.size());
    std::vector<float> got_scales(ref_scales.size());
    download(q, got_packed, bp);
    download(q, got_scales, bs);

    uint32_t nib_bad = 0, sc_bad = 0;
    for (size_t i = 0; i < got_packed.size(); ++i) {
        if (got_packed[i] != ref_packed[i]) ++nib_bad;
    }
    for (size_t i = 0; i < got_scales.size(); ++i) {
        if (got_scales[i] != ref_scales[i]) ++sc_bad;
    }
    printf("pack_w4:      nibble_mismatch=%u scale_mismatch=%u (of %zu/%zu)\n",
           nib_bad, sc_bad, got_packed.size(), got_scales.size());
    if (nib_bad || sc_bad) ++failures;

    CHECK(vx_quant_unpack_w4(q, bp.addr, bs.addr, bdq.addr, N, K, GROUP));
    CHECK(vx_queue_flush(q));
    std::vector<float> got_dq(w.size());
    download(q, got_dq, bdq);
    uint32_t dq_bad = 0, dq_abs_bad = 0;
    double dq_mean_abs = 0.0;
    for (uint32_t i = 0; i < w.size(); ++i) {
        float expect = nib(ref_packed, i / K, i % K, K) *
                       ref_scales[(i / K) * groups + (i % K) / GROUP];
        if (got_dq[i] != expect) ++dq_bad;
        // Rounding error bound: |dequant - w| <= scale (q may round by 1);
        // per-element relative error is unbounded for small |q|, so the
        // absolute-per-group bound is the honest check.
        float s = ref_scales[(i / K) * groups + (i % K) / GROUP];
        double abs_err = fabs((double)got_dq[i] - (double)w[i]);
        dq_mean_abs += abs_err;
        if (abs_err > (double)s * 1.001) ++dq_abs_bad;
    }
    dq_mean_abs /= w.size();
    printf("unpack_w4:    bitexact_bad=%u abs_violations=%u mean_abs_err=%.2e\n",
           dq_bad, dq_abs_bad, dq_mean_abs);
    if (dq_bad || dq_abs_bad) ++failures;

    // ---- W4A16 GEMM -----------------------------------------------------------
    std::vector<float> act((size_t)M * K);
    for (auto& x : act) x = frand();
    std::vector<uint16_t> act16(act.size());
    for (size_t i = 0; i < act.size(); ++i) act16[i] = vx_f32_to_fp16(act[i]);

    std::vector<double> gemm_ref((size_t)M * N, 0.0);
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t nn = 0; nn < N; ++nn) {
            double s = 0.0;
            for (uint32_t kk = 0; kk < K; ++kk) {
                double wq = nib(ref_packed, nn, kk, K) *
                            ref_scales[(size_t)nn * groups + kk / GROUP];
                s += (double)vx_fp16_to_f32(act16[(size_t)m * K + kk]) * wq;
            }
            gemm_ref[(size_t)m * N + nn] = s;
        }
    }

    DevBuf ba = make_buf(dev, act16.size() * 2);
    DevBuf bo = make_buf(dev, (size_t)M * N * 4);
    upload(q, ba, act16.data(), act16.size() * 2);
    CHECK(vx_quant_gemm_w4a16(q, ba.addr, bp.addr, bs.addr, bo.addr,
                              M, N, K, GROUP));
    CHECK(vx_queue_flush(q));
    std::vector<float> got_out((size_t)M * N);
    download(q, got_out, bo);
    {
        uint32_t bad = 0;
        double maxrel = 0.0;
        for (size_t i = 0; i < got_out.size(); ++i) {
            double rel = fabs((double)got_out[i] - gemm_ref[i]) /
                         (fabs(gemm_ref[i]) + 1e-3);
            if (rel > maxrel) maxrel = rel;
            if (rel > 5e-3) ++bad;
        }
        printf("gemm_w4a16:   max_rel=%.2e bad=%u (M=%u N=%u K=%u g=%u)\n",
               maxrel, bad, M, N, K, GROUP);
        if (bad) ++failures;
    }

    // ---- W8A8 GEMM ------------------------------------------------------------
    // Per-tensor activation scale, per-row weight scale (amax/127).
    float act_amax = 0.0f;
    for (float x : act) act_amax = fmaxf(act_amax, fabsf(x));
    float act_scale = act_amax > 0.0f ? act_amax / 127.0f : 1.0f;
    std::vector<int8_t> act8(act.size());
    for (size_t i = 0; i < act.size(); ++i) {
        long v = lroundf(act[i] / act_scale);
        if (v > 127) v = 127;
        if (v < -127) v = -127;  // symmetric range
        act8[i] = (int8_t)v;
    }
    std::vector<int8_t> w8(w.size());
    std::vector<float> w_scale(N);
    for (uint32_t r = 0; r < N; ++r) {
        float amax = 0.0f;
        for (uint32_t c = 0; c < K; ++c) amax = fmaxf(amax, fabsf(w[(size_t)r * K + c]));
        w_scale[r] = amax > 0.0f ? amax / 127.0f : 1.0f;
        for (uint32_t c = 0; c < K; ++c) {
            long v = lroundf(w[(size_t)r * K + c] / w_scale[r]);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            w8[(size_t)r * K + c] = (int8_t)v;
        }
    }
    std::vector<double> gemm8_ref((size_t)M * N, 0.0);
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t nn = 0; nn < N; ++nn) {
            int32_t acc = 0;
            for (uint32_t kk = 0; kk < K; ++kk) {
                acc += (int32_t)act8[(size_t)m * K + kk] * (int32_t)w8[(size_t)nn * K + kk];
            }
            gemm8_ref[(size_t)m * N + nn] =
                (double)acc * (double)act_scale * (double)w_scale[nn];
        }
    }

    DevBuf ba8 = make_buf(dev, act8.size());
    DevBuf bw8 = make_buf(dev, w8.size());
    DevBuf bws = make_buf(dev, w_scale.size() * 4);
    upload(q, ba8, act8.data(), act8.size());
    upload(q, bw8, w8.data(), w8.size());
    upload(q, bws, w_scale.data(), w_scale.size() * 4);
    CHECK(vx_quant_gemm_w8a8(q, ba8.addr, bw8.addr, bws.addr, act_scale,
                             bo.addr, M, N, K));
    CHECK(vx_queue_flush(q));
    download(q, got_out, bo);
    {
        uint32_t bad = 0;
        double maxrel = 0.0;
        for (size_t i = 0; i < got_out.size(); ++i) {
            double rel = fabs((double)got_out[i] - gemm8_ref[i]) /
                         (fabs(gemm8_ref[i]) + 1e-2);
            if (rel > maxrel) maxrel = rel;
            if (rel > 5e-3) ++bad;
        }
        printf("gemm_w8a8:    max_rel=%.2e bad=%u\n", maxrel, bad);
        if (bad) ++failures;
    }

    // ---- FP8 e4m3 / e5m2 GEMM (P6-02 software path) --------------------
    for (uint32_t mode = 0; mode < 2; ++mode) {
        // Quantize act/w in fp8 on the host with the same header the
        // kernels use; reference = quantized values, double accumulate.
        std::vector<uint8_t> a8((size_t)M * K), w8q((size_t)N * K);
        for (size_t i = 0; i < a8.size(); ++i)
            a8[i] = mode ? vx_f32_to_e5m2(act[i]) : vx_f32_to_e4m3(act[i]);
        for (size_t i = 0; i < w8q.size(); ++i)
            w8q[i] = mode ? vx_f32_to_e5m2(w[i]) : vx_f32_to_e4m3(w[i]);
        std::vector<double> ref8((size_t)M * N, 0.0);
        for (uint32_t m = 0; m < M; ++m)
            for (uint32_t nn = 0; nn < N; ++nn) {
                double acc = 0.0;
                for (uint32_t kk = 0; kk < K; ++kk) {
                    double av = mode ? (double)vx_e5m2_to_f32(a8[(size_t)m * K + kk])
                                     : (double)vx_e4m3_to_f32(a8[(size_t)m * K + kk]);
                    double wv = mode ? (double)vx_e5m2_to_f32(w8q[(size_t)nn * K + kk])
                                     : (double)vx_e4m3_to_f32(w8q[(size_t)nn * K + kk]);
                    acc += av * wv;
                }
                ref8[(size_t)m * N + nn] = acc;
            }
        DevBuf bfa = make_buf(dev, a8.size());
        DevBuf bfw = make_buf(dev, w8q.size());
        upload(q, bfa, a8.data(), a8.size());
        upload(q, bfw, w8q.data(), w8q.size());
        CHECK(vx_quant_gemm_fp8(q, bfa.addr, bfw.addr, bo.addr, M, N, K, mode));
        CHECK(vx_queue_flush(q));
        download(q, got_out, bo);
        uint32_t bad = 0;
        double maxrel = 0.0;
        for (size_t i = 0; i < got_out.size(); ++i) {
            double rel = fabs((double)got_out[i] - ref8[i]) /
                         (fabs(ref8[i]) + 1e-2);
            if (rel > maxrel) maxrel = rel;
            if (rel > 2e-3) ++bad;
        }
        printf("gemm_fp8_%s: max_rel=%.2e bad=%u\n",
               mode ? "e5m2" : "e4m3", maxrel, bad);
        if (bad) ++failures;
        vx_buffer_release(bfa.h);
        vx_buffer_release(bfw.h);
    }

    for (DevBuf* b : {&bw, &bp, &bs, &bdq, &ba, &bo, &ba8, &bw8, &bws}) {
        vx_buffer_release(b->h);
    }
    vx_quant_finalize();
    vx_queue_release(q);
    vx_device_release(dev);

    if (failures == 0) {
        printf("PASSED\n");
        return 0;
    }
    fprintf(stderr, "FAILED (%d)\n", failures);
    return 1;
}
