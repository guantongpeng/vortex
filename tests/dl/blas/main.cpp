// Vortex DL BLAS acceptance (plan P3-01): tiled GEMM for FP32, FP16 and
// BF16 storage against a CPU double-precision reference. The reference for
// the low-precision dtypes is computed from the *converted* inputs, so the
// comparison isolates kernel correctness from input quantization.

#include <vortex/blas.h>
#include <vortex/dtypes.h>
#include <vortex2.h>

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

static unsigned int g_seed = 20260911;
static float frand() {
    g_seed = g_seed * 1103515245u + 12345u;
    return ((float)(g_seed >> 8 & 0xffffu) / 65536.0f - 0.5f) * 2.0f;
}

struct Buffers {
    vx_buffer_h a = nullptr, b = nullptr, c = nullptr;
    uint64_t a_addr = 0, b_addr = 0, c_addr = 0;
    void* c_host = nullptr;
};

static int run_gemm(vx_device_h dev, vx_queue_h q, vx_blas_dtype dt,
                    uint32_t M, uint32_t N, uint32_t K,
                    const std::vector<float>& hf_a,
                    const std::vector<float>& hf_b,
                    const std::vector<double>& ref) {
    // Quantize inputs per dtype; the reference already matches the
    // quantized values.
    size_t a_elems = (size_t)M * K, b_elems = (size_t)K * N;
    size_t a_bytes, b_bytes;
    std::vector<uint8_t> qa, qb;
    if (dt == VX_BLAS_F32) {
        a_bytes = a_elems * 4;
        b_bytes = b_elems * 4;
        qa.resize(a_bytes);
        qb.resize(b_bytes);
        memcpy(qa.data(), hf_a.data(), a_bytes);
        memcpy(qb.data(), hf_b.data(), b_bytes);
    } else {
        const int w = 2;
        a_bytes = a_elems * w;
        b_bytes = b_elems * w;
        qa.resize(a_bytes);
        qb.resize(b_bytes);
        uint16_t* pa = (uint16_t*)qa.data();
        uint16_t* pb = (uint16_t*)qb.data();
        for (size_t i = 0; i < a_elems; ++i)
            pa[i] = dt == VX_BLAS_F16 ? vx_f32_to_fp16(hf_a[i])
                                      : vx_f32_to_bf16(hf_a[i]);
        for (size_t i = 0; i < b_elems; ++i)
            pb[i] = dt == VX_BLAS_F16 ? vx_f32_to_fp16(hf_b[i])
                                      : vx_f32_to_bf16(hf_b[i]);
    }

    Buffers bufs;
    CHECK(vx_buffer_create(dev, a_bytes, 0, &bufs.a));
    CHECK(vx_buffer_create(dev, b_bytes, 0, &bufs.b));
    CHECK(vx_buffer_create(dev, (size_t)M * N * 4, 0, &bufs.c));
    CHECK(vx_buffer_address(bufs.a, &bufs.a_addr));
    CHECK(vx_buffer_address(bufs.b, &bufs.b_addr));
    CHECK(vx_buffer_address(bufs.c, &bufs.c_addr));

    vx_event_h ev = nullptr;
    CHECK(vx_enqueue_write(q, bufs.a, 0, qa.data(), a_bytes, 0, nullptr, &ev));
    CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);
    CHECK(vx_enqueue_write(q, bufs.b, 0, qb.data(), b_bytes, 0, nullptr, &ev));
    CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);

    const float alpha = 1.5f, beta = 0.5f;
    // Pre-fill C with a known value so the beta path is exercised.
    std::vector<float> c_init((size_t)M * N, 0.25f);
    CHECK(vx_enqueue_write(q, bufs.c, 0, c_init.data(), c_init.size() * 4,
                           0, nullptr, &ev));
    CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);

    vx_blas_status st = vx_blas_gemm(q, dt, M, N, K, alpha, beta,
                                     bufs.a_addr, bufs.b_addr, bufs.c_addr);
    if (st != VX_BLAS_OK) {
        fprintf(stderr, "vx_blas_gemm failed: %d\n", st);
        return 1;
    }

    std::vector<float> got((size_t)M * N);
    CHECK(vx_enqueue_read(q, got.data(), bufs.c, 0, got.size() * 4,
                          0, nullptr, &ev));
    CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);

    double max_abs = 0.0, max_rel = 0.0;
    uint32_t errors = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        double expect = alpha * ref[i] + beta * 0.25;
        double diff = fabs((double)got[i] - expect);
        double rel = diff / (fabs(expect) + 1e-6);
        if (diff > max_abs) max_abs = diff;
        if (rel > max_rel) max_rel = rel;
        if (rel > 2e-3) {
            if (errors < 3) {
                fprintf(stderr, "  [%zu] got=%.6f expect=%.6f\n", i, got[i], expect);
            }
            ++errors;
        }
    }

    printf("%-10s M=%u N=%u K=%u -> %s: max_rel=%.2e errors=%u\n",
           dt == VX_BLAS_F32 ? "f32" : dt == VX_BLAS_F16 ? "f16" : "bf16",
           M, N, K, vx_blas_kernel_name(dt), max_rel, errors);

    vx_buffer_release(bufs.a);
    vx_buffer_release(bufs.b);
    vx_buffer_release(bufs.c);
    return errors == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    const char* vxbin = (argc > 1) ? argv[1] : "blas.vxbin";

    vx_device_h dev = nullptr;
    CHECK(vx_device_open(0, &dev));
    vx_queue_h q = nullptr;
    CHECK(vx_queue_create(dev, nullptr, &q));
    if (vx_blas_init(dev, vxbin) != VX_BLAS_OK) {
        fprintf(stderr, "FAILED: vx_blas_init(%s)\n", vxbin);
        return 1;
    }

    int failures = 0;

    // Non-tile-multiple dims exercise the masking paths: 48, 20 (not %16), K=13 (not %8).
    const uint32_t M = 48, N = 20, K = 13;
    std::vector<float> hf_a((size_t)M * K), hf_b((size_t)K * N);

    // Reference per dtype (inputs quantized the same way the test packs them).
    std::vector<double> q_a_f16((size_t)M * K), q_b_f16((size_t)K * N);
    std::vector<double> q_a_bf((size_t)M * K), q_b_bf((size_t)K * N);
    std::vector<double> ref_f32((size_t)M * N, 0.0);
    std::vector<double> ref_f16((size_t)M * N, 0.0);
    std::vector<double> ref_bf((size_t)M * N, 0.0);

    for (size_t i = 0; i < hf_a.size(); ++i) {
        float x = frand();
        hf_a[i] = x;
        q_a_f16[i] = (double)vx_fp16_to_f32(vx_f32_to_fp16(x));
        q_a_bf[i] = (double)vx_bf16_to_f32(vx_f32_to_bf16(x));
    }
    for (size_t i = 0; i < hf_b.size(); ++i) {
        float x = frand();
        hf_b[i] = x;
        q_b_f16[i] = (double)vx_fp16_to_f32(vx_f32_to_fp16(x));
        q_b_bf[i] = (double)vx_bf16_to_f32(vx_f32_to_bf16(x));
    }
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            double s32 = 0, s16 = 0, sbf = 0;
            for (uint32_t k = 0; k < K; ++k) {
                s32 += (double)hf_a[(size_t)m * K + k] * hf_b[(size_t)k * N + n];
                s16 += q_a_f16[(size_t)m * K + k] * q_b_f16[(size_t)k * N + n];
                sbf += q_a_bf[(size_t)m * K + k] * q_b_bf[(size_t)k * N + n];
            }
            ref_f32[(size_t)m * N + n] = s32;
            ref_f16[(size_t)m * N + n] = s16;
            ref_bf[(size_t)m * N + n] = sbf;
        }
    }

    failures += run_gemm(dev, q, VX_BLAS_F32, M, N, K, hf_a, hf_b, ref_f32);
    failures += run_gemm(dev, q, VX_BLAS_F16, M, N, K, hf_a, hf_b, ref_f16);
    failures += run_gemm(dev, q, VX_BLAS_BF16, M, N, K, hf_a, hf_b, ref_bf);

    vx_blas_finalize();
    vx_queue_release(q);
    vx_device_release(dev);

    if (failures == 0) {
        printf("PASSED\n");
        return 0;
    }
    fprintf(stderr, "FAILED (%d dtype runs)\n", failures);
    return 1;
}
