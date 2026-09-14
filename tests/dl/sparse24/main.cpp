// Vortex DL 2:4 structured-sparsity acceptance (plan P6-02 Q5):
//   1. prune bit-exactness vs a host mirror (same tie-break: lower index),
//      plus an independent top-2 oracle over every group,
//   2. exact-50% sparsity from the mask fields (K divisible by 4),
//   3. sparse GEMM == dense GEMM of the pruned matrix (double reference),
//   4. a poisoned-values rerun proving the GEMM gates loads on the mask
//      (pruned slots overwritten with 1e30 must not change one output bit).

#include <vortex/sparse24.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <algorithm>
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

// Host mirror of the prune kernel: top-2 magnitudes per consecutive group
// of 4 along K, ties broken to the lower index (ascending scan, strict
// '>'). Pure selection + copies, no arithmetic on the values, so the
// device output is expected to be bit-identical.
static void cpu_prune(const std::vector<float>& w, uint32_t n, uint32_t k,
                      std::vector<float>& values,
                      std::vector<uint8_t>& mask) {
    const uint32_t mrb = (k + 3) / 4;
    values.assign((size_t)n * k, 0.0f);
    mask.assign((size_t)n * mrb, 0);
    for (uint32_t r = 0; r < n; ++r) {
        for (uint32_t g = 0; g < mrb; ++g) {
            const uint32_t c0 = g * 4;
            const uint32_t gsz = std::min(4u, k - c0);
            uint32_t i1 = 0;
            for (uint32_t i = 1; i < gsz; ++i) {
                if (std::fabs(w[(size_t)r * k + c0 + i]) >
                    std::fabs(w[(size_t)r * k + c0 + i1])) {
                    i1 = i;
                }
            }
            int32_t i2 = -1;
            for (uint32_t i = 0; i < gsz; ++i) {
                if (i == i1) continue;
                if (i2 < 0 ||
                    std::fabs(w[(size_t)r * k + c0 + i]) >
                        std::fabs(w[(size_t)r * k + c0 + i2])) {
                    i2 = (int32_t)i;
                }
            }
            for (uint32_t i = 0; i < gsz; ++i) {
                const bool keep = (i == i1 || (int32_t)i == i2);
                values[(size_t)r * k + c0 + i] =
                    keep ? w[(size_t)r * k + c0 + i] : 0.0f;
                if (keep) mask[(size_t)r * mrb + g] |= (uint8_t)(1u << (2 * i));
            }
        }
    }
}

// Independent semantic oracle (does not reuse the selection code): every
// pruned element must be strictly smaller in magnitude than every kept
// element of its group, or equal with a HIGHER index (tie-break check);
// exactly min(2, gsz) elements kept; every 2-bit field is 0 or 1.
static uint32_t check_oracle(const std::vector<float>& w,
                             const std::vector<uint8_t>& mask,
                             uint32_t n, uint32_t k) {
    uint32_t bad = 0;
    const uint32_t mrb = (k + 3) / 4;
    for (uint32_t r = 0; r < n; ++r) {
        for (uint32_t g = 0; g < mrb; ++g) {
            const uint32_t c0 = g * 4;
            const uint32_t gsz = std::min(4u, k - c0);
            const uint8_t mb = mask[(size_t)r * mrb + g];
            std::vector<uint32_t> kept, pruned;
            for (uint32_t i = 0; i < gsz; ++i) {
                const uint32_t f = (mb >> (2 * i)) & 3u;
                if (f > 1) ++bad;  // reserved field value
                (f == 1 ? kept : pruned).push_back(i);
            }
            if (kept.size() != (gsz < 2 ? gsz : 2)) {
                ++bad;
                continue;
            }
            for (uint32_t p : pruned) {
                for (uint32_t q : kept) {
                    const float ap = std::fabs(w[(size_t)r * k + c0 + p]);
                    const float aq = std::fabs(w[(size_t)r * k + c0 + q]);
                    if (!(ap < aq || (ap == aq && p > q))) ++bad;
                }
            }
            // Trailing fields of a ragged final group must be zero.
            for (uint32_t i = gsz; i < 4; ++i) {
                if (((mb >> (2 * i)) & 3u) != 0) ++bad;
            }
        }
    }
    return bad;
}

static uint32_t count_kept(const std::vector<uint8_t>& mask) {
    uint32_t kept = 0;
    for (uint8_t mb : mask) {
        for (uint32_t i = 0; i < 4; ++i) {
            if (((mb >> (2 * i)) & 3u) == 1u) ++kept;
        }
    }
    return kept;
}

// Runs prune on device and checks bit-exactness + oracle. Returns 0 on
// success; fills device-side values/mask for the GEMM stage.
static uint32_t run_prune_case(vx_device_h dev, vx_queue_h q,
                               const std::vector<float>& w, uint32_t n,
                               uint32_t k, const char* tag,
                               std::vector<float>* dev_values,
                               std::vector<uint8_t>* dev_mask) {
    std::vector<float> ref_values;
    std::vector<uint8_t> ref_mask;
    cpu_prune(w, n, k, ref_values, ref_mask);

    DevBuf bw = make_buf(dev, w.size() * 4);
    DevBuf bv = make_buf(dev, ref_values.size() * 4);
    DevBuf bm = make_buf(dev, ref_mask.size());
    upload(q, bw, w.data(), w.size() * 4);
    CHECK(vx_sparse24_prune(q, bw.addr, bv.addr, bm.addr, n, k));
    CHECK(vx_queue_flush(q));

    std::vector<float> got_values(ref_values.size());
    std::vector<uint8_t> got_mask(ref_mask.size());
    download(q, got_values, bv);
    download(q, got_mask, bm);

    uint32_t vbad = 0, mbad = 0;
    for (size_t i = 0; i < got_values.size(); ++i) {
        if (memcmp(&got_values[i], &ref_values[i], sizeof(float))) ++vbad;
    }
    for (size_t i = 0; i < got_mask.size(); ++i) {
        if (got_mask[i] != ref_mask[i]) ++mbad;
    }
    const uint32_t obad = check_oracle(w, got_mask, n, k);
    printf("prune %-5s: values_mismatch=%u/%zu mask_mismatch=%u/%zu "
           "oracle_violations=%u\n",
           tag, vbad, got_values.size(), mbad, got_mask.size(), obad);

    if (dev_values) dev_values->swap(got_values);
    if (dev_mask) dev_mask->swap(got_mask);

    vx_buffer_release(bw.h);
    vx_buffer_release(bv.h);
    vx_buffer_release(bm.h);
    return vbad + mbad + obad;
}

int main(int argc, char** argv) {
    const char* vxbin = (argc > 1) ? argv[1] : "sparse24.vxbin";

    vx_device_h dev = nullptr;
    CHECK(vx_device_open(0, &dev));
    vx_queue_h q = nullptr;
    CHECK(vx_queue_create(dev, nullptr, &q));
    if (vx_sparse24_init(dev, vxbin) != VX_SPARSE24_OK) {
        fprintf(stderr, "FAILED: vx_sparse24_init(%s)\n", vxbin);
        return 1;
    }

    int failures = 0;
    const uint32_t M = 40, N = 33, K = 100;

    // ---- Weights with planted corner cases -----------------------------------
    std::vector<float> w((size_t)N * K);
    for (auto& x : w) x = frand();
    for (uint32_t c = 0; c < 4; ++c) w[5 * K + c] = 0.0f;          // all-zero group
    w[7 * K + 10] = 0.5f; w[7 * K + 11] = -0.5f;                    // 4-way tie:
    w[7 * K + 12] = 0.5f; w[7 * K + 13] = -0.5f;                    // keep 10, 11
    w[7 * K + 14] = 0.25f; w[7 * K + 15] = 1.0f;                    // tied maxima:
    w[7 * K + 16] = -1.0f; w[7 * K + 17] = 0.25f;                   // keep 15, 16
    w[9 * K + 20] = 0.3f; w[9 * K + 21] = 0.3f;                     // 2nd-place tie:
    w[9 * K + 22] = -0.9f; w[9 * K + 23] = 0.2f;                    // keep 22, 20

    std::vector<float> dev_values;
    std::vector<uint8_t> dev_mask;
    failures += run_prune_case(dev, q, w, N, K, "K=100", &dev_values,
                               &dev_mask);

    // Ragged-tail generality: K=6 -> groups of 4 and 2 (both kept in tail).
    {
        const uint32_t n2 = 3, k2 = 6;
        std::vector<float> w2((size_t)n2 * k2);
        for (auto& x : w2) x = frand();
        w2[0] = -0.7f; w2[1] = 0.7f;  // tie in first group
        failures += run_prune_case(dev, q, w2, n2, k2, "K=6", nullptr,
                                   nullptr);
    }

    // ---- Exact 50% sparsity (mask-field accounting) ---------------------------
    {
        const uint32_t kept = count_kept(dev_mask);
        const uint32_t total = N * K;
        const double frac = 100.0 * (double)(total - kept) / (double)total;
        printf("sparsity:     kept=%u/%u fields, pruned=%.4f%% (expected "
               "kept=%u, 50.0000%%)\n",
               kept, total, frac, total / 2);
        if (kept != total / 2 || frac != 50.0) ++failures;
    }

    // ---- Sparse GEMM vs dense GEMM of the pruned matrix -----------------------
    std::vector<float> act((size_t)M * K);
    for (auto& x : act) x = frand();

    std::vector<double> ref((size_t)M * N, 0.0);
    std::vector<double> scale((size_t)M * N, 0.0);  // sum |a*w| per cell
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t nn = 0; nn < N; ++nn) {
            double s = 0.0, sc = 0.0;
            for (uint32_t kk = 0; kk < K; ++kk) {
                const double p = (double)act[(size_t)m * K + kk] *
                                 (double)dev_values[(size_t)nn * K + kk];
                s += p;
                sc += fabs(p);
            }
            ref[(size_t)m * N + nn] = s;
            scale[(size_t)m * N + nn] = sc;
        }
    }

    DevBuf ba = make_buf(dev, act.size() * 4);
    DevBuf bv = make_buf(dev, dev_values.size() * 4);
    DevBuf bm = make_buf(dev, dev_mask.size());
    DevBuf bo = make_buf(dev, (size_t)M * N * 4);
    upload(q, ba, act.data(), act.size() * 4);
    upload(q, bv, dev_values.data(), dev_values.size() * 4);
    upload(q, bm, dev_mask.data(), dev_mask.size());

    CHECK(vx_sparse24_gemm(q, ba.addr, bv.addr, bm.addr, bo.addr, M, N, K));
    CHECK(vx_queue_flush(q));
    std::vector<float> got_out((size_t)M * N);
    download(q, got_out, bo);

    {
        uint32_t bad = 0;
        double maxrel = 0.0;
        for (size_t i = 0; i < got_out.size(); ++i) {
            // Relative to the accumulated-magnitude scale sum|a*w| (double):
            // the honest denominator for FP32 accumulation, immune to
            // cancellation-driven |ref| ~ 0 blowups yet strict (1e-5) where
            // the cell is well-conditioned (scale ~ |ref| there).
            const double rel = fabs((double)got_out[i] - ref[i]) /
                               (scale[i] + 1e-30);
            if (rel > maxrel) maxrel = rel;
            if (rel > 1e-5) ++bad;
        }
        printf("gemm_2:4:     max_rel=%.2e bad=%u (M=%u N=%u K=%u, tol 1e-5 "
               "vs double dense-of-pruned)\n",
               maxrel, bad, M, N, K);
        if (bad) ++failures;
    }

    // ---- Mask-gate proof: poison the pruned slots, output must not change ----
    {
        std::vector<float> poison = dev_values;
        for (size_t i = 0; i < poison.size(); ++i) {
            const uint32_t r = (uint32_t)(i / K);
            const uint32_t mb = dev_mask[r * ((K + 3) / 4) + (i % K) / 4];
            if (((mb >> (2 * (i & 3))) & 3u) != 1u) poison[i] = 1e30f;
        }
        DevBuf bp = make_buf(dev, poison.size() * 4);
        DevBuf bo2 = make_buf(dev, (size_t)M * N * 4);
        upload(q, bp, poison.data(), poison.size() * 4);
        CHECK(vx_sparse24_gemm(q, ba.addr, bp.addr, bm.addr, bo2.addr, M, N,
                               K));
        CHECK(vx_queue_flush(q));
        std::vector<float> got2((size_t)M * N);
        download(q, got2, bo2);
        uint32_t bad = 0;
        for (size_t i = 0; i < got2.size(); ++i) {
            if (memcmp(&got2[i], &got_out[i], sizeof(float))) ++bad;
        }
        printf("mask_gate:    poisoned-pruned rerun bitwise_mismatch=%u "
               "(metadata gates the loads)\n",
               bad);
        if (bad) ++failures;
        vx_buffer_release(bp.h);
        vx_buffer_release(bo2.h);
    }

    for (DevBuf* b : {&ba, &bv, &bm, &bo}) vx_buffer_release(b->h);
    vx_sparse24_finalize();
    vx_queue_release(q);
    vx_device_release(dev);

    if (failures == 0) {
        printf("PASSED\n");
        return 0;
    }
    fprintf(stderr, "FAILED (%d)\n", failures);
    return 1;
}
