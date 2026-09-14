// Vortex DL attention acceptance (plan P3 second tier): scaled dot-product
// attention forward vs a double-precision CPU reference with max-shift
// softmax. Covers full and causal masking, odd L, B>1, D and L below the
// 16-thread CTA width, and both the 1/sqrt(D) default-scale path and an
// explicit user scale. Fixed seed; exit code reflects the result.

#include <vortex2.h>

#include <vortex/attn.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
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

// Double-precision reference. `scale` is the double value of the exact
// f32 scale word the kernel consumed (host computes 1/sqrt(D) in double
// and casts to f32; the reference mirrors that cast), so the comparison
// isolates device f32 arithmetic, not scale rounding.
static void attn_ref(const std::vector<float>& Q, const std::vector<float>& K,
                     const std::vector<float>& V, uint32_t B, uint32_t H,
                     uint32_t L, uint32_t D, uint32_t causal, double scale,
                     std::vector<double>& O) {
    O.assign(Q.size(), 0.0);
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t h = 0; h < H; ++h) {
            const size_t base = ((size_t)b * H + h) * L * D;
            for (uint32_t qq = 0; qq < L; ++qq) {
                const uint32_t jmax = causal ? qq : L - 1;
                std::vector<double> s(jmax + 1);
                double m = -INFINITY;
                for (uint32_t j = 0; j <= jmax; ++j) {
                    double dot = 0.0;
                    for (uint32_t d = 0; d < D; ++d) {
                        dot += (double)Q[base + (size_t)qq * D + d] *
                               (double)K[base + (size_t)j * D + d];
                    }
                    s[j] = dot * scale;
                    if (s[j] > m) m = s[j];
                }
                double sum = 0.0;
                for (uint32_t j = 0; j <= jmax; ++j) {
                    s[j] = std::exp(s[j] - m);
                    sum += s[j];
                }
                for (uint32_t d = 0; d < D; ++d) {
                    double acc = 0.0;
                    for (uint32_t j = 0; j <= jmax; ++j) {
                        acc += s[j] * (double)V[base + (size_t)j * D + d];
                    }
                    O[base + (size_t)qq * D + d] = acc / sum;
                }
            }
        }
    }
}

struct Case {
    const char* name;
    uint32_t b, h, l, d, causal;
    bool explicit_scale;  // use vx_attn_forward_scaled with 1.0/D
    float q_amp;          // Q amplitude (stress the max-shift path)
};

// bad if d/(|ref| + atol) > rtol. rtol = 1e-4 is the honest relative
// tolerance for well-scaled outputs (f32 device arithmetic, measured
// errors ~1e-7 there — 100x headroom). atol = 1e-2 is a SCALE floor, not
// a per-element abs tolerance: O is a convex combination of V (|V| <= 1
// here, and |O| <= max|V| in general), so cancellation-heavy entries can
// be arbitrarily close to 0 while the f32 accumulation error over <= L
// prob-weighted terms stays ABSOLUTE (~L*2^-24*max|V| bound; measured
// max_abs 1.7e-7 over all cases). The floor makes the near-zero absolute
// cap rtol*atol = 1e-6 — 6x above the measured worst — while entries
// with |ref| >= 0.1 are governed by the plain 1e-4 relative criterion.
static void check(const Case& c, const std::vector<float>& got,
                  const std::vector<double>& ref, double rtol, double atol,
                  int* failures) {
    uint32_t bad = 0;
    double max_rel = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        double dd = std::fabs((double)got[i] - ref[i]);
        double r = dd / (std::fabs(ref[i]) + atol);
        if (dd > max_abs) max_abs = dd;
        if (r > max_rel) max_rel = r;
        if (r > rtol) {
            if (bad < 3) {
                fprintf(stderr, "  %s[%zu] got=%.7f ref=%.7f\n", c.name, i,
                        (double)got[i], ref[i]);
            }
            ++bad;
        }
    }
    printf("%-24s n=%-5zu max_rel=%.2e max_abs=%.2e bad=%u\n", c.name,
           got.size(), max_rel, max_abs, bad);
    if (bad) ++*failures;
}

static int run_case(vx_device_h dev, vx_queue_h q, const Case& c) {
    const size_t n = (size_t)c.b * c.h * c.l * c.d;
    std::vector<float> Q(n), K(n), V(n);
    for (size_t i = 0; i < n; ++i) {
        Q[i] = frand() * c.q_amp;
        K[i] = frand();
        V[i] = frand();
    }

    // Mirror the exact f32 scale word the device consumes.
    double sc;
    if (c.explicit_scale) {
        sc = (double)(float)(1.0 / (double)c.d);
    } else {
        sc = (double)(float)(1.0 / std::sqrt((double)c.d));
    }
    std::vector<double> ref;
    attn_ref(Q, K, V, c.b, c.h, c.l, c.d, c.causal, sc, ref);

    DevBuf bq = make_buf(dev, n * 4);
    DevBuf bk = make_buf(dev, n * 4);
    DevBuf bv = make_buf(dev, n * 4);
    DevBuf bo = make_buf(dev, n * 4);
    upload(q, bq, Q.data(), n * 4);
    upload(q, bk, K.data(), n * 4);
    upload(q, bv, V.data(), n * 4);

    vx_attn_status st;
    if (c.explicit_scale) {
        st = vx_attn_forward_scaled(q, bq.addr, bk.addr, bv.addr, bo.addr,
                                    c.b, c.h, c.l, c.d, c.causal,
                                    1.0 / (double)c.d);
    } else {
        st = vx_attn_forward(q, bq.addr, bk.addr, bv.addr, bo.addr,
                             c.b, c.h, c.l, c.d, c.causal);
    }
    if (st != VX_ATTN_OK) {
        fprintf(stderr, "FAILED: %s launch status %d\n", c.name, (int)st);
        return 1;
    }
    CHECK(vx_queue_flush(q));

    std::vector<float> got(n, -12345.0f);
    download(q, got, bo);

    int failures = 0;
    check(c, got, ref, 1e-4, 1e-2, &failures);

    vx_buffer_release(bq.h);
    vx_buffer_release(bk.h);
    vx_buffer_release(bv.h);
    vx_buffer_release(bo.h);
    return failures;
}

int main(int argc, char** argv) {
    const char* vxbin = (argc > 1) ? argv[1] : "attn.vxbin";

    vx_device_h dev = nullptr;
    CHECK(vx_device_open(0, &dev));
    vx_queue_h q = nullptr;
    CHECK(vx_queue_create(dev, nullptr, &q));
    if (vx_attn_init(dev, vxbin) != VX_ATTN_OK) {
        fprintf(stderr, "FAILED: vx_attn_init(%s)\n", vxbin);
        return 1;
    }

    // Required shape (odd L) full + causal, default 1/sqrt(D) scale; a
    // small B=2 / D=8 / L=8 case with explicit scale (scale path, B>1
    // indexing, D and L below the CTA width); and a plan-cap shape with
    // amplified Q to stress the max-shift softmax.
    const Case cases[] = {
        {"full   1x2x33x16", 1, 2, 33, 16, 0, false, 1.0f},
        {"causal 1x2x33x16", 1, 2, 33, 16, 1, false, 1.0f},
        {"causal 2x1x8x8 sc=1/D", 2, 1, 8, 8, 1, true, 1.0f},
        {"causal 1x4x64x16 amp3", 1, 4, 64, 16, 1, false, 3.0f},
    };

    int failures = 0;
    for (const Case& c : cases) {
        if (run_case(dev, q, c) != 0) ++failures;
    }

    vx_attn_finalize();
    vx_queue_release(q);
    vx_device_release(dev);

    if (failures == 0) {
        printf("PASSED\n");
        return 0;
    }
    fprintf(stderr, "FAILED (%d)\n", failures);
    return 1;
}
