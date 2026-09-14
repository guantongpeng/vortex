// Vortex DL LLM support-ops acceptance (plan P7): RoPE, SwiGLU,
// embedding gather, KV-cache append vs CPU references.
//
// RoPE and SwiGLU are checked against double-precision references at
// rel 1e-5. For RoPE the per-element denominator is |x[2i]| + |x[2i+1]|
// (the L1 norm of the rotated input pair) rather than |ref|: a rotation
// can cancel (x0*cos ~= x1*sin) making pure element-relative error
// unbounded, while the float32 pipeline error is proportional to the
// input pair magnitude (rotation is norm-preserving, |ref| <= |x0|+|x1|,
// so this is still a relative measure at the scale of the data). The
// float32 error floor itself (f32 angle p*freq, sinf/cosf/powf ~1 ulp)
// is ~5e-6 of that scale with angles up to 63 rad, i.e. under the 1e-5
// bound. SwiGLU has no cancellation, so it uses plain |ref|.
//
// Embedding gather and KV-cache append are pure f32 copies: compared
// bit-exactly, and the KV test also verifies rows outside
// [base, base+new_tokens) are untouched.

#include <vortex/llm.h>
#include <vortex2.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

#define CHECK(expr) do { \
    if ((expr) != 0) { \
        fprintf(stderr, "FAILED at %s:%d: '%s'\n", __FILE__, __LINE__, #expr); \
        return 1; \
    } \
} while (0)

static unsigned int g_seed = 20260915;
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

// Double reference mirroring the kernel op order: freq via
// pow(10000, (-2*i)/dim), angle = p*freq, rotation on the pair.
static void rope_ref(const std::vector<float>& x, uint32_t seq, uint32_t dim,
                     std::vector<double>& ref) {
    ref.assign(x.size(), 0.0);
    const uint32_t half = dim / 2;
    for (uint32_t p = 0; p < seq; ++p) {
        for (uint32_t i = 0; i < half; ++i) {
            const double freq = pow(10000.0, (-2.0 * (double)i) / (double)dim);
            const double a = (double)p * freq;
            const double c = cos(a), s = sin(a);
            const size_t b = (size_t)p * dim + 2 * i;
            const double x0 = x[b], x1 = x[b + 1];
            ref[b] = x0 * c - x1 * s;
            ref[b + 1] = x0 * s + x1 * c;
        }
    }
    if (dim % 2) {  // odd trailing column is untouched by the kernel
        for (uint32_t p = 0; p < seq; ++p) {
            ref[(size_t)p * dim + dim - 1] = x[(size_t)p * dim + dim - 1];
        }
    }
}

static int run_rope(vx_device_h dev, vx_queue_h q, uint32_t seq,
                    uint32_t dim, int& failures) {
    std::vector<float> x((size_t)seq * dim);
    for (auto& v : x) v = frand();
    x[0] = 0.0f;                    // zero element in a pair
    x[(size_t)(seq / 2) * dim + 1] = 0.0f;
    x[(size_t)(seq / 2) * dim] = 0.0f;  // fully zero pair stays zero

    std::vector<double> ref;
    rope_ref(x, seq, dim, ref);

    DevBuf bx = make_buf(dev, x.size() * 4);
    upload(q, bx, x.data(), x.size() * 4);
    CHECK(vx_llm_rope(q, bx.addr, seq, dim));
    CHECK(vx_queue_flush(q));
    std::vector<float> got(x.size());
    download(q, got, bx);
    vx_buffer_release(bx.h);

    uint32_t bad = 0;
    double max_rel = 0.0, max_abs = 0.0;
    const uint32_t half = dim / 2;
    for (uint32_t p = 0; p < seq; ++p) {
        for (uint32_t i = 0; i < half; ++i) {
            const size_t b = (size_t)p * dim + 2 * i;
            const double denom = fabs((double)x[b]) + fabs((double)x[b + 1]);
            for (size_t e = b; e <= b + 1; ++e) {
                const double err = fabs((double)got[e] - ref[e]);
                const double rel = err / (denom > 0.0 ? denom : 1.0);
                if (err > max_abs) max_abs = err;
                if (rel > max_rel) max_rel = rel;
                if (rel > 1e-5) ++bad;
            }
        }
        if (dim % 2) {  // untouched trailing column must be bit-identical
            const size_t e = (size_t)p * dim + dim - 1;
            if (got[e] != x[e]) ++bad;
        }
    }
    printf("rope[%ux%u%s]:  max_rel=%.2e max_abs=%.2e bad=%u\n", seq, dim,
           dim % 2 ? " odd" : "", max_rel, max_abs, bad);
    if (bad) ++failures;
    return 0;
}

int main(int argc, char** argv) {
    const char* vxbin = (argc > 1) ? argv[1] : "llm.vxbin";

    vx_device_h dev = nullptr;
    CHECK(vx_device_open(0, &dev));
    vx_queue_h q = nullptr;
    CHECK(vx_queue_create(dev, nullptr, &q));
    if (vx_llm_init(dev, vxbin) != VX_LLM_OK) {
        fprintf(stderr, "FAILED: vx_llm_init(%s)\n", vxbin);
        return 1;
    }

    int failures = 0;

    // ---- RoPE ------------------------------------------------------------
    CHECK(run_rope(dev, q, 64, 64, failures));  // max angle 63 rad
    CHECK(run_rope(dev, q, 8, 33, failures));   // odd dim guard

    // ---- SwiGLU ------------------------------------------------------------
    {
        const uint32_t n = 1024;
        std::vector<float> a(n), b(n);
        for (uint32_t i = 0; i < n; ++i) {
            a[i] = frand() * 4.0f;  // push sigmoid into its tails too
            b[i] = frand();
        }
        a[7] = 0.0f;   // silu(0) = 0
        a[9] = -9.0f;  // deep negative tail
        a[11] = 9.0f;  // deep positive tail
        b[13] = 0.0f;

        std::vector<double> ref(n);
        for (uint32_t i = 0; i < n; ++i) {
            const double gate = (double)a[i] / (1.0 + exp(-(double)a[i]));
            ref[i] = gate * (double)b[i];
        }

        DevBuf ba = make_buf(dev, n * 4);
        DevBuf bb = make_buf(dev, n * 4);
        DevBuf bo = make_buf(dev, n * 4);
        upload(q, ba, a.data(), n * 4);
        upload(q, bb, b.data(), n * 4);
        CHECK(vx_llm_swiglu(q, ba.addr, bb.addr, bo.addr, n));
        CHECK(vx_queue_flush(q));
        std::vector<float> got(n);
        download(q, got, bo);
        vx_buffer_release(ba.h);
        vx_buffer_release(bb.h);
        vx_buffer_release(bo.h);

        uint32_t bad = 0;
        double max_rel = 0.0;
        for (uint32_t i = 0; i < n; ++i) {
            const double err = fabs((double)got[i] - ref[i]);
            const double rel = err / (fabs(ref[i]) + 1e-30);
            if (rel > max_rel) max_rel = rel;
            if (rel > 1e-5) ++bad;
        }
        printf("swiglu[n=%u]:   max_rel=%.2e bad=%u\n", n, max_rel, bad);
        if (bad) ++failures;
    }

    // ---- Embedding gather ---------------------------------------------------
    {
        const uint32_t vocab = 100, dim = 32, n = 32;
        std::vector<float> table((size_t)vocab * dim);
        for (auto& v : table) v = frand();
        std::vector<uint32_t> ids(n);
        for (uint32_t t = 0; t < n; ++t)
            ids[t] = (uint32_t)(frand() * 0.5f + 0.5f) * vocab;
        ids[0] = 0;              // first row
        ids[1] = vocab - 1;      // last row
        ids[2] = vocab - 1;      // duplicate of the last row
        for (uint32_t t = 3; t < n; ++t) ids[t] = t % vocab;  // ordered sweep

        DevBuf bt = make_buf(dev, table.size() * 4);
        DevBuf bi = make_buf(dev, ids.size() * 4);
        DevBuf bo = make_buf(dev, (size_t)n * dim * 4);
        upload(q, bt, table.data(), table.size() * 4);
        upload(q, bi, ids.data(), ids.size() * 4);
        CHECK(vx_llm_embedding(q, bt.addr, bi.addr, bo.addr, dim, n));
        CHECK(vx_queue_flush(q));
        std::vector<float> got((size_t)n * dim);
        download(q, got, bo);
        vx_buffer_release(bt.h);
        vx_buffer_release(bi.h);
        vx_buffer_release(bo.h);

        uint32_t bad = 0;
        for (uint32_t t = 0; t < n; ++t) {
            for (uint32_t d = 0; d < dim; ++d) {
                const float expect = table[(size_t)ids[t] * dim + d];
                if (got[(size_t)t * dim + d] != expect) ++bad;
            }
        }
        printf("embedding:      mismatch=%u (of %u) last_row=%u\n", bad,
               n * dim, ids[1]);
        if (bad) ++failures;
    }

    // ---- KV-cache append ----------------------------------------------------
    {
        const uint32_t capacity = 128, dim = 32, base = 40, nt = 24;
        std::vector<float> orig((size_t)capacity * dim);
        for (auto& v : orig) v = frand();
        std::vector<float> newk((size_t)nt * dim);
        for (auto& v : newk) v = frand();

        DevBuf bc = make_buf(dev, orig.size() * 4);
        DevBuf bk = make_buf(dev, newk.size() * 4);
        upload(q, bc, orig.data(), orig.size() * 4);
        upload(q, bk, newk.data(), newk.size() * 4);
        CHECK(vx_llm_kv_append(q, bc.addr, bk.addr, capacity, base, nt, dim));
        CHECK(vx_queue_flush(q));
        std::vector<float> got(orig.size());
        download(q, got, bc);
        vx_buffer_release(bc.h);
        vx_buffer_release(bk.h);

        uint32_t copy_bad = 0, untouched_bad = 0;
        for (uint32_t r = 0; r < capacity; ++r) {
            const bool inside = r >= base && r < base + nt;
            for (uint32_t d = 0; d < dim; ++d) {
                const float expect = inside
                                         ? newk[(size_t)(r - base) * dim + d]
                                         : orig[(size_t)r * dim + d];
                if (got[(size_t)r * dim + d] != expect) {
                    if (inside) ++copy_bad; else ++untouched_bad;
                }
            }
        }
        printf("kv_append:      copy_bad=%u untouched_bad=%u (rows %u..%u of "
               "%u)\n", copy_bad, untouched_bad, base, base + nt - 1,
               capacity);
        if (copy_bad || untouched_bad) ++failures;
    }

    vx_llm_finalize();
    vx_queue_release(q);
    vx_device_release(dev);

    if (failures == 0) {
        printf("PASSED\n");
        return 0;
    }
    fprintf(stderr, "FAILED (%d)\n", failures);
    return 1;
}
