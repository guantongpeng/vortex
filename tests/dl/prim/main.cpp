// Vortex DL prim/norm acceptance (plan P3-02): activations, reductions,
// argmax, softmax, layernorm, rmsnorm against the double-precision
// reference in ../ref.h. Fixed seed, includes boundary values (zeros,
// equal-max elements for argmax determinism).

#include <vortex/prim.h>
#include <vortex2.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "../ref.h"

#define CHECK(expr) do { \
    if ((expr) != 0) { \
        fprintf(stderr, "FAILED at %s:%d: '%s'\n", __FILE__, __LINE__, #expr); \
        return 1; \
    } \
} while (0)

static unsigned int g_seed = 20260914;
static float frand() {
    g_seed = g_seed * 1103515245u + 12345u;
    return ((float)(g_seed >> 8 & 0xffffu) / 65536.0f - 0.5f) * 4.0f;
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

static void download(vx_queue_h q, void* dst, const DevBuf& b,
                     size_t bytes) {
    vx_event_h ev = nullptr;
    vx_enqueue_read(q, dst, b.h, 0, bytes, 0, nullptr, &ev);
    vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE);
    vx_event_release(ev);
}

static int check(const char* name, const std::vector<float>& got,
                 const std::vector<double>& ref, double rtol, double atol,
                 int* failures) {
    double max_rel = 0.0;
    uint32_t bad = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        double d = std::fabs((double)got[i] - ref[i]);
        double r = d / (std::fabs(ref[i]) + atol);
        if (r > max_rel) max_rel = r;
        if (r > rtol) {
            if (bad < 3) {
                fprintf(stderr, "  %s[%zu] got=%.6f ref=%.6f\n", name, i,
                        (double)got[i], ref[i]);
            }
            ++bad;
        }
    }
    printf("%-14s n=%-5zu max_rel=%.2e bad=%u\n", name, got.size(), max_rel, bad);
    if (bad) ++*failures;
    return bad == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    const char* vxbin = (argc > 1) ? argv[1] : "prim.vxbin";

    vx_device_h dev = nullptr;
    CHECK(vx_device_open(0, &dev));
    vx_queue_h q = nullptr;
    CHECK(vx_queue_create(dev, nullptr, &q));
    if (vx_prim_init(dev, vxbin) != VX_PRIM_OK) {
        fprintf(stderr, "FAILED: vx_prim_init(%s)\n", vxbin);
        return 1;
    }

    int failures = 0;
    const uint32_t N = 777;  // non-multiple of block sizes

    // ---- activations -----------------------------------------------------
    std::vector<float> in(N);
    for (auto& x : in) x = frand();
    in[0] = 0.0f;  // boundary
    DevBuf din = make_buf(dev, N * 4), dout = make_buf(dev, N * 4);
    upload(q, din, in.data(), N * 4);

    std::vector<float> got(N);
    struct ActCase { const char* name; vx_prim_op op; };
    for (ActCase ac : {ActCase{"relu", VX_PRIM_OP_RELU},
                       ActCase{"gelu", VX_PRIM_OP_GELU},
                       ActCase{"silu", VX_PRIM_OP_SILU}}) {
        CHECK(vx_prim_unary(q, ac.op, din.addr, dout.addr, N));
        CHECK(vx_queue_flush(q));
        download(q, got.data(), dout, N * 4);
        std::vector<double> r(N);
        for (uint32_t i = 0; i < N; ++i) {
            r[i] = ac.op == VX_PRIM_OP_RELU ? std::max((double)in[i], 0.0)
                 : ac.op == VX_PRIM_OP_GELU ? ref::gelu(in[i])
                                            : ref::silu(in[i]);
        }
        check(ac.name, got, r, 2e-5, 1e-6, &failures);
    }

    // ---- reductions ------------------------------------------------------
    {
        double rsum = 0.0, rmax = -INFINITY;
        for (float x : in) {
            rsum += x;
            rmax = std::max(rmax, (double)x);
        }
        DevBuf dout4 = make_buf(dev, 4);
        float got1 = 0.0f;
        CHECK(vx_prim_reduce(q, VX_PRIM_OP_SUM, din.addr, dout4.addr, N));
        CHECK(vx_queue_flush(q));
        download(q, &got1, dout4, 4);
        char buf[32];
        snprintf(buf, sizeof(buf), "%.4f", rsum);
        std::vector<float> g{got1};
        std::vector<double> r{rsum};
        check("reduce_sum", g, r, 1e-4, 1e-4, &failures);

        CHECK(vx_prim_reduce(q, VX_PRIM_OP_MAX, din.addr, dout4.addr, N));
        CHECK(vx_queue_flush(q));
        download(q, &got1, dout4, 4);
        g[0] = got1;
        r[0] = rmax;
        check("reduce_max", g, r, 1e-6, 1e-6, &failures);

        // argmax: plant a unique max so the result is deterministic
        std::vector<float> vin = in;
        vin[123] = 100.0f;
        vin[500] = 100.0f;  // tie: expect the FIRST index (123)
        upload(q, din, vin.data(), N * 4);
        uint32_t got_idx = ~0u;
        CHECK(vx_prim_reduce(q, VX_PRIM_OP_ARGMAX, din.addr, dout4.addr, N));
        CHECK(vx_queue_flush(q));
        download(q, &got_idx, dout4, 4);
        printf("%-14s got=%u expect=123\n", "reduce_argmax", got_idx);
        if (got_idx != 123) ++failures;
        vx_buffer_release(dout4.h);
    }

    // ---- softmax -----------------------------------------------------------
    {
        const uint32_t R = 12, C = 65;  // C not a multiple of 16
        std::vector<float> sin((size_t)R * C);
        for (auto& x : sin) x = frand();
        sin[0] = 50.0f;  // large value exercises the max-shift
        DevBuf bin = make_buf(dev, sin.size() * 4);
        DevBuf bout = make_buf(dev, sin.size() * 4);
        upload(q, bin, sin.data(), sin.size() * 4);
        std::vector<float> sgot(sin.size());
        CHECK(vx_prim_softmax(q, bin.addr, bout.addr, R, C));
        CHECK(vx_queue_flush(q));
        download(q, sgot.data(), bout, sgot.size() * 4);
        std::vector<double> sref;
        ref::softmax(sin, R, C, sref);
        check("softmax", sgot, sref, 2e-5, 1e-6, &failures);
        vx_buffer_release(bin.h);
        vx_buffer_release(bout.h);
    }

    // ---- layernorm / rmsnorm ------------------------------------------------
    {
        const uint32_t R = 8, C = 33;
        std::vector<float> nin((size_t)R * C), gamma(C), beta(C);
        for (auto& x : nin) x = frand();
        for (auto& x : gamma) x = frand() * 0.5f + 0.75f;
        for (auto& x : beta) x = frand() * 0.1f;
        DevBuf bin = make_buf(dev, nin.size() * 4);
        DevBuf bg = make_buf(dev, C * 4);
        DevBuf bb = make_buf(dev, C * 4);
        DevBuf bout = make_buf(dev, nin.size() * 4);
        upload(q, bin, nin.data(), nin.size() * 4);
        upload(q, bg, gamma.data(), C * 4);
        upload(q, bb, beta.data(), C * 4);
        std::vector<float> ngot(nin.size());

        CHECK(vx_prim_layernorm(q, bin.addr, bg.addr, bb.addr, bout.addr,
                                R, C, 1e-5f));
        CHECK(vx_queue_flush(q));
        download(q, ngot.data(), bout, ngot.size() * 4);
        std::vector<double> nref;
        ref::layernorm(nin, gamma, beta, R, C, 1e-5, nref);
        check("layernorm", ngot, nref, 2e-4, 1e-4, &failures);

        CHECK(vx_prim_rmsnorm(q, bin.addr, bg.addr, bout.addr, R, C, 1e-5f));
        CHECK(vx_queue_flush(q));
        download(q, ngot.data(), bout, ngot.size() * 4);
        ref::rmsnorm(nin, gamma, R, C, 1e-5, nref);
        check("rmsnorm", ngot, nref, 2e-4, 1e-4, &failures);

        vx_buffer_release(bin.h);
        vx_buffer_release(bg.h);
        vx_buffer_release(bb.h);
        vx_buffer_release(bout.h);
    }

    vx_buffer_release(din.h);
    vx_buffer_release(dout.h);
    vx_prim_finalize();
    vx_queue_release(q);
    vx_device_release(dev);

    if (failures == 0) {
        printf("PASSED\n");
        return 0;
    }
    fprintf(stderr, "FAILED (%d)\n", failures);
    return 1;
}
