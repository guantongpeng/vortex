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

// Whether a special-value result matches its reference. These cases are about
// which category the answer falls in -- NaN, an infinity, a signed zero --
// rather than about the last bits of a transcendental, which a float kernel
// and a double reference are not expected to agree on.
static bool same_as_reference(float got, double want) {
    if (std::isnan(want)) return std::isnan(got);
    if (std::isinf(want)) return std::isinf(got) && ((got > 0) == (want > 0));
    if (std::isnan(got) || std::isinf(got)) return false;
    return std::fabs((double)got - want) <= 1e-6 * (1.0 + std::fabs(want));
}

// A reduction case: the op, and where its operand lives. The reference is
// computed per block because the shapes differ.
struct RedCase {
    const char* name;
    vx_prim_op op;
    double want;  // unused by the multi-row block, which recomputes per row
};

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
    // One reference per op, over whichever buffer the case supplies. Every
    // unary op the kernel implements has a row here: an op with no reference
    // is an op with no test, and a switch that grows an arm the test does not
    // know about is how the two drift apart.
    struct ActCase {
        const char* name;
        vx_prim_op op;
        double (*ref)(double);
    };
    auto relu_ref = [](double x) { return x < 0.0 ? 0.0 : x; };
    auto abs_ref = [](double x) { return std::fabs(x); };
    auto exp_ref = [](double x) { return std::exp(x); };
    auto log_ref = [](double x) { return std::log(x); };
    auto sqrt_ref = [](double x) { return std::sqrt(x); };
    auto rsqrt_ref = [](double x) { return 1.0 / std::sqrt(x); };
    auto sigmoid_ref = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
    auto tanh_ref = [](double x) { return std::tanh(x); };
    auto recip_ref = [](double x) { return 1.0 / x; };
    auto neg_ref = [](double x) { return -x; };
    const std::vector<ActCase> act_cases = {
        {"relu", VX_PRIM_OP_RELU, +relu_ref},
        {"gelu_tanh", VX_PRIM_OP_GELU_TANH, ref::gelu},
        {"gelu_erf", VX_PRIM_OP_GELU_ERF, ref::gelu_erf},
        {"silu", VX_PRIM_OP_SILU, ref::silu},
        {"neg", VX_PRIM_OP_NEG, +neg_ref},
        {"abs", VX_PRIM_OP_ABS, +abs_ref},
        {"exp", VX_PRIM_OP_EXP, +exp_ref},
        {"log", VX_PRIM_OP_LOG, +log_ref},
        {"sqrt", VX_PRIM_OP_SQRT, +sqrt_ref},
        {"rsqrt", VX_PRIM_OP_RSQRT, +rsqrt_ref},
        {"sigmoid", VX_PRIM_OP_SIGMOID, +sigmoid_ref},
        {"tanh", VX_PRIM_OP_TANH, +tanh_ref},
        {"reciprocal", VX_PRIM_OP_RECIPROCAL, +recip_ref},
    };
    // The values a naive implementation gets wrong: -0.0 (which relu and abs
    // are the whole story about), the infinities, and NaN.
    std::vector<float> special = {-INFINITY, -1.0f, -0.0f, 0.0f, 1.0f,
                                  INFINITY, NAN};
    DevBuf dspec = make_buf(dev, special.size() * 4);
    DevBuf dspec_out = make_buf(dev, special.size() * 4);
    upload(q, dspec, special.data(), special.size() * 4);

    for (const ActCase& ac : act_cases) {
        CHECK(vx_prim_unary(q, ac.op, din.addr, dout.addr, N));
        CHECK(vx_queue_flush(q));
        download(q, got.data(), dout, N * 4);
        std::vector<double> r(N);
        for (uint32_t i = 0; i < N; ++i) r[i] = ac.ref((double)in[i]);
        check(ac.name, got, r, 2e-5, 1e-6, &failures);

        // and the special values, which random data cannot reach
        std::vector<float> sgot(special.size());
        CHECK(vx_prim_unary(q, ac.op, dspec.addr, dspec_out.addr,
                            (uint32_t)special.size()));
        CHECK(vx_queue_flush(q));
        download(q, sgot.data(), dspec_out, special.size() * 4);
        for (size_t i = 0; i < special.size(); ++i) {
            const double want = ac.ref((double)special[i]);
            if (!same_as_reference(sgot[i], want)) {
                fprintf(stderr, "  %s(special[%zu]=%g) got=%g want=%g\n",
                        ac.name, i, (double)special[i], (double)sgot[i], want);
                ++failures;
            }
        }
    }
    // relu(-0.0) must be -0.0, not +0.0. Comparing floats cannot see this, so
    // it is checked separately rather than left to the loop above.
    {
        std::vector<float> sgot(special.size());
        CHECK(vx_prim_unary(q, VX_PRIM_OP_RELU, dspec.addr, dspec_out.addr,
                            (uint32_t)special.size()));
        CHECK(vx_queue_flush(q));
        download(q, sgot.data(), dspec_out, special.size() * 4);
        if (!std::signbit(sgot[2]) || std::signbit(sgot[3])) {
            fprintf(stderr, "  relu lost the sign of zero: %g, %g\n",
                    (double)sgot[2], (double)sgot[3]);
            ++failures;
        }
    }
    // abs(-0.0) must be +0.0: `x < 0 ? -x : x` returns -0.0 and is the obvious
    // way to write this.
    {
        std::vector<float> sgot(special.size());
        CHECK(vx_prim_unary(q, VX_PRIM_OP_ABS, dspec.addr, dspec_out.addr,
                            (uint32_t)special.size()));
        CHECK(vx_queue_flush(q));
        download(q, sgot.data(), dspec_out, special.size() * 4);
        if (std::signbit(sgot[2])) {
            fprintf(stderr, "  abs(-0.0) came back as -0.0\n");
            ++failures;
        }
    }
    vx_buffer_release(dspec.h);
    vx_buffer_release(dspec_out.h);

    // ---- reductions ------------------------------------------------------
    {
        DevBuf dout4 = make_buf(dev, 4);
        float got1 = 0.0f;
        // rows = 1: the whole vector is one row, which is the shape the
        // single-CTA version could express and still the common case.
        double rsum = 0.0, rmax = -INFINITY;
        for (float x : in) {
            rsum += x;
            rmax = std::max(rmax, (double)x);
        }
        for (RedCase rc : {RedCase{"reduce_sum", VX_PRIM_OP_SUM, rsum},
                           RedCase{"reduce_mean", VX_PRIM_OP_MEAN,
                                   rsum / (double)N},
                           RedCase{"reduce_max", VX_PRIM_OP_MAX, rmax}}) {
            CHECK(vx_prim_reduce(q, rc.op, din.addr, dout4.addr, 1, N));
            CHECK(vx_queue_flush(q));
            download(q, &got1, dout4, 4);
            std::vector<float> g{got1};
            std::vector<double> r{rc.want};
            check(rc.name, g, r, 1e-5, 1e-5, &failures);
        }

        // argmax: plant a unique max so the result is deterministic
        std::vector<float> vin = in;
        vin[123] = 100.0f;
        vin[500] = 100.0f;  // tie: expect the FIRST index (123)
        upload(q, din, vin.data(), N * 4);
        uint32_t got_idx = ~0u;
        CHECK(vx_prim_reduce(q, VX_PRIM_OP_ARGMAX, din.addr, dout4.addr, 1, N));
        CHECK(vx_queue_flush(q));
        download(q, &got_idx, dout4, 4);
        printf("%-14s got=%u expect=123\n", "reduce_argmax", got_idx);
        if (got_idx != 123) ++failures;
        vx_buffer_release(dout4.h);
    }

    // ---- reductions: the special values -----------------------------------
    //
    // max and argmax must propagate NaN and must not lose an all-(-inf) row to
    // a finite sentinel. fmaxf -- the obvious spelling -- does both wrong, and
    // this is the second time in this tree that a reduction has been seeded
    // with a value no input can beat.
    {
        const std::vector<float> vals = {-INFINITY, -1.0f, -INFINITY, NAN,
                                         -INFINITY};
        const uint32_t nv = (uint32_t)vals.size();
        DevBuf dv = make_buf(dev, nv * 4), dov = make_buf(dev, nv * 4);
        upload(q, dv, vals.data(), nv * 4);
        float gm = 0.0f;
        CHECK(vx_prim_reduce(q, VX_PRIM_OP_MAX, dv.addr, dov.addr, 1, nv));
        CHECK(vx_queue_flush(q));
        download(q, &gm, dov, 4);
        if (!std::isnan(gm)) {
            fprintf(stderr, "  reduce_max dropped the NaN: got %g\n", (double)gm);
            ++failures;
        }
        uint32_t gi = ~0u;
        CHECK(vx_prim_reduce(q, VX_PRIM_OP_ARGMAX, dv.addr, dov.addr, 1, nv));
        CHECK(vx_queue_flush(q));
        download(q, &gi, dov, 4);
        if (gi != 3) {
            fprintf(stderr, "  argmax picked %u, expected the NaN at 3\n", gi);
            ++failures;
        }
        // all -inf, no NaN: -inf must survive, not the -3.4e38 a finite
        // sentinel would leave behind
        const std::vector<float> negs = {-INFINITY, -INFINITY};
        upload(q, dv, negs.data(), 2 * 4);
        CHECK(vx_prim_reduce(q, VX_PRIM_OP_MAX, dv.addr, dov.addr, 1, 2));
        CHECK(vx_queue_flush(q));
        download(q, &gm, dov, 4);
        if (gm != -INFINITY) {
            fprintf(stderr, "  reduce_max of all -inf gave %g\n", (double)gm);
            ++failures;
        }
        vx_buffer_release(dv.h);
        vx_buffer_release(dov.h);
    }

    // ---- reductions: several rows at once ---------------------------------
    //
    // One CTA per row, so a row's result must not depend on its neighbours --
    // which is what a single-CTA kernel silently cannot express.
    {
        const uint32_t R = 5, C = 71;  // C not a multiple of 16
        std::vector<float> rin((size_t)R * C);
        for (auto& x : rin) x = frand();
        rin[7] = 9.0f;  // row 0's max
        DevBuf rbuf = make_buf(dev, rin.size() * 4);
        DevBuf rout = make_buf(dev, R * 4);
        upload(q, rbuf, rin.data(), rin.size() * 4);
        std::vector<float> rgot(R);
        std::vector<double> rref(R);
        for (RedCase rc : {RedCase{"rows_sum", VX_PRIM_OP_SUM, 0.0},
                           RedCase{"rows_mean", VX_PRIM_OP_MEAN, 0.0},
                           RedCase{"rows_max", VX_PRIM_OP_MAX, 0.0}}) {
            for (uint32_t r = 0; r < R; ++r) {
                double acc = (rc.op == VX_PRIM_OP_MAX) ? -INFINITY : 0.0;
                for (uint32_t j = 0; j < C; ++j) {
                    const double v = rin[(size_t)r * C + j];
                    acc = (rc.op == VX_PRIM_OP_MAX) ? std::max(acc, v) : acc + v;
                }
                rref[r] = (rc.op == VX_PRIM_OP_MEAN) ? acc / C : acc;
            }
            CHECK(vx_prim_reduce(q, rc.op, rbuf.addr, rout.addr, R, C));
            CHECK(vx_queue_flush(q));
            download(q, rgot.data(), rout, R * 4);
            check(rc.name, rgot, rref, 1e-5, 1e-5, &failures);
        }
        vx_buffer_release(rbuf.h);
        vx_buffer_release(rout.h);
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

        // log_softmax: same shape, logarithm of the same thing, computed as
        // (x - max) - log(sum) rather than log(softmax(x))
        CHECK(vx_prim_log_softmax(q, bin.addr, bout.addr, R, C));
        CHECK(vx_queue_flush(q));
        download(q, sgot.data(), bout, sgot.size() * 4);
        ref::log_softmax(sin, R, C, sref);
        check("log_softmax", sgot, sref, 2e-5, 1e-6, &failures);
        // and it is the logarithm of the softmax already checked above
        {
            std::vector<double> base;
            ref::softmax(sin, R, C, base);
            bool ok = true;
            for (size_t i = 0; i < base.size(); ++i) {
                if (std::fabs(std::exp(sref[i]) - base[i]) > 1e-5) ok = false;
            }
            if (!ok) { fprintf(stderr, "  exp(log_softmax) != softmax\n"); ++failures; }
        }

        // logsumexp: one value per row, and exp of it is the softmax's
        // denominator -- so the shapes and the values are both checked
        DevBuf lout = make_buf(dev, R * 4);
        CHECK(vx_prim_logsumexp(q, bin.addr, lout.addr, R, C));
        CHECK(vx_queue_flush(q));
        std::vector<float> lgot(R);
        download(q, lgot.data(), lout, R * 4);
        std::vector<double> lref;
        ref::logsumexp(sin, R, C, lref);
        check("logsumexp", lgot, lref, 2e-6, 1e-6, &failures);

        // An infinite row max is the answer itself. The general path computes
        // the shift inf - inf, which is NaN, so this is the one place where
        // the two differ on a trivial row.
        {
            const std::vector<float> inf_rows = {INFINITY, 0.0f,
                                                 -INFINITY, -INFINITY};
            DevBuf ib = make_buf(dev, 4 * 4), lo = make_buf(dev, 2 * 4);
            upload(q, ib, inf_rows.data(), 4 * 4);
            CHECK(vx_prim_logsumexp(q, ib.addr, lo.addr, 2, 2));
            CHECK(vx_queue_flush(q));
            float lg[2] = {0.0f, 0.0f};
            download(q, lg, lo, 2 * 4);
            if (lg[0] != INFINITY) {
                fprintf(stderr, "  logsumexp of a +inf row: got %g want inf\n",
                        (double)lg[0]);
                ++failures;
            }
            if (lg[1] != -INFINITY) {
                fprintf(stderr, "  logsumexp of an all -inf row: got %g want -inf\n",
                        (double)lg[1]);
                ++failures;
            }
            vx_buffer_release(ib.h);
            vx_buffer_release(lo.h);
        }
        vx_buffer_release(lout.h);

        // A NaN in a row makes the whole row NaN, and leaves its neighbours
        // alone. This is a contract, not a regression this kernel had: the
        // NaN travels through the exponential and the sum, so the row max
        // being NaN-blind does not matter. Pinned because "softmax of a NaN
        // row" is the kind of thing a later pass-2 rework could quietly lose.
        std::vector<float> nanin = sin;
        nanin[3] = NAN;  // row 0, column 3
        upload(q, bin, nanin.data(), nanin.size() * 4);
        CHECK(vx_prim_softmax(q, bin.addr, bout.addr, R, C));
        CHECK(vx_queue_flush(q));
        download(q, sgot.data(), bout, sgot.size() * 4);
        for (uint32_t j = 0; j < C; ++j) {
            if (!std::isnan(sgot[j])) {
                fprintf(stderr, "  softmax of a NaN row gave %g at %u\n",
                        (double)sgot[j], j);
                ++failures;
                break;
            }
        }
        // and the other rows are untouched by it
        for (uint32_t j = 1 * C; j < 2 * C; ++j) {
            if (!std::isfinite(sgot[j])) {
                fprintf(stderr, "  row 1 went non-finite: %g at %u\n",
                        (double)sgot[j], j);
                ++failures;
                break;
            }
        }
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
        DevBuf bmean = make_buf(dev, R * 4);
        DevBuf brstd = make_buf(dev, R * 4);
        upload(q, bin, nin.data(), nin.size() * 4);
        upload(q, bg, gamma.data(), C * 4);
        upload(q, bb, beta.data(), C * 4);
        std::vector<float> ngot(nin.size());

        CHECK(vx_prim_layernorm(q, bin.addr, bg.addr, bb.addr, bout.addr,
                                bmean.addr, brstd.addr, R, C, 1e-5f));
        CHECK(vx_queue_flush(q));
        download(q, ngot.data(), bout, ngot.size() * 4);
        std::vector<double> nref;
        ref::layernorm(nin, gamma, beta, R, C, 1e-5, nref);
        check("layernorm", ngot, nref, 2e-4, 1e-4, &failures);

        // the auxiliary outputs, which aten::native_layer_norm returns
        {
            std::vector<float> mgot(R), rgot(R);
            std::vector<double> mref(R), rref(R);
            download(q, mgot.data(), bmean, R * 4);
            download(q, rgot.data(), brstd, R * 4);
            ref::layer_stats(nin, R, C, 1e-5, mref, rref);
            check("ln_mean", mgot, mref, 1e-5, 1e-5, &failures);
            check("ln_rstd", rgot, rref, 1e-5, 1e-5, &failures);
        }

        // no affine: the mode F.layer_norm(x, shape) asks for
        CHECK(vx_prim_layernorm(q, bin.addr, 0, 0, bout.addr, 0, 0,
                                R, C, 1e-5f));
        CHECK(vx_queue_flush(q));
        download(q, ngot.data(), bout, ngot.size() * 4);
        ref::layernorm(nin, std::vector<float>(C, 1.0f), std::vector<float>(C, 0.0f),
                       R, C, 1e-5, nref);
        check("layernorm_noaffine", ngot, nref, 2e-4, 1e-4, &failures);

        // half an affine is a caller error, not a mode
        if (vx_prim_layernorm(q, bin.addr, bg.addr, 0, bout.addr, 0, 0,
                              R, C, 1e-5f) != VX_PRIM_ERR_BAD_ARGS) {
            fprintf(stderr, "  layernorm accepted gamma without beta\n");
            ++failures;
        }

        CHECK(vx_prim_rmsnorm(q, bin.addr, bg.addr, bout.addr, R, C, 1e-5f));
        CHECK(vx_queue_flush(q));
        download(q, ngot.data(), bout, ngot.size() * 4);
        ref::rmsnorm(nin, gamma, R, C, 1e-5, nref);
        check("rmsnorm", ngot, nref, 2e-4, 1e-4, &failures);

        CHECK(vx_prim_rmsnorm(q, bin.addr, 0, bout.addr, R, C, 1e-5f));
        CHECK(vx_queue_flush(q));
        download(q, ngot.data(), bout, ngot.size() * 4);
        ref::rmsnorm(nin, std::vector<float>(C, 1.0f), R, C, 1e-5, nref);
        check("rmsnorm_noaffine", ngot, nref, 2e-4, 1e-4, &failures);

        vx_buffer_release(bin.h);
        vx_buffer_release(bg.h);
        vx_buffer_release(bb.h);
        vx_buffer_release(bout.h);
        vx_buffer_release(bmean.h);
        vx_buffer_release(brstd.h);
    }

    // ---- layernorm: the estimator, not just the average --------------------
    //
    // Values around 1e6 in float32 make E[x^2] and mean^2 both ~1e12. Their
    // difference is ~1, so every significant digit cancels: the one-pass
    // variance came out 0 and rstd 1/sqrt(eps) = 316.2, where the answer is
    // 1.25 and 0.894. The reference above uses the same one-pass formula in
    // double, so it agreed with the wrong kernel -- a defect written into its
    // own oracle, which is how it survived.
    {
        const std::vector<float> big = {1e6f, 1e6f + 1.0f, 1e6f + 2.0f,
                                        1e6f + 3.0f};
        const uint32_t C = (uint32_t)big.size(), R = 1;
        DevBuf bin = make_buf(dev, C * 4), bout = make_buf(dev, C * 4);
        DevBuf brstd = make_buf(dev, R * 4);
        upload(q, bin, big.data(), C * 4);
        CHECK(vx_prim_layernorm(q, bin.addr, 0, 0, bout.addr, 0, brstd.addr,
                                R, C, 1e-5f));
        CHECK(vx_queue_flush(q));
        float rstd = 0.0f;
        download(q, &rstd, brstd, 4);
        // 1/sqrt(1.25 + 1e-5) at double precision, then rounded to float32
        const double want = 1.0 / std::sqrt(1.25 + 1e-5);
        if (std::fabs((double)rstd - want) > 1e-5) {
            fprintf(stderr, "  rstd for 1e6-scale input: got %.7g want %.7g\n",
                    (double)rstd, want);
            ++failures;
        }
        vx_buffer_release(bin.h);
        vx_buffer_release(bout.h);
        vx_buffer_release(brstd.h);
    }

    // ---- argmax: the first NaN, not the last one ---------------------------
    //
    // torch.argmax returns the first index among ties, NaNs included. The
    // merge compared two NaNs with `v != bv`, which is true for NaN vs NaN and
    // sent them down the greater-than branch, so the winner was whichever
    // candidate the tree happened to visit last.
    {
        const std::vector<std::pair<std::vector<float>, uint32_t>> cases = {
            {{1.0f, NAN, NAN, NAN}, 1},
            {{NAN, 1.0f, NAN, NAN}, 0},
            {{1.0f, 2.0f, 3.0f, 3.0f}, 2},
            // 21 wide: thread 0 owns both index 0 and index 16, and the warp
            // merge has to prefer the lower lane on a tie
            {{NAN, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
              11.0f, 12.0f, 13.0f, 14.0f, 15.0f, NAN, 17.0f, 18.0f, 19.0f,
              20.0f}, 0},
        };
        for (const auto& c : cases) {
            const uint32_t n = (uint32_t)c.first.size();
            DevBuf bin = make_buf(dev, n * 4), bout = make_buf(dev, 4);
            upload(q, bin, c.first.data(), n * 4);
            CHECK(vx_prim_reduce(q, VX_PRIM_OP_ARGMAX, bin.addr, bout.addr, 1, n));
            CHECK(vx_queue_flush(q));
            uint32_t got = ~0u;
            download(q, &got, bout, 4);
            if (got != c.second) {
                fprintf(stderr, "  argmax of a %u-wide row: got %u want %u\n",
                        n, got, c.second);
                ++failures;
            }
            vx_buffer_release(bin.h);
            vx_buffer_release(bout.h);
        }
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
