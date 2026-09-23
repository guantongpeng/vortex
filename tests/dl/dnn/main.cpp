// Vortex DL dnn acceptance (plan P3-02 second tier): direct conv2d,
// max/avg pooling and inference batch-norm against double-precision CPU
// references. Shapes mirror a small CNN stage, including stride-2 conv
// and padded pooling (the ResNet building blocks P5-02 needs).

#include <vortex/dnn.h>
#include <vortex2.h>

#include <cmath>
#include <cstdio>
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

static void download(vx_queue_h q, std::vector<float>& dst, const DevBuf& b) {
    vx_event_h ev = nullptr;
    vx_enqueue_read(q, dst.data(), b.h, 0, dst.size() * 4, 0, nullptr, &ev);
    vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE);
    vx_event_release(ev);
}

int main(int argc, char** argv) {
    const char* vxbin = (argc > 1) ? argv[1] : "dnn.vxbin";

    vx_device_h dev = nullptr;
    CHECK(vx_device_open(0, &dev));
    vx_queue_h q = nullptr;
    CHECK(vx_queue_create(dev, nullptr, &q));
    if (vx_dnn_init(dev, vxbin) != VX_DNN_OK) {
        fprintf(stderr, "FAILED: vx_dnn_init(%s)\n", vxbin);
        return 1;
    }

    int failures = 0;

    // ---- init argument checking ------------------------------------
    // init is idempotent only for the same device and image. It used to
    // return OK without looking at either, so a second caller with a
    // different image silently got the first caller's kernels.
    if (vx_dnn_init(dev, vxbin) != VX_DNN_OK) {
        printf("init re-init:  not idempotent for the same arguments\n");
        ++failures;
    }
    if (vx_dnn_init(dev, "/no/such/image.vxbin") !=
        VX_DNN_ERR_ALREADY_INITIALIZED) {
        printf("init other:    accepted a different image without complaint\n");
        ++failures;
    }
    printf("init checks:   ok\n");

    // ---- conv2d: 1x8x9x9 -> 16x7x7, 3x3 pad 1 stride 1, with bias -----
    {
        const uint32_t N = 1, CI = 8, HI = 9, WI = 9, CO = 16;
        const uint32_t KH = 3, KW = 3, P = 1, S = 1;
        const uint32_t HO = (HI + 2 * P - KH) / S + 1;  // 9
        const uint32_t WO = (WI + 2 * P - KW) / S + 1;  // 9

        std::vector<float> in((size_t)N * CI * HI * WI);
        std::vector<float> w((size_t)CO * CI * KH * KW);
        std::vector<float> bias(CO);
        for (auto& x : in) x = frand();
        for (auto& x : w) x = frand() * 0.5f;
        for (auto& x : bias) x = frand() * 0.1f;

        DevBuf bi = make_buf(dev, in.size() * 4);
        DevBuf bw = make_buf(dev, w.size() * 4);
        DevBuf bb = make_buf(dev, bias.size() * 4);
        DevBuf bo = make_buf(dev, (size_t)N * CO * HO * WO * 4);
        upload(q, bi, in.data(), in.size() * 4);
        upload(q, bw, w.data(), w.size() * 4);
        upload(q, bb, bias.data(), bias.size() * 4);

        CHECK(vx_dnn_conv2d(q, bi.addr, bw.addr, bb.addr, bo.addr,
                            N, CI, HI, WI, CO, KH, KW, P, P, S, S, 1));
        CHECK(vx_queue_flush(q));
        std::vector<float> got((size_t)N * CO * HO * WO);
        download(q, got, bo);

        uint32_t bad = 0;
        double maxrel = 0.0;
        for (uint32_t n = 0; n < N; ++n)
            for (uint32_t co = 0; co < CO; ++co)
                for (uint32_t oy = 0; oy < HO; ++oy)
                    for (uint32_t ox = 0; ox < WO; ++ox) {
                        double ref = bias[co];
                        for (uint32_t ci = 0; ci < CI; ++ci)
                            for (uint32_t kh = 0; kh < KH; ++kh)
                                for (uint32_t kw = 0; kw < KW; ++kw) {
                                    int iy = (int)(oy * S + kh) - (int)P;
                                    int ix = (int)(ox * S + kw) - (int)P;
                                    if (iy < 0 || iy >= (int)HI || ix < 0 ||
                                        ix >= (int)WI)
                                        continue;
                                    ref += (double)in[((size_t)n * CI + ci) * HI * WI + (size_t)iy * WI + ix] *
                                           (double)w[((size_t)co * CI + ci) * KH * KW + kh * KW + kw];
                                }
                        double gotv = got[((size_t)n * CO + co) * HO * WO + (size_t)oy * WO + ox];
                        double rel = std::fabs(gotv - ref) / (std::fabs(ref) + 1e-3);
                        if (rel > maxrel) maxrel = rel;
                        if (rel > 2e-3) ++bad;
                    }
        printf("conv2d 3x3p1:  max_rel=%.2e bad=%u (%ux%ux%u -> %ux%ux%u)\n",
               maxrel, bad, CI, HI, WI, CO, HO, WO);
        if (bad) ++failures;

        // ---- max pool 3x3 s2 p1 + avg pool 2x2 s2 ---------------------
        const uint32_t PK = 3, PP = 1, PS = 2;
        const uint32_t PH2 = (HO + 2 * PP - PK) / PS + 1;  // 5
        const uint32_t PW2 = (WO + 2 * PP - PK) / PS + 1;
        DevBuf bp = make_buf(dev, (size_t)N * CO * PH2 * PW2 * 4);
        CHECK(vx_dnn_pool2d(q, bo.addr, bp.addr, N, CO, HO, WO, PK, PK, PP, PP,
                            PS, PS, 0));
        CHECK(vx_queue_flush(q));
        std::vector<float> gotp((size_t)N * CO * PH2 * PW2);
        download(q, gotp, bp);

        uint32_t badp = 0;
        for (uint32_t n = 0; n < N; ++n)
            for (uint32_t c = 0; c < CO; ++c)
                for (uint32_t oy = 0; oy < PH2; ++oy)
                    for (uint32_t ox = 0; ox < PW2; ++ox) {
                        float ref = -3.4e38f;
                        for (uint32_t kh = 0; kh < PK; ++kh) {
                            int iy = (int)(oy * PS + kh) - (int)PP;
                            if (iy < 0 || iy >= (int)HO) continue;
                            for (uint32_t kw = 0; kw < PK; ++kw) {
                                int ix = (int)(ox * PS + kw) - (int)PP;
                                if (ix < 0 || ix >= (int)WO) continue;
                                float v = got[((size_t)n * CO + c) * HO * WO + (size_t)iy * WO + ix];
                                if (v > ref) ref = v;
                            }
                        }
                        if (gotp[((size_t)n * CO + c) * PH2 * PW2 + (size_t)oy * PW2 + ox] != ref) ++badp;
                    }
        printf("maxpool 3x3s2: bad=%u\n", badp);
        if (badp) ++failures;

        // ---- batch norm on the conv output -----------------------------
        std::vector<float> mean(CO), var(CO), gw(CO), gb(CO);
        for (uint32_t c = 0; c < CO; ++c) {
            mean[c] = frand() * 0.1f;
            var[c] = 0.5f + (float)c * 0.01f;
            gw[c] = 0.8f + frand() * 0.4f;
            gb[c] = frand() * 0.1f;
        }
        DevBuf bm = make_buf(dev, CO * 4), br = make_buf(dev, CO * 4);
        DevBuf bgw = make_buf(dev, CO * 4), bgb = make_buf(dev, CO * 4);
        upload(q, bm, mean.data(), CO * 4);
        upload(q, br, var.data(), CO * 4);
        upload(q, bgw, gw.data(), CO * 4);
        upload(q, bgb, gb.data(), CO * 4);
        CHECK(vx_dnn_bn_affine(q, bo.addr, bm.addr, br.addr, bgw.addr,
                               bgb.addr, bo.addr, N, CO, HO * WO, 1e-5f));
        CHECK(vx_queue_flush(q));
        std::vector<float> gotbn((size_t)N * CO * HO * WO);
        download(q, gotbn, bo);

        uint32_t badbn = 0;
        for (size_t i = 0; i < gotbn.size(); ++i) {
            uint32_t c = (i / (HO * WO)) % CO;
            const double r = 1.0 / std::sqrt((double)var[c] + 1e-5);
            double ref = ((double)got[i] - mean[c]) * r * gw[c] + gb[c];
            if (std::fabs(gotbn[i] - ref) > 1e-5 * (std::fabs(ref) + 1.0)) ++badbn;
        }
        printf("bn_affine:     bad=%u\n", badbn);
        if (badbn) ++failures;

        // ---- batch norm with N > 1 ------------------------------------
        // The per-channel span must be H*W. Deriving it as total/c gives
        // N*H*W and selects the right channel only when N == 1, which is why
        // the check above -- running at N == 1 -- could never see it.
        {
            const uint32_t BN = 2, BC = 3, BH = 2, BW = 2;
            const uint32_t span = BH * BW, total_bn = BN * BC * span;
            std::vector<float> ib(total_bn), mb(BC), vb(BC), wb(BC), bb(BC);
            for (uint32_t c = 0; c < BC; ++c) {
                mb[c] = 0.1f * (float)c;
                vb[c] = 0.5f + 0.25f * (float)c;
                wb[c] = 1.0f + 0.5f * (float)c;
                bb[c] = -0.2f * (float)c;
            }
            for (uint32_t i = 0; i < total_bn; ++i) ib[i] = 0.01f * (float)i;
            DevBuf di = make_buf(dev, total_bn * 4), dm = make_buf(dev, BC * 4);
            DevBuf dv = make_buf(dev, BC * 4), dw = make_buf(dev, BC * 4);
            DevBuf db = make_buf(dev, BC * 4), dout = make_buf(dev, total_bn * 4);
            upload(q, di, ib.data(), total_bn * 4);
            upload(q, dm, mb.data(), BC * 4);
            upload(q, dv, vb.data(), BC * 4);
            upload(q, dw, wb.data(), BC * 4);
            upload(q, db, bb.data(), BC * 4);
            CHECK(vx_dnn_bn_affine(q, di.addr, dm.addr, dv.addr, dw.addr,
                                   db.addr, dout.addr, BN, BC, span, 1e-5f));
            CHECK(vx_queue_flush(q));
            std::vector<float> got_b(total_bn);
            download(q, got_b, dout);
            uint32_t bad_b = 0;
            for (uint32_t i = 0; i < total_bn; ++i) {
                const uint32_t ch = (i / span) % BC;   // NOT total/BC
                const double r = 1.0 / std::sqrt((double)vb[ch] + 1e-5);
                const double ref = ((double)ib[i] - mb[ch]) * r * wb[ch] + bb[ch];
                if (std::fabs(got_b[i] - ref) > 1e-5 * (std::fabs(ref) + 1.0)) {
                    ++bad_b;
                }
            }
            printf("bn_affine N=2: bad=%u\n", bad_b);
            if (bad_b) ++failures;
        }

        // ---- batch norm with no affine --------------------------------
        // weight and bias are optional; the caller signals that by passing
        // null for both rather than buffers of ones and zeros.
        {
            const uint32_t BN = 2, BC = 3, BH = 2, BW = 2;
            const uint32_t span = BH * BW, total_bn = BN * BC * span;
            std::vector<float> ib(total_bn), mb(BC), vb(BC);
            for (uint32_t c = 0; c < BC; ++c) {
                mb[c] = 0.1f * (float)c;
                vb[c] = 0.5f + 0.25f * (float)c;
            }
            for (uint32_t i = 0; i < total_bn; ++i) ib[i] = 0.01f * (float)i;
            DevBuf di = make_buf(dev, total_bn * 4), dm = make_buf(dev, BC * 4);
            DevBuf dv = make_buf(dev, BC * 4), dout = make_buf(dev, total_bn * 4);
            upload(q, di, ib.data(), total_bn * 4);
            upload(q, dm, mb.data(), BC * 4);
            upload(q, dv, vb.data(), BC * 4);
            CHECK(vx_dnn_bn_affine(q, di.addr, dm.addr, dv.addr, 0, 0,
                                   dout.addr, BN, BC, span, 1e-5f));
            CHECK(vx_queue_flush(q));
            std::vector<float> got_na(total_bn);
            download(q, got_na, dout);
            uint32_t bad_na = 0;
            for (uint32_t i = 0; i < total_bn; ++i) {
                const uint32_t ch = (i / span) % BC;
                const double r = 1.0 / std::sqrt((double)vb[ch] + 1e-5);
                const double ref = ((double)ib[i] - mb[ch]) * r;
                if (std::fabs(got_na[i] - ref) > 1e-5 * (std::fabs(ref) + 1.0)) {
                    ++bad_na;
                }
            }
            printf("bn_affine none:bad=%u\n", bad_na);
            if (bad_na) ++failures;
            // one without the other is a caller mistake, not a request
            if (vx_dnn_bn_affine(q, di.addr, dm.addr, dv.addr, dm.addr, 0,
                                 dout.addr, BN, BC, span, 1e-5f) == 0) {
                printf("bn_affine half: accepted a weight without a bias\n");
                ++failures;
            }
        }

        // ---- max pool over -inf and NaN -------------------------------
        // The accumulator used to start at the finite -3.4e38f, so an all
        // -inf window came back as that sentinel and a NaN window lost the
        // NaN. Neither is visible to a test that only uses finite inputs.
        {
            const uint32_t PN = 1, PC = 2, PH = 2, PW = 2;
            // channel 0 is entirely -inf (so its window is a real all -inf
            // window); channel 1 contains a NaN
            std::vector<float> pin = {-INFINITY, -INFINITY, -INFINITY, -INFINITY,
                                      NAN, 0.0f, 3.0f, 4.0f};
            DevBuf dpi = make_buf(dev, PN * PC * PH * PW * 4);
            DevBuf dpo = make_buf(dev, PN * PC * 1 * 1 * 4);
            upload(q, dpi, pin.data(), PN * PC * PH * PW * 4);
            // (q, in, out, n, c, hi, wi, kh, kw, ph, pw, sh, sw, op)
            CHECK(vx_dnn_pool2d(q, dpi.addr, dpo.addr, PN, PC, PH, PW,
                                2, 2, 0, 0, 2, 2, 0));
            CHECK(vx_queue_flush(q));
            std::vector<float> pout(PN * PC);
            download(q, pout, dpo);
            uint32_t bad_p = 0;
            if (pout[0] != -INFINITY) ++bad_p;      // all -inf window
            if (!std::isnan(pout[1])) ++bad_p;      // NaN must propagate
            printf("maxpool spec:  bad=%u (got %g, %g)\n", bad_p, pout[0], pout[1]);
            if (bad_p) ++failures;
        }

        // ---- stride-2 conv (downsampling path) -------------------------
        const uint32_t S2 = 2;
        const uint32_t HS = (HI + 2 * P - KH) / S2 + 1;  // 5
        const uint32_t WS = (WI + 2 * P - KW) / S2 + 1;
        DevBuf bos = make_buf(dev, (size_t)N * CO * HS * WS * 4);
        CHECK(vx_dnn_conv2d(q, bi.addr, bw.addr, 0, bos.addr,
                            N, CI, HI, WI, CO, KH, KW, P, P, S2, S2, 1));
        CHECK(vx_queue_flush(q));
        std::vector<float> gots((size_t)N * CO * HS * WS);
        download(q, gots, bos);
        uint32_t bads = 0;
        double maxrels = 0.0;
        for (uint32_t co = 0; co < CO; ++co)
            for (uint32_t oy = 0; oy < HS; ++oy)
                for (uint32_t ox = 0; ox < WS; ++ox) {
                    double ref = 0.0;
                    for (uint32_t ci = 0; ci < CI; ++ci)
                        for (uint32_t kh = 0; kh < KH; ++kh)
                            for (uint32_t kw = 0; kw < KW; ++kw) {
                                int iy = (int)(oy * S2 + kh) - (int)P;
                                int ix = (int)(ox * S2 + kw) - (int)P;
                                if (iy < 0 || iy >= (int)HI || ix < 0 || ix >= (int)WI) continue;
                                ref += (double)in[(size_t)ci * HI * WI + (size_t)iy * WI + ix] *
                                       (double)w[((size_t)co * CI + ci) * KH * KW + kh * KW + kw];
                            }
                    double gotv = gots[(size_t)co * HS * WS + (size_t)oy * WS + ox];
                    double rel = std::fabs(gotv - ref) / (std::fabs(ref) + 1e-3);
                    if (rel > maxrels) maxrels = rel;
                    if (rel > 2e-3) ++bads;
                }
        printf("conv2d s2:     max_rel=%.2e bad=%u\n", maxrels, bads);
        if (bads) ++failures;

        for (DevBuf* b : {&bi, &bw, &bb, &bo, &bp, &bm, &br, &bgw, &bgb, &bos}) {
            vx_buffer_release(b->h);
        }
    }

    vx_dnn_finalize();
    vx_queue_release(q);
    vx_device_release(dev);

    if (failures == 0) {
        printf("PASSED\n");
        return 0;
    }
    fprintf(stderr, "FAILED (%d)\n", failures);
    return 1;
}
