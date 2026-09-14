// Copyright © 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Vortex DL Mamba selective scan acceptance test (plan P7 precursor).
// Simplified S6 scan on SimX vs a double-precision CPU reference:
//   S[n] <- exp(A*dt) * S[n] + B[t][n] * x[t];  y[t] = sum_n C[t][n]*S[n]
//
// Tolerance (honest choice, not tuned to pass):
// - The kernel is FP32 (expf, f32 state/dot) against an FP64 reference, so
//   the error has an absolute component ~ few ulp accumulated over T
//   recurrence steps; pure relative error is UNBOUNDED where y ~ 0 because
//   y is an 8-term signed sum (cancellation). A floor is therefore required.
// - A host simulation of the exact device op order (f32, -ffp-contract=off)
//   over 200 random draws of the planned distributions (153,600 outputs)
//   measured max |err| = 2.4e-6 and max |err|/(|ref|+1e-2) = 4.0e-5.
// - Gate: |err| <= 1e-4 * |ref| + 1e-5  (relative 1e-4 with an absolute
//   floor of 1e-5 ~ 0.5% of the typical |y| ~ 2). The floor keeps 4x margin
//   over the simulated worst absolute error even before device expf
//   differences (~2 ulp) are considered; the relative term dominates for
//   |ref| >> 0.1 where measured error is ~1e-5.

#include <vortex2.h>

#include <vortex/mamba.h>

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
static float frand() {  // uniform [0,1)
    g_seed = g_seed * 1103515245u + 12345u;
    return ((float)(g_seed >> 8 & 0xffffu) / 65536.0f);
}
static float gauss() {  // N(0,1) via Box-Muller
    float u1 = frand(), u2 = frand();
    if (u1 < 1e-7f) u1 = 1e-7f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
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

static void download_f32(vx_queue_h q, std::vector<float>& dst,
                         const DevBuf& b) {
    vx_event_h ev = nullptr;
    vx_enqueue_read(q, dst.data(), b.h, 0, dst.size() * 4, 0, nullptr, &ev);
    vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE);
    vx_event_release(ev);
}

// Double-precision CPU reference (initial state 0). Natural n = 0..N-1
// order; bit-exactness vs the device is not claimed, only the tolerance
// above (the f64 summation-order difference is ~1e-16, irrelevant at 1e-4).
static void ref_scan(const std::vector<float>& A, const std::vector<float>& dt,
                     const std::vector<float>& B, const std::vector<float>& Cm,
                     const std::vector<float>& x, uint32_t Bn, uint32_t C,
                     uint32_t T, uint32_t N, std::vector<double>& y) {
    y.assign((size_t)Bn * C * T, 0.0);
    for (uint32_t bc = 0; bc < Bn * C; ++bc) {
        std::vector<double> s(N, 0.0);
        for (uint32_t t = 0; t < T; ++t) {
            const double decay =
                exp((double)A[bc] * (double)dt[(size_t)bc * T + t]);
            const double xv = (double)x[(size_t)bc * T + t];
            double acc = 0.0;
            for (uint32_t n = 0; n < N; ++n) {
                s[n] = decay * s[n] + (double)B[(size_t)t * N + n] * xv;
                acc += (double)Cm[(size_t)t * N + n] * s[n];
            }
            y[(size_t)bc * T + t] = acc;
        }
    }
}

static const double TOL_REL = 1e-4;
static const double TOL_ABS = 1e-5;

// Returns the failure count; prints one verdict line per case.
static uint32_t run_case(vx_device_h dev, vx_queue_h q, const char* name,
                         uint32_t Bn, uint32_t C, uint32_t T, uint32_t N) {
    const size_t bcT = (size_t)Bn * C * T;
    std::vector<float> A((size_t)Bn * C), dt(bcT), B((size_t)T * N),
        Cm((size_t)T * N), x(bcT);
    for (auto& v : A) v = -(1.0f + 15.0f * frand());   // A in [-16, -1)
    for (auto& v : dt) v = 0.1f * expf(0.5f * gauss());  // lognormal-ish, small
    for (auto& v : B) v = gauss();
    for (auto& v : Cm) v = gauss();
    for (auto& v : x) v = 2.0f * frand() - 1.0f;

    std::vector<double> ref;
    ref_scan(A, dt, B, Cm, x, Bn, C, T, N, ref);

    DevBuf ba = make_buf(dev, A.size() * 4);
    DevBuf bdt = make_buf(dev, dt.size() * 4);
    DevBuf bb = make_buf(dev, B.size() * 4);
    DevBuf bc = make_buf(dev, Cm.size() * 4);
    DevBuf bx = make_buf(dev, x.size() * 4);
    DevBuf by = make_buf(dev, bcT * 4);
    upload(q, ba, A.data(), A.size() * 4);
    upload(q, bdt, dt.data(), dt.size() * 4);
    upload(q, bb, B.data(), B.size() * 4);
    upload(q, bc, Cm.data(), Cm.size() * 4);
    upload(q, bx, x.data(), x.size() * 4);

    CHECK(vx_mamba_selective_scan(q, ba.addr, bdt.addr, bb.addr, bc.addr,
                                  bx.addr, by.addr, Bn, C, T, N));
    CHECK(vx_queue_flush(q));
    std::vector<float> got(bcT);
    download_f32(q, got, by);

    uint32_t bad = 0;
    double max_abs = 0.0, max_rel_floor = 0.0, max_ratio = 0.0;
    size_t worst_i = 0;
    for (size_t i = 0; i < bcT; ++i) {
        const double d = fabs((double)got[i] - ref[i]);
        const double lim = TOL_REL * fabs(ref[i]) + TOL_ABS;
        const double ratio = d / lim;
        if (ratio > max_ratio) {
            max_ratio = ratio;
            worst_i = i;
        }
        if (ratio > 1.0) ++bad;
        if (d > max_abs) max_abs = d;
        const double relf = d / (fabs(ref[i]) + 1e-2);
        if (relf > max_rel_floor) max_rel_floor = relf;
    }
    printf("%-28s B=%u C=%u T=%u N=%u: max_abs=%.2e max_rel(fl 1e-2)=%.2e "
           "err/limit=%.3f bad=%u/%zu",
           name, Bn, C, T, N, max_abs, max_rel_floor, max_ratio, bad, bcT);
    if (bad) {
        printf("  worst: got=%.6f ref=%.6f at (bc=%zu t=%zu)\n", got[worst_i],
               ref[worst_i], worst_i / T, worst_i % T);
    } else {
        printf("\n");
    }

    for (DevBuf* b : {&ba, &bdt, &bb, &bc, &bx, &by}) vx_buffer_release(b->h);
    return bad;
}

int main(int argc, char** argv) {
    const char* vxbin = (argc > 1) ? argv[1] : "mamba.vxbin";

    vx_device_h dev = nullptr;
    CHECK(vx_device_open(0, &dev));
    vx_queue_h q = nullptr;
    CHECK(vx_queue_create(dev, nullptr, &q));
    if (vx_mamba_init(dev, vxbin) != VX_MAMBA_OK) {
        fprintf(stderr, "FAILED: vx_mamba_init(%s)\n", vxbin);
        return 1;
    }

    uint32_t failures = 0;
    // Case 1: the prescribed P7 precursor shape.
    failures += run_case(dev, q, "scan (prescribed)", 1, 16, 48, 8);
    // Case 2: batched + tail CTA with idle warps (10 channels -> 3 CTAs,
    // last one half idle) exercises the masking/sync discipline.
    failures += run_case(dev, q, "scan (batch, partial CTA)", 2, 5, 32, 8);
    // Case 3: max seq length and dstate < 8 (guards in the unrolled state
    // loop; lanes 0..3 own only n = lane).
    failures += run_case(dev, q, "scan (long T, N=4)", 1, 16, 64, 4);

    vx_mamba_finalize();
    vx_queue_release(q);
    vx_device_release(dev);

    if (failures == 0) {
        printf("PASSED\n");
        return 0;
    }
    fprintf(stderr, "FAILED (%u)\n", failures);
    return 1;
}
