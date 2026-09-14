// VX_MXFP8 device kernels (plan P6-02 Q4): microscaling FP8 pack
// (per-group amax -> E8M0 + E4M3 codes), dequant to FP32, and an
// on-the-fly-decoding GEMM in the 16x16-tile / 4-warp-CTA / LMEM-staging
// shape of the existing W4A16/FP8 kernels. Arg-block fields are read
// directly (VOLT -O3 cached-arg miscompile workaround); no float atomics.

#include <hip/hip_runtime.h>
#include <math.h>

#include "quant_fp8.h"
#include "mxfp8_args.h"

// ---------------------------------------------------------------------------
// Pack pass 1: per-(row, group) amax -> E8M0 scale bytes. Grid-stride,
// single-warp CTAs, one thread per (row, group); a group is scanned
// sequentially by its owning thread (same op order as the CPU mirror).
// ---------------------------------------------------------------------------

__global__ void mxfp8_scales_kernel(vx_mxfp8_pack_args_t* arg) {
    const uint32_t groups =
        (arg->cols + arg->group - 1) / arg->group;
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < arg->rows * groups; idx += blockDim.x * gridDim.x) {
        const uint32_t r = idx / groups;
        const uint32_t g = idx % groups;
        const uint32_t c0 = g * arg->group;
        const uint32_t c1 =
            c0 + arg->group < arg->cols ? c0 + arg->group : arg->cols;
        float amax = 0.0f;
        const float* row =
            (const float*)(uintptr_t)arg->src + (size_t)r * arg->cols;
        for (uint32_t c = c0; c < c1; ++c) {
            float a = fabsf(row[c]);
            if (a > amax) amax = a;
        }
        ((uint8_t*)(uintptr_t)arg->scales)[idx] = vx_mxfp8_e8m0(amax);
    }
}

// ---------------------------------------------------------------------------
// Pack pass 2: per-element E4M3 code against the group scale written by
// pass 1 (in-order queue). One thread per element.
// ---------------------------------------------------------------------------

__global__ void mxfp8_pack_kernel(vx_mxfp8_pack_args_t* arg) {
    const uint32_t groups =
        (arg->cols + arg->group - 1) / arg->group;
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < arg->rows * arg->cols; idx += blockDim.x * gridDim.x) {
        const uint32_t c = idx % arg->cols;
        const uint8_t x = ((const uint8_t*)(uintptr_t)arg->scales)
            [(size_t)(idx / arg->cols) * groups + c / arg->group];
        ((uint8_t*)(uintptr_t)arg->codes)[idx] =
            vx_mxfp8_quant(((const float*)(uintptr_t)arg->src)[idx], x);
    }
}

// ---------------------------------------------------------------------------
// Dequant: E4M3 code * 2^(E8M0-127) -> FP32. Exact (power-of-two scale).
// ---------------------------------------------------------------------------

__global__ void mxfp8_dequant_kernel(vx_mxfp8_dequant_args_t* arg) {
    const uint32_t groups =
        (arg->cols + arg->group - 1) / arg->group;
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < arg->rows * arg->cols; idx += blockDim.x * gridDim.x) {
        const uint32_t c = idx % arg->cols;
        const uint8_t x = ((const uint8_t*)(uintptr_t)arg->scales)
            [(size_t)(idx / arg->cols) * groups + c / arg->group];
        ((float*)(uintptr_t)arg->out)[idx] = vx_mxfp8_dequant(
            ((const uint8_t*)(uintptr_t)arg->codes)[idx], x);
    }
}

// ---------------------------------------------------------------------------
// MXFP8 GEMM: 16x16 output tile per 4-warp CTA, K in chunks of 8; codes are
// decoded (code * group scale) at LMEM staging time and staged as FP32, so
// the multiply-accumulate loop is a plain FP32 GEMM. Out-of-range staging
// (partial tiles, K tail) loads zeros.
// ---------------------------------------------------------------------------

#define MX_TILE 16
#define MX_KC 8

__global__ void mxfp8_gemm_kernel(vx_mxfp8_gemm_args_t* arg) {
    float* lA = (float*)__local_mem();           // [16][8] activations
    float* lB = lA + MX_TILE * MX_KC;            // [8][16] weights
    const uint32_t t = threadIdx.x;
    const uint32_t r0 = (t / 4) * 4, c0 = (t % 4) * 4;
    const uint32_t m0 = blockIdx.y * MX_TILE;    // output row base (M)
    const uint32_t n0 = blockIdx.x * MX_TILE;    // output col base (N)
    const uint32_t groups =
        (arg->k + arg->group - 1) / arg->group;

    float acc[4][4] = {{0}};

    for (uint32_t k0 = 0; k0 < arg->k; k0 += MX_KC) {
        for (uint32_t e = t; e < MX_TILE * MX_KC; e += 16) {
            const uint32_t r = e / MX_KC;
            const uint32_t kk = e % MX_KC;
            const uint32_t gm = m0 + r;
            const uint32_t gk = k0 + kk;
            float a = 0.0f, w = 0.0f;
            if (gm < arg->m && gk < arg->k) {
                const uint8_t code = ((const uint8_t*)(uintptr_t)arg->act_codes)
                    [(size_t)gm * arg->k + gk];
                const uint8_t sa = ((const uint8_t*)(uintptr_t)arg->act_scales)
                    [(size_t)gm * groups + gk / arg->group];
                a = vx_mxfp8_dequant(code, sa);
            }
            if (n0 + r < arg->n && gk < arg->k) {
                const uint32_t row = n0 + r;  // weight/output channel
                const uint8_t code = ((const uint8_t*)(uintptr_t)arg->w_codes)
                    [(size_t)row * arg->k + gk];
                const uint8_t sw = ((const uint8_t*)(uintptr_t)arg->w_scales)
                    [(size_t)row * groups + gk / arg->group];
                w = vx_mxfp8_dequant(code, sw);
            }
            lA[r * MX_KC + kk] = a;
            lB[kk * MX_TILE + r] = w;
        }
        __syncthreads();

        for (uint32_t kk = 0; kk < MX_KC; ++kk) {
            for (uint32_t ii = 0; ii < 4; ++ii) {
                float a = lA[(r0 + ii) * MX_KC + kk];
                for (uint32_t jj = 0; jj < 4; ++jj) {
                    acc[ii][jj] += a * lB[kk * MX_TILE + c0 + jj];
                }
            }
        }
        __syncthreads();
    }

    for (uint32_t ii = 0; ii < 4; ++ii) {
        for (uint32_t jj = 0; jj < 4; ++jj) {
            const uint32_t gm = m0 + r0 + ii;
            const uint32_t gn = n0 + c0 + jj;
            if (gm < arg->m && gn < arg->n) {
                ((float*)(uintptr_t)arg->out)[(size_t)gm * arg->n + gn] =
                    acc[ii][jj];
            }
        }
    }
}
