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

// Vortex DL quant kernels (plan P6-01): W4A16 weight-only packing,
// dequantization, on-the-fly decompressing GEMM, and INT8 W8A8 GEMM.
// Elementwise packing/dequant kernels are grid-stride single-warp CTAs;
// the GEMMs reuse the 16x16-tile / 4-warp-CTA / LMEM-staging shape of
// blas_kernels.hip (decompressed tiles are staged as FP32, FP32
// accumulate). No arg-field caching (VOLT workaround), no float atomics.

#include <hip/hip_runtime.h>
#include <vortex/dtypes.h>
#include <math.h>

#include "quant_args.h"
#include "quant_fp8.h"

#define Q_TILE 16
#define Q_KC 8

// ---------------------------------------------------------------------------
// Pack FP32 -> int4 nibbles + per-group scales
// ---------------------------------------------------------------------------

__global__ void quant_pack4_kernel(vx_quant_pack4_args_t* arg) {
    const uint32_t n = arg->n, k = arg->k, gs = arg->group;
    const uint32_t row_bytes = (k + 1) / 2;
    const uint32_t total = n * row_bytes;
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        const uint32_t r = idx / row_bytes;
        const uint32_t c0 = (idx % row_bytes) * 2;
        const uint32_t groups = (k + gs - 1) / gs;
        const float* wrow = (const float*)(uintptr_t)arg->weights + (size_t)r * k;
        const float* srow = (const float*)(uintptr_t)arg->scales + (size_t)r * groups;
        auto quantize = [&](uint32_t c) -> int {
            float w = wrow[c];
            int q = (int)__builtin_lroundf(w / srow[c / gs]);
            if (q > 7) q = 7;
            if (q < -8) q = -8;
            return q;
        };
        // One thread owns one whole byte: both nibbles are computed here,
        // so there is no cross-thread read-modify-write on the same byte.
        const int q0 = quantize(c0);
        const int q1 = (c0 + 1 < k) ? quantize(c0 + 1) : 0;
        ((uint8_t*)(uintptr_t)arg->packed)[idx] =
            (uint8_t)((q0 & 0x0f) | ((q1 & 0x0f) << 4));
    }
}

// Scales are computed in a separate pass (one warp per (row, group)):
// scale = amax/7 over the group, or 1.0 when the group is all zeros.
__global__ void quant_scales4_kernel(vx_quant_pack4_args_t* arg) {
    const uint32_t k = arg->k, gs = arg->group;
    const uint32_t groups = (k + gs - 1) / gs;
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < arg->n * groups; idx += blockDim.x * gridDim.x) {
        const uint32_t r = idx / groups;
        const uint32_t g = idx % groups;
        const uint32_t c0 = g * gs;
        const uint32_t c1 = c0 + gs < k ? c0 + gs : k;
        float amax = 0.0f;
        const float* row = (const float*)(uintptr_t)arg->weights + (size_t)r * k;
        for (uint32_t c = c0; c < c1; ++c) {
            float a = fabsf(row[c]);
            if (a > amax) amax = a;
        }
        ((float*)(uintptr_t)arg->scales)[idx] = amax > 0.0f ? amax / 7.0f : 1.0f;
    }
}

// ---------------------------------------------------------------------------
// Dequant int4 + scales -> FP32
// ---------------------------------------------------------------------------

__global__ void quant_unpack4_kernel(vx_quant_unpack4_args_t* arg) {
    const uint32_t k = arg->k, gs = arg->group;
    const uint32_t groups = (k + gs - 1) / gs;
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < arg->n * k; idx += blockDim.x * gridDim.x) {
        const uint32_t r = idx / k;
        const uint32_t c = idx % k;
        const uint8_t byte =
            ((const uint8_t*)(uintptr_t)arg->packed)[(size_t)r * ((k + 1) / 2) + c / 2];
        const int q = (c % 2 == 0) ? (int)(byte & 0x0f) : (int)(byte >> 4);
        const int qs = (q >= 8) ? q - 16 : q;
        const float scale =
            ((const float*)(uintptr_t)arg->scales)[(size_t)r * groups + c / gs];
        ((float*)(uintptr_t)arg->out)[idx] = (float)qs * scale;
    }
}

// ---------------------------------------------------------------------------
// W4A16 GEMM: 16x16 output tile per 4-warp CTA, K in chunks of 8; the
// weight tile is decompressed into the LMEM staging buffer (as FP32)
// before the dot products, so the inner loop matches the FP32 GEMM.
// ---------------------------------------------------------------------------

__global__ void quant_gemm_w4a16_kernel(vx_quant_gemm_w4a16_args_t* arg) {
    float* lA = (float*)__local_mem();          // [16][8] activations
    float* lW = lA + Q_TILE * Q_KC;             // [8][16] decompressed weights
    const uint32_t t = threadIdx.x;
    const uint32_t r0 = (t / 4) * 4, c0 = (t % 4) * 4;
    const uint32_t m0 = blockIdx.y * Q_TILE;    // output row base (M)
    const uint32_t n0 = blockIdx.x * Q_TILE;    // output col base (N)
    const uint32_t groups = (arg->k + arg->group - 1) / arg->group;

    float acc[4][4] = {{0}};

    for (uint32_t k0 = 0; k0 < arg->k; k0 += Q_KC) {
        for (uint32_t e = t; e < Q_TILE * Q_KC; e += 16) {
            const uint32_t r = e / Q_KC;
            const uint32_t kk = e % Q_KC;
            const uint32_t gm = m0 + r;
            const uint32_t gk = k0 + kk;
            float a = 0.0f, w = 0.0f;
            if (gm < arg->m && gk < arg->k) {
                a = vx_fp16_to_f32(((const vx_fp16_t*)(uintptr_t)arg->act)
                                   [(size_t)gm * arg->k + gk]);
            }
            if (n0 + r < arg->n && gk < arg->k) {
                const uint32_t row = n0 + r;  // weight/output channel
                const uint8_t byte = ((const uint8_t*)(uintptr_t)arg->packed)
                    [(size_t)row * ((arg->k + 1) / 2) + gk / 2];
                const int q = (gk % 2 == 0) ? (int)(byte & 0x0f)
                                            : (int)(byte >> 4);
                const int qs = (q >= 8) ? q - 16 : q;
                const float scale = ((const float*)(uintptr_t)arg->scales)
                    [(size_t)row * groups + gk / arg->group];
                w = (float)qs * scale;
            }
            lA[r * Q_KC + kk] = a;
            lW[kk * Q_TILE + r] = w;
        }
        __syncthreads();

        for (uint32_t kk = 0; kk < Q_KC; ++kk) {
            for (uint32_t ii = 0; ii < 4; ++ii) {
                float a = lA[(r0 + ii) * Q_KC + kk];
                for (uint32_t jj = 0; jj < 4; ++jj) {
                    acc[ii][jj] += a * lW[kk * Q_TILE + c0 + jj];
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

// ---------------------------------------------------------------------------
// INT8 W8A8 GEMM: act int8 (per-tensor scale) x weights int8 (per-row
// scale); int32 accumulate per chunk, scaled once in the epilogue.
// ---------------------------------------------------------------------------

__global__ void quant_gemm_w8a8_kernel(vx_quant_gemm_w8a8_args_t* arg) {
    float* lA = (float*)__local_mem();   // staged as float for reuse
    float* lW = lA + Q_TILE * Q_KC;
    const uint32_t t = threadIdx.x;
    const uint32_t r0 = (t / 4) * 4, c0 = (t % 4) * 4;
    const uint32_t m0 = blockIdx.y * Q_TILE;
    const uint32_t n0 = blockIdx.x * Q_TILE;

    int32_t acc[4][4] = {{0}};

    for (uint32_t k0 = 0; k0 < arg->k; k0 += Q_KC) {
        for (uint32_t e = t; e < Q_TILE * Q_KC; e += 16) {
            const uint32_t r = e / Q_KC;
            const uint32_t kk = e % Q_KC;
            const uint32_t gm = m0 + r;
            const uint32_t gk = k0 + kk;
            int a = 0, w = 0;
            if (gm < arg->m && gk < arg->k) {
                a = (int)((const int8_t*)(uintptr_t)arg->act)[(size_t)gm * arg->k + gk];
            }
            if (n0 + r < arg->n && gk < arg->k) {
                w = (int)((const int8_t*)(uintptr_t)arg->weights)
                    [(size_t)(n0 + r) * arg->k + gk];
            }
            lA[r * Q_KC + kk] = (float)a;
            lW[kk * Q_TILE + r] = (float)w;
        }
        __syncthreads();

        for (uint32_t kk = 0; kk < Q_KC; ++kk) {
            for (uint32_t ii = 0; ii < 4; ++ii) {
                float a = lA[(r0 + ii) * Q_KC + kk];
                for (uint32_t jj = 0; jj < 4; ++jj) {
                    acc[ii][jj] += (int32_t)(a * lW[kk * Q_TILE + c0 + jj]);
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
                const float wscale =
                    ((const float*)(uintptr_t)arg->w_scale)[gn];
                ((float*)(uintptr_t)arg->out)[(size_t)gm * arg->n + gn] =
                    (float)acc[ii][jj] * arg->act_scale * wscale;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FP8 W8A8 GEMM: same 16x16-tile shape, codes converted at staging time.
// ---------------------------------------------------------------------------

__global__ void quant_gemm_fp8_kernel(vx_quant_gemm_fp8_args_t* arg) {
    float* lA = (float*)__local_mem();
    float* lB = lA + Q_TILE * Q_KC;
    const uint32_t t = threadIdx.x;
    const uint32_t r0 = (t / 4) * 4, c0 = (t % 4) * 4;
    const uint32_t m0 = blockIdx.y * Q_TILE;
    const uint32_t n0 = blockIdx.x * Q_TILE;
    float acc[4][4] = {{0}};
    for (uint32_t k0 = 0; k0 < arg->k; k0 += Q_KC) {
        for (uint32_t e = t; e < Q_TILE * Q_KC; e += 16) {
            const uint32_t r = e / Q_KC, kk = e % Q_KC;
            const uint32_t gm = m0 + r, gk = k0 + kk;
            float a = 0.0f, w = 0.0f;
            if (gm < arg->m && gk < arg->k) {
                const uint8_t c = ((const uint8_t*)(uintptr_t)arg->act)
                    [(size_t)gm * arg->k + gk];
                a = arg->mode == 0 ? vx_e4m3_to_f32(c) : vx_e5m2_to_f32(c);
            }
            if (n0 + r < arg->n && gk < arg->k) {
                const uint8_t c = ((const uint8_t*)(uintptr_t)arg->weights)
                    [(size_t)(n0 + r) * arg->k + gk];
                w = arg->mode == 0 ? vx_e4m3_to_f32(c) : vx_e5m2_to_f32(c);
            }
            lA[r * Q_KC + kk] = a;
            lB[kk * Q_TILE + r] = w;
        }
        __syncthreads();
        for (uint32_t kk = 0; kk < Q_KC; ++kk)
            for (uint32_t ii = 0; ii < 4; ++ii) {
                float a = lA[(r0 + ii) * Q_KC + kk];
                for (uint32_t jj = 0; jj < 4; ++jj)
                    acc[ii][jj] += a * lB[kk * Q_TILE + c0 + jj];
            }
        __syncthreads();
    }
    for (uint32_t ii = 0; ii < 4; ++ii)
        for (uint32_t jj = 0; jj < 4; ++jj) {
            const uint32_t gm = m0 + r0 + ii, gn = n0 + c0 + jj;
            if (gm < arg->m && gn < arg->n)
                ((float*)(uintptr_t)arg->out)[(size_t)gm * arg->n + gn] =
                    acc[ii][jj];
        }
}
