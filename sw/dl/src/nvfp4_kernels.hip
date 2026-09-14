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

// NVFP4 kernels (plan P6-02 Q4 software path): scale generation, packing,
// unpacking, and a W4A16-style GEMM that decodes FP4 weights at staging
// time. Elementwise kernels are grid-stride single-warp CTAs (same shape as
// the quant P6-01 kernels); the GEMM reuses the 16x16-tile / 4-warp-CTA /
// LMEM-staging pattern of quant_gemm_w4a16_kernel. Arg-block fields are
// read directly from the arg pointer at each use (VOLT -O3 workaround for
// cached arg-block loads). No float atomics anywhere.

#include <hip/hip_runtime.h>
#include <vortex/dtypes.h>
#include <math.h>

#include "nvfp4_args.h"

#define NV_TILE 16
#define NV_KC 8

// ---------------------------------------------------------------------------
// Per-(row, group) E4M3 scales: amax over the group -> amax/6 (f32) ->
// / tscale -> E4M3 byte. All-zero group -> scale 1/tscale (avoids q = inf).
// ---------------------------------------------------------------------------

__global__ void nvfp4_scales_kernel(vx_nvfp4_pack_args_t* arg) {
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < arg->n * ((arg->k + arg->group - 1) / arg->group);
         idx += blockDim.x * gridDim.x) {
        const uint32_t groups = (arg->k + arg->group - 1) / arg->group;
        const uint32_t r = idx / groups;
        const uint32_t g = idx % groups;
        const uint32_t c0 = g * arg->group;
        const uint32_t c1 = c0 + arg->group < arg->k ? c0 + arg->group : arg->k;
        float amax = 0.0f;
        for (uint32_t c = c0; c < c1; ++c) {
            float a = fabsf(((const float*)(uintptr_t)arg->weights)
                                [(size_t)r * arg->k + c]);
            if (a > amax) amax = a;
        }
        float s = amax > 0.0f ? amax / 6.0f : 1.0f;
        s = s / arg->tscale;
        ((uint8_t*)(uintptr_t)arg->scales)[idx] = vx_nvfp4_f32_to_e4m3(s);
    }
}

// ---------------------------------------------------------------------------
// Pack f32 -> E2M1 nibble pairs. One thread owns one whole output byte
// (both nibbles), so there is no cross-thread read-modify-write.
// ---------------------------------------------------------------------------

__global__ void nvfp4_pack_kernel(vx_nvfp4_pack_args_t* arg) {
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < arg->n * ((arg->k + 1) / 2); idx += blockDim.x * gridDim.x) {
        const uint32_t r = idx / ((arg->k + 1) / 2);
        const uint32_t c0 = (idx % ((arg->k + 1) / 2)) * 2;
        const uint32_t groups = (arg->k + arg->group - 1) / arg->group;
        uint32_t q0 = 0, q1 = 0;
        for (uint32_t j = 0; j < 2; ++j) {
            const uint32_t c = c0 + j;
            if (c >= arg->k) break;  // odd K: high nibble stays 0
            float q = ((const float*)(uintptr_t)arg->weights)
                          [(size_t)r * arg->k + c];
            q = q / arg->tscale;
            const float E = vx_nvfp4_e4m3_to_f32(
                ((const uint8_t*)(uintptr_t)arg->scales)
                    [(size_t)r * groups + c / arg->group]);
            q = q / E;
            const uint32_t code = vx_f32_to_e2m1(q);
            if (j == 0) q0 = code; else q1 = code;
        }
        ((uint8_t*)(uintptr_t)arg->packed)[idx] =
            (uint8_t)(q0 | (q1 << 4));
    }
}

// ---------------------------------------------------------------------------
// Dequant: out = (e2m1(code) * E4M3_scale) * tscale, two f32 multiplies.
// ---------------------------------------------------------------------------

__global__ void nvfp4_unpack_kernel(vx_nvfp4_unpack_args_t* arg) {
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < arg->n * arg->k; idx += blockDim.x * gridDim.x) {
        const uint32_t r = idx / arg->k;
        const uint32_t c = idx % arg->k;
        const uint8_t byte = ((const uint8_t*)(uintptr_t)arg->packed)
            [(size_t)r * ((arg->k + 1) / 2) + c / 2];
        const uint32_t code = (c % 2 == 0) ? (uint32_t)(byte & 0x0f)
                                           : (uint32_t)(byte >> 4);
        const float v = vx_e2m1_to_f32((uint8_t)code);
        const float E = vx_nvfp4_e4m3_to_f32(
            ((const uint8_t*)(uintptr_t)arg->scales)
                [(size_t)r * ((arg->k + arg->group - 1) / arg->group) +
                 c / arg->group]);
        float o = v * E;
        o = o * arg->tscale;
        ((float*)(uintptr_t)arg->out)[idx] = o;
    }
}

// ---------------------------------------------------------------------------
// W4A16-style GEMM: 16x16 output tile per 4-warp CTA, K in chunks of 8.
// Activations (fp16 storage) and weights (E2M1 x E4M3 scale x tscale) are
// decoded to f32 during LMEM staging; accumulation is FP32 over ascending
// k, matching the tile pattern of quant_gemm_w4a16_kernel exactly.
// ---------------------------------------------------------------------------

__global__ void nvfp4_gemm_kernel(vx_nvfp4_gemm_args_t* arg) {
    float* lA = (float*)__local_mem();      // [16][8] activations (f32)
    float* lW = lA + NV_TILE * NV_KC;       // [8][16] decoded weights (f32)
    const uint32_t t = threadIdx.x;
    const uint32_t r0 = (t / 4) * 4, c0 = (t % 4) * 4;
    const uint32_t m0 = blockIdx.y * NV_TILE;   // output row base (M)
    const uint32_t n0 = blockIdx.x * NV_TILE;   // output col base (N)

    float acc[4][4] = {{0}};

    for (uint32_t k0 = 0; k0 < arg->k; k0 += NV_KC) {
        for (uint32_t e = t; e < NV_TILE * NV_KC; e += 16) {
            const uint32_t r = e / NV_KC;
            const uint32_t kk = e % NV_KC;
            const uint32_t gm = m0 + r;
            const uint32_t gk = k0 + kk;
            float a = 0.0f, w = 0.0f;
            if (gm < arg->m && gk < arg->k) {
                a = vx_fp16_to_f32(((const vx_fp16_t*)(uintptr_t)arg->act)
                                       [(size_t)gm * arg->k + gk]);
            }
            if (n0 + r < arg->n && gk < arg->k) {
                const uint32_t row = n0 + r;  // weight/output channel
                const uint8_t byte =
                    ((const uint8_t*)(uintptr_t)arg->packed)
                        [(size_t)row * ((arg->k + 1) / 2) + gk / 2];
                const uint32_t code = (gk % 2 == 0)
                                          ? (uint32_t)(byte & 0x0f)
                                          : (uint32_t)(byte >> 4);
                w = vx_e2m1_to_f32((uint8_t)code) *
                    vx_nvfp4_e4m3_to_f32(
                        ((const uint8_t*)(uintptr_t)arg->scales)
                            [(size_t)row *
                                 ((arg->k + arg->group - 1) / arg->group) +
                             gk / arg->group]);
                w = w * arg->tscale;
            }
            lA[r * NV_KC + kk] = a;
            lW[kk * NV_TILE + r] = w;
        }
        __syncthreads();

        for (uint32_t kk = 0; kk < NV_KC; ++kk) {
            for (uint32_t ii = 0; ii < 4; ++ii) {
                float a = lA[(r0 + ii) * NV_KC + kk];
                for (uint32_t jj = 0; jj < 4; ++jj) {
                    acc[ii][jj] += a * lW[kk * NV_TILE + c0 + jj];
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
