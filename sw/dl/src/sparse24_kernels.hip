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

// Vortex DL 2:4 structured-sparsity kernels (plan P6-02 Q5): magnitude
// top-2-of-4 pruning with a per-row mask, and a sparse GEMM whose weight
// staging consults the mask on device (pruned elements are never loaded).
// Prune is a grid-stride elementwise kernel (one thread per row/group);
// the GEMM reuses the 16x16-tile / 4-warp-CTA / LMEM-staging shape of
// quant_kernels.hip with FP32 accumulate. Arg-block fields are read
// directly at each use (VOLT -O3 cached-arg miscompile workaround); no
// float atomics anywhere.

#include <hip/hip_runtime.h>
#include <math.h>

#include "sparse24_args.h"

#define S24_TILE 16
#define S24_KC 8

// ---------------------------------------------------------------------------
// Prune: dense N x K f32 -> values (zeros in pruned slots) + row mask.
// Top-2 magnitudes per consecutive group of 4 along K; ties break to the
// lower index (ascending scan with strict '>' keeps the first equal).
// ---------------------------------------------------------------------------

__global__ void sparse24_prune_kernel(vx_sparse24_prune_args_t* arg) {
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < arg->n * ((arg->k + 3) / 4);
         idx += blockDim.x * gridDim.x) {
        const uint32_t r = idx / ((arg->k + 3) / 4);
        const uint32_t g = idx % ((arg->k + 3) / 4);
        const uint32_t c0 = g * 4;
        const uint32_t gsz = (arg->k - c0 < 4) ? (arg->k - c0) : 4;
        const float* wrow =
            (const float*)(uintptr_t)arg->weights + (size_t)r * arg->k;
        float* vrow = (float*)(uintptr_t)arg->values + (size_t)r * arg->k;

        // Largest magnitude, ties -> lowest index.
        uint32_t i1 = 0;
        for (uint32_t i = 1; i < gsz; ++i) {
            if (fabsf(wrow[c0 + i]) > fabsf(wrow[c0 + i1])) i1 = i;
        }
        // Second largest among the rest, same tie-break.
        uint32_t i2 = 0xffffffffu;
        for (uint32_t i = 0; i < gsz; ++i) {
            if (i == i1) continue;
            if (i2 == 0xffffffffu ||
                fabsf(wrow[c0 + i]) > fabsf(wrow[c0 + i2])) {
                i2 = i;
            }
        }

        uint8_t mb = 0;
        for (uint32_t i = 0; i < gsz; ++i) {
            const uint32_t keep = (i == i1 || i == i2) ? 1u : 0u;
            vrow[c0 + i] = keep ? wrow[c0 + i] : 0.0f;
            mb = (uint8_t)(mb | (uint8_t)(keep << (2 * i)));
        }
        ((uint8_t*)(uintptr_t)arg->mask)[idx] = mb;
    }
}

// ---------------------------------------------------------------------------
// Sparse GEMM: C[M][N] = act[M][K] x values[N][K]^T, FP32 accumulate.
// 16x16 output tile per 4-warp CTA, K in chunks of 8; act and weight tiles
// are staged to LMEM as FP32. The weight staging is metadata-driven: the
// 2-bit kept field is read from the mask BEFORE the values array is
// touched, so pruned elements are skipped, not multiplied by a zero that
// happens to sit in memory.
// ---------------------------------------------------------------------------

__global__ void sparse24_gemm_kernel(vx_sparse24_gemm_args_t* arg) {
    float* lA = (float*)__local_mem();           // [16][8] activations
    float* lW = lA + S24_TILE * S24_KC;          // [8][16] sparse weights
    const uint32_t t = threadIdx.x;
    const uint32_t r0 = (t / 4) * 4, c0 = (t % 4) * 4;
    const uint32_t m0 = blockIdx.y * S24_TILE;   // output row base (M)
    const uint32_t n0 = blockIdx.x * S24_TILE;   // output col base (N)

    float acc[4][4] = {{0}};

    for (uint32_t k0 = 0; k0 < arg->k; k0 += S24_KC) {
        for (uint32_t e = t; e < S24_TILE * S24_KC; e += 16) {
            const uint32_t r = e / S24_KC;
            const uint32_t kk = e % S24_KC;
            const uint32_t gm = m0 + r;
            const uint32_t gk = k0 + kk;
            float a = 0.0f, w = 0.0f;
            if (gm < arg->m && gk < arg->k) {
                a = ((const float*)(uintptr_t)arg->act)[(size_t)gm * arg->k + gk];
            }
            if (n0 + r < arg->n && gk < arg->k) {
                const uint32_t row = n0 + r;  // weight/output channel
                const uint8_t mb =
                    ((const uint8_t*)(uintptr_t)arg->mask)
                        [(size_t)row * ((arg->k + 3) / 4) + (gk >> 2)];
                // 2-bit little-endian field, value 1 = kept.
                if (((mb >> (2 * (gk & 3))) & 3u) == 1u) {
                    w = ((const float*)(uintptr_t)arg->values)
                        [(size_t)row * arg->k + gk];
                }
            }
            lA[r * S24_KC + kk] = a;
            lW[kk * S24_TILE + r] = w;
        }
        __syncthreads();

        for (uint32_t kk = 0; kk < S24_KC; ++kk) {
            for (uint32_t ii = 0; ii < 4; ++ii) {
                float a = lA[(r0 + ii) * S24_KC + kk];
                for (uint32_t jj = 0; jj < 4; ++jj) {
                    acc[ii][jj] += a * lW[kk * S24_TILE + c0 + jj];
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
