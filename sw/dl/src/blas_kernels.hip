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

// Vortex DL BLAS kernels (plan P3-01). KMU image: compiled by hipcc-vortex
// with --kernel-lib=vortex2, launched by the host library in sw/dl.
//
// Tiled FPU GEMM: each CTA computes one 16x16 output tile with 16 threads
// (4 warps — a real multi-warp CTA with __syncthreads and LMEM staging,
// which the KMU path supports and the legacy spawn path does not). arg->K is
// processed in chunks of 8; A/B tile elements are staged to LMEM as FP32
// (FP16/BF16 are storage formats converted at load, FP32 accumulate —
// the correctness shape the W4A16 and TCU paths will build on).
//
// Out-of-range loads (arg->M/arg->N/arg->K remainders) stage zeros; stores are guarded.

#include <hip/hip_runtime.h>
#include <vortex/dtypes.h>

#include "blas_args.h"

#define BLAS_TILE 16
#define BLAS_KC   8
#define BLAS_LMEM_FLOATS (2 * BLAS_TILE * BLAS_KC)  // 256 floats = 1 KiB

template <int DT>
static void __attribute__((noinline))
gemm_tile_body(vx_blas_gemm_args_t* arg) {

    // KMU local memory: sized by the launch's lmem_size field (the host
    // library requests 1 KiB for the A+B tiles).
    // TOOLCHAIN WORKAROUND (measured 2026-09-14): do NOT cache arg->M/N/K/
    // alpha/beta into local const variables under -O3 — VOLT miscompiles
    // the cached loads (wrong results across CTAs; exact mechanism TBD,
    // likely load hoisting around the staged argument block). Reading
    // arg-> directly is correct. Revisit when the VOLT fix lands.
    float* lA = (float*)__local_mem();      // [16][8]
    float* lB = lA + BLAS_TILE * BLAS_KC;   // [8][16]

    const uint32_t t = threadIdx.x;          // 0..15
    const uint32_t r0 = (t / 4) * 4;         // this thread's 4x4 rows
    const uint32_t c0 = (t % 4) * 4;         // this thread's 4x4 cols
    const uint32_t m0 = blockIdx.y * BLAS_TILE;
    const uint32_t n0 = blockIdx.x * BLAS_TILE;

    float acc[4][4] = {{0}};

    for (uint32_t k0 = 0; k0 < arg->K; k0 += BLAS_KC) {
        // Stage both tiles: 128 elements each, 16 threads, 8 per thread.
        for (uint32_t e = t; e < BLAS_TILE * BLAS_KC; e += 16) {
            const uint32_t r = e / BLAS_KC;
            const uint32_t k = e % BLAS_KC;
            const uint32_t gr = m0 + r;
            const uint32_t gk = k0 + k;
            float a = 0.0f, b = 0.0f;
            if (gr < arg->M && gk < arg->K) {
                if (DT == 0) {
                    a = ((const float*)(uintptr_t)arg->A)[(size_t)gr * arg->K + gk];
                } else if (DT == 1) {
                    a = vx_fp16_to_f32(((const vx_fp16_t*)(uintptr_t)arg->A)[(size_t)gr * arg->K + gk]);
                } else {
                    a = vx_bf16_to_f32(((const vx_bf16_t*)(uintptr_t)arg->A)[(size_t)gr * arg->K + gk]);
                }
            }
            if (gk < arg->K && n0 + r < arg->N) {
                if (DT == 0) {
                    const size_t bi = arg->transb
                                          ? (size_t)(n0 + r) * arg->K + gk
                                          : (size_t)gk * arg->N + n0 + r;
                    b = ((const float*)(uintptr_t)arg->B)[bi];
                } else if (DT == 1) {
                    const size_t bi = arg->transb
                                          ? (size_t)(n0 + r) * arg->K + gk
                                          : (size_t)gk * arg->N + n0 + r;
                    b = vx_fp16_to_f32(((const vx_fp16_t*)(uintptr_t)arg->B)[bi]);
                } else {
                    const size_t bi = arg->transb
                                          ? (size_t)(n0 + r) * arg->K + gk
                                          : (size_t)gk * arg->N + n0 + r;
                    b = vx_bf16_to_f32(((const vx_bf16_t*)(uintptr_t)arg->B)[bi]);
                }
            }
            lA[r * BLAS_KC + k] = a;
            lB[k * BLAS_TILE + r] = b;  // r doubles as column index for B
        }
        __syncthreads();

        for (uint32_t kk = 0; kk < BLAS_KC; ++kk) {
            for (uint32_t ii = 0; ii < 4; ++ii) {
                float a = lA[(r0 + ii) * BLAS_KC + kk];
                for (uint32_t jj = 0; jj < 4; ++jj) {
                    acc[ii][jj] += a * lB[kk * BLAS_TILE + c0 + jj];
                }
            }
        }
        __syncthreads();
    }

    // Epilogue: C = arg->alpha * acc + arg->beta * C
    for (uint32_t ii = 0; ii < 4; ++ii) {
        for (uint32_t jj = 0; jj < 4; ++jj) {
            const uint32_t gm = m0 + r0 + ii;
            const uint32_t gn = n0 + c0 + jj;
            if (gm < arg->M && gn < arg->N) {
                float* c = (float*)(uintptr_t)arg->C;
                float old = (arg->beta != 0.0f) ? c[(size_t)gm * arg->N + gn] : 0.0f;
                c[(size_t)gm * arg->N + gn] = arg->alpha * acc[ii][jj] + arg->beta * old;
            }
        }
    }
}

__global__ void sgemm_f32_kernel(vx_blas_gemm_args_t* arg) {
    gemm_tile_body<0>(arg);
}

__global__ void hgemm_f16_kernel(vx_blas_gemm_args_t* arg) {
    gemm_tile_body<1>(arg);
}

__global__ void bgemm_bf16_kernel(vx_blas_gemm_args_t* arg) {
    gemm_tile_body<2>(arg);
}
