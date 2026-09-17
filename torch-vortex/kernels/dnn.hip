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

// torch-vortex dnn kernels (plan P5-02): the ResNet op set as KMU images.
// Same shapes as sw/dl's dnn/blas kernels (one 4-warp CTA per row/tile,
// LMEM staging), with rv64-width pointer arg blocks (torch-vortex builds
// against an rv64 tree; see the package README).

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

// Argument blocks are defined once, for both the device and host compilers, in
// torch_kernel_args.h. Do not declare them locally: a second definition is
// exactly how the host/device layout drift started.
#include "torch_kernel_args.h"

// conv2d lives in the DL library now (vx_dnn_conv2d). Keeping a second
// copy here is what W3.1 exists to remove.

// pooling lives in the DL library now (vx_dnn_pool2d).

// ---- batch-norm affine -----------------------------------------------------
__global__ void tv_bn_affine_kernel(bn_args_t* arg) {
    // arg->hw is H*W, the per-channel span in NCHW. Deriving it as total/c
    // gives N*H*W instead, which selects the right channel only when N == 1 --
    // the bug the batch>1 tests exist to catch.
    const uint32_t plane = arg->hw;
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < arg->total; idx += blockDim.x * gridDim.x) {
        const uint32_t ch = (idx / plane) % arg->c;
        const float x = ((const float*)(uintptr_t)arg->in)[idx];
        const float mean = ((const float*)(uintptr_t)arg->mean)[ch];
        const float var = ((const float*)(uintptr_t)arg->var)[ch];
        const float w = arg->has_affine
                            ? ((const float*)(uintptr_t)arg->weight)[ch]
                            : 1.0f;
        const float b = arg->has_affine
                            ? ((const float*)(uintptr_t)arg->bias)[ch]
                            : 0.0f;
        // deliberate: 1/sqrt, not rsqrt, so the result is bit-comparable with
        // the CPU reference instead of leaving an avoidable ULP to absorb
        const float rstd = 1.0f / sqrtf(var + arg->eps);
        ((float*)(uintptr_t)arg->out)[idx] = (x - mean) * rstd * w + b;
    }
}

// ---- mm: 16x16 tile per 4-warp CTA, K in chunks of 8, LMEM staging --------
__global__ void tv_mm_kernel(mm_args_t* arg) {
    float* lA = (float*)__local_mem();
    float* lB = lA + 16 * 8;
    const uint32_t t = threadIdx.x;
    const uint32_t r0 = (t / 4) * 4, c0 = (t % 4) * 4;
    const uint32_t m0 = blockIdx.y * 16;
    const uint32_t n0 = blockIdx.x * 16;
    float acc[4][4] = {{0}};
    for (uint32_t k0 = 0; k0 < arg->k; k0 += 8) {
        for (uint32_t e = t; e < 16 * 8; e += 16) {
            const uint32_t r = e / 8, kk = e % 8;
            float a = 0.0f, b = 0.0f;
            if (m0 + r < arg->m && k0 + kk < arg->k)
                a = ((const float*)(uintptr_t)arg->a)[(size_t)(m0 + r) * arg->k + k0 + kk];
            if (k0 + kk < arg->k && n0 + r < arg->n) {
                if (arg->transb) {
                    // b is [n][k]: element (k0+kk, n0+r) of b^T
                    b = ((const float*)(uintptr_t)arg->b)[(size_t)(n0 + r) * arg->k + k0 + kk];
                } else {
                    b = ((const float*)(uintptr_t)arg->b)[(size_t)(k0 + kk) * arg->n + n0 + r];
                }
            }
            lA[r * 8 + kk] = a;
            lB[kk * 16 + r] = b;
        }
        __syncthreads();
        for (uint32_t kk = 0; kk < 8; ++kk)
            for (uint32_t ii = 0; ii < 4; ++ii) {
                float a = lA[(r0 + ii) * 8 + kk];
                for (uint32_t jj = 0; jj < 4; ++jj)
                    acc[ii][jj] += a * lB[kk * 16 + c0 + jj];
            }
        __syncthreads();
    }
    for (uint32_t ii = 0; ii < 4; ++ii)
        for (uint32_t jj = 0; jj < 4; ++jj) {
            const uint32_t gm = m0 + r0 + ii, gn = n0 + c0 + jj;
            if (gm < arg->m && gn < arg->n)
                ((float*)(uintptr_t)arg->out)[(size_t)gm * arg->n + gn] = acc[ii][jj];
        }
}

// In-place epilogue for torch.addmm: out = alpha*out + beta*self.
//
// Kept out of tv_mm_kernel on purpose. Adding this branch to that kernel's
// epilogue makes VOLT generate code that accumulates incorrectly for any
// shape whose n is not a multiple of 16 -- the accumulation picks up values
// that were never loaded. A minimal repro is the branch itself: replacing the
// body with `v += arg->beta;` still reproduces it, while the same code here in
// a plain elementwise loop is correct. See W5.4 of
// docs/mydocs/pytorch_plan.md; tests/test_matmul.py pins the separation.
__global__ void tv_mm_epilogue_kernel(mm_epilogue_args_t* arg) {
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < arg->m * arg->n; idx += blockDim.x * gridDim.x) {
        float v = arg->alpha * ((float*)(uintptr_t)arg->out)[idx];
        // beta == 0 must not read self at all: torch.addmm ignores it then,
        // and 0 * Inf would be NaN.
        if (arg->self_kind != 0 && arg->beta != 0.0f) {
            const float* self = (const float*)(uintptr_t)arg->self;
            const uint32_t i = idx / arg->n, j = idx % arg->n;
            v += arg->beta *
                 ((arg->self_kind == 1) ? self[j] : self[(size_t)i * arg->n + j]);
        }
        ((float*)(uintptr_t)arg->out)[idx] = v;
    }
}

__global__ void tv_bias_add_kernel(bias_args_t* arg) {
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < arg->m * arg->n; idx += blockDim.x * gridDim.x) {
        const uint32_t j = idx % arg->n;
        ((float*)(uintptr_t)arg->out)[idx] +=
            ((const float*)(uintptr_t)arg->bias)[j];
    }
}
