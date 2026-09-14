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

// ---- conv2d (NCHW, dilation 1, groups 1) ----------------------------------
struct conv_args_t {
    uint64_t in, weight, bias, out;
    uint32_t n, ci, hi, wi, co, ho, wo;
    uint32_t kh, kw, ph, pw, sh, sw, has_bias;
};

__global__ void tv_conv2d_kernel(conv_args_t* arg) {
    const uint32_t n = blockIdx.z;
    const uint32_t co = blockIdx.y;
    const uint32_t oy = blockIdx.x;

    const float* w = (const float*)(uintptr_t)arg->weight +
                     ((size_t)co * arg->ci) * (arg->kh * arg->kw);
    float* stage = (float*)__local_mem();
    const uint32_t wtotal = arg->ci * arg->kh * arg->kw;
    for (uint32_t e = threadIdx.x; e < wtotal; e += 16) {
        stage[e] = w[e];
    }
    __syncthreads();

    const float* in_base = (const float*)(uintptr_t)arg->in +
                           (size_t)n * arg->ci * arg->hi * arg->wi;
    float* out_row = (float*)(uintptr_t)arg->out +
                     (((size_t)n * arg->co + co) * arg->ho + oy) * arg->wo;
    const float bias = arg->has_bias
                           ? ((const float*)(uintptr_t)arg->bias)[co]
                           : 0.0f;

    for (uint32_t ox = threadIdx.x; ox < arg->wo; ox += 16) {
        float acc = bias;
        uint32_t we = 0;
        for (uint32_t ci = 0; ci < arg->ci; ++ci) {
            const float* in_ch = in_base + (size_t)ci * arg->hi * arg->wi;
            for (uint32_t kh = 0; kh < arg->kh; ++kh) {
                const int32_t iy = (int32_t)(oy * arg->sh + kh) - (int32_t)arg->ph;
                if (iy < 0 || iy >= (int32_t)arg->hi) {
                    we += arg->kw;
                    continue;
                }
                for (uint32_t kw = 0; kw < arg->kw; ++kw) {
                    const int32_t ix = (int32_t)(ox * arg->sw + kw) - (int32_t)arg->pw;
                    if (ix >= 0 && ix < (int32_t)arg->wi) {
                        acc += in_ch[(size_t)iy * arg->wi + ix] * stage[we];
                    }
                    ++we;
                }
            }
        }
        out_row[ox] = acc;
    }
}

// ---- pooling (op 0 = max, 1 = avg) ----------------------------------------
struct pool_args_t {
    uint64_t in, out;
    uint32_t n, c, hi, wi, ho, wo;
    uint32_t kh, kw, ph, pw, sh, sw, op;
};

__global__ void tv_pool2d_kernel(pool_args_t* arg) {
    const uint32_t n = blockIdx.z;
    const uint32_t c = blockIdx.y;
    const uint32_t oy = blockIdx.x;
    const float* in_ch = (const float*)(uintptr_t)arg->in +
                         (((size_t)n * arg->c) + c) * arg->hi * arg->wi;
    float* out_row = (float*)(uintptr_t)arg->out +
                     (((size_t)n * arg->c + c) * arg->ho + oy) * arg->wo;
    for (uint32_t ox = threadIdx.x; ox < arg->wo; ox += 16) {
        float acc = (arg->op == 0) ? -3.4e38f : 0.0f;
        uint32_t count = 0;
        for (uint32_t kh = 0; kh < arg->kh; ++kh) {
            const int32_t iy = (int32_t)(oy * arg->sh + kh) - (int32_t)arg->ph;
            if (iy < 0 || iy >= (int32_t)arg->hi) continue;
            for (uint32_t kw = 0; kw < arg->kw; ++kw) {
                const int32_t ix = (int32_t)(ox * arg->sw + kw) - (int32_t)arg->pw;
                if (ix < 0 || ix >= (int32_t)arg->wi) continue;
                float v = in_ch[(size_t)iy * arg->wi + ix];
                if (arg->op == 0) {
                    if (v > acc) acc = v;
                } else {
                    acc += v;
                }
                ++count;
            }
        }
        out_row[ox] = (arg->op == 0) ? acc : (count ? acc / (float)count : 0.0f);
    }
}

// ---- batch-norm affine -----------------------------------------------------
struct bn_args_t {
    uint64_t in, mean, rstd, weight, bias, out;
    uint32_t total, c;
};

__global__ void tv_bn_affine_kernel(bn_args_t* arg) {
    const uint32_t plane = arg->total / arg->c;
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < arg->total; idx += blockDim.x * gridDim.x) {
        const uint32_t ch = (idx / plane) % arg->c;
        const float x = ((const float*)(uintptr_t)arg->in)[idx];
        ((float*)(uintptr_t)arg->out)[idx] =
            (x - ((const float*)(uintptr_t)arg->mean)[ch]) *
                ((const float*)(uintptr_t)arg->rstd)[ch] *
                ((const float*)(uintptr_t)arg->weight)[ch] +
            ((const float*)(uintptr_t)arg->bias)[ch];
    }
}

// ---- mm: 16x16 tile per 4-warp CTA, K in chunks of 8, LMEM staging --------
struct mm_args_t {
    uint64_t a, b, out;
    uint32_t m, n, k;
    uint32_t transb;  // b layout: 0 = [k][n] row-major, 1 = [n][k] (i.e. use b^T)
};

// out[i, j] += bias[j] (row broadcast), for linear/addmm epilogues.
struct bias_args_t {
    uint64_t out, bias;
    uint32_t m, n;
};

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

__global__ void tv_bias_add_kernel(bias_args_t* arg) {
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < arg->m * arg->n; idx += blockDim.x * gridDim.x) {
        const uint32_t j = idx % arg->n;
        ((float*)(uintptr_t)arg->out)[idx] +=
            ((const float*)(uintptr_t)arg->bias)[j];
    }
}
