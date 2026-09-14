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

// Vortex DL prim/norm kernels (plan P3-02). KMU image compiled by
// hipcc-vortex with --kernel-lib=vortex2.
//
// Shapes follow the P2-03 findings: elementwise work is grid-stride over
// single-warp CTAs (no cross-warp state); reductions/softmax/norm use one
// 16-thread multi-warp CTA per row with LMEM staging + __syncthreads,
// which the KMU path supports. No float atomics anywhere (the ZACAS
// livelock finding). Arg fields are read directly, never cached into
// locals (the VOLT -O3 miscompile workaround).

#include <hip/hip_runtime.h>
#include <math.h>

#include "prim_args.h"

#define PRIM_BLOCK 16
#define PRIM_LMEM_FLOATS 64  // staging for per-warp partials

// ---------------------------------------------------------------------------
// Elementwise unary
// ---------------------------------------------------------------------------

__global__ void prim_unary_kernel(vx_prim_unary_args_t* arg) {
    const float* in = (const float*)(uintptr_t)arg->in;
    float* out = (float*)(uintptr_t)arg->out;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < arg->n;
         i += blockDim.x * gridDim.x) {
        float x = in[i];
        float y;
        switch (arg->op) {
        case VX_PRIM_RELU:
            y = x > 0.0f ? x : 0.0f;
            break;
        case VX_PRIM_GELU:
            // tanh approximation (matches the reference harness)
            y = 0.5f * x * (1.0f + tanhf(
                0.7978845608028654f * (x + 0.044715f * x * x * x)));
            break;
        case VX_PRIM_SILU:
            y = x / (1.0f + expf(-x));
            break;
        default:  // VX_PRIM_NEG
            y = -x;
            break;
        }
        out[i] = y;
    }
}

// ---------------------------------------------------------------------------
// Block reduce: one 16-thread CTA over the whole vector (gridDim.x = 1).
// Multi-warp CTA with LMEM staging; out[0] = result (f32 or u32 argmax).
// ---------------------------------------------------------------------------

static __device__ float warp_sum(float x) {
    for (uint32_t off = warpSize / 2; off > 0; off >>= 1) {
        x += __shfl_down(x, (int)off, warpSize);
    }
    return x;
}

static __device__ float warp_max(float x) {
    for (uint32_t off = warpSize / 2; off > 0; off >>= 1) {
        x = fmaxf(x, __shfl_down(x, (int)off, warpSize));
    }
    return x;
}

__global__ void prim_reduce_kernel(vx_prim_reduce_args_t* arg) {
    float* tbuf = (float*)__local_mem();
    const uint32_t lane = threadIdx.x % warpSize;
    const uint32_t wid = threadIdx.x / warpSize;

    float acc = (arg->op == 1) ? -INFINITY : 0.0f;
    float seen = -INFINITY;  // for argmax: best value seen by this lane
    uint32_t seen_idx = 0;

    for (uint32_t i = threadIdx.x; i < arg->n; i += PRIM_BLOCK) {
        float x = ((const float*)(uintptr_t)arg->in)[i];
        if (arg->op == 0) {
            acc += x;
        } else {
            if (x > seen) {
                seen = x;
                seen_idx = i;
            }
            acc = fmaxf(acc, x);
        }
    }

    // argmax reduces (value, index) pairs: warp shuffle the max value,
    // then pick the smallest index among lanes holding it.
    if (arg->op == 2) {
        for (uint32_t off = warpSize / 2; off > 0; off >>= 1) {
            float ov = __shfl_down(seen, (int)off, warpSize);
            uint32_t oi = __shfl_down(seen_idx, (int)off, warpSize);
            if (ov > seen || (ov == seen && oi < seen_idx)) {
                seen = ov;
                seen_idx = oi;
            }
        }
        uint32_t* ubuf = (uint32_t*)tbuf;
        if (lane == 0) {
            tbuf[wid] = seen;
            ubuf[PRIM_LMEM_FLOATS / 2 + wid] = seen_idx;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float bv = tbuf[0];
            uint32_t bi = ubuf[PRIM_LMEM_FLOATS / 2];
            for (uint32_t w = 1; w < PRIM_BLOCK / warpSize; ++w) {
                float v = tbuf[w];
                uint32_t i2 = ubuf[PRIM_LMEM_FLOATS / 2 + w];
                if (v > bv || (v == bv && i2 < bi)) {
                    bv = v;
                    bi = i2;
                }
            }
            ((uint32_t*)(uintptr_t)arg->out)[0] = bi;
        }
        return;
    }

    acc = (arg->op == 1) ? warp_max(acc) : warp_sum(acc);
    if (lane == 0) {
        tbuf[wid] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = tbuf[0];
        for (uint32_t w = 1; w < PRIM_BLOCK / warpSize; ++w) {
            v = (arg->op == 1) ? fmaxf(v, tbuf[w]) : v + tbuf[w];
        }
        ((float*)(uintptr_t)arg->out)[0] = v;
    }
}

// ---------------------------------------------------------------------------
// Softmax: one 16-thread CTA per row.
// ---------------------------------------------------------------------------

__global__ void prim_softmax_kernel(vx_prim_rowargs_t* arg) {
    float* tbuf = (float*)__local_mem();
    const uint32_t row = blockIdx.x;
    const float* in = (const float*)(uintptr_t)arg->in + (size_t)row * arg->cols;
    float* out = (float*)(uintptr_t)arg->out + (size_t)row * arg->cols;
    const uint32_t cols = arg->cols;

    // pass 1: row max
    float m = -INFINITY;
    for (uint32_t j = threadIdx.x; j < cols; j += PRIM_BLOCK) {
        m = fmaxf(m, in[j]);
    }
    m = warp_max(m);
    if (threadIdx.x % warpSize == 0) {
        tbuf[threadIdx.x / warpSize] = m;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = tbuf[0];
        for (uint32_t w = 1; w < PRIM_BLOCK / warpSize; ++w) v = fmaxf(v, tbuf[w]);
        tbuf[8] = v;  // broadcast slot
    }
    __syncthreads();
    m = tbuf[8];

    // pass 2: exp(x - max) and sum
    float s = 0.0f;
    for (uint32_t j = threadIdx.x; j < cols; j += PRIM_BLOCK) {
        float e = expf(in[j] - m);
        out[j] = e;
        s += e;
    }
    s = warp_sum(s);
    if (threadIdx.x % warpSize == 0) {
        tbuf[threadIdx.x / warpSize] = s;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = tbuf[0];
        for (uint32_t w = 1; w < PRIM_BLOCK / warpSize; ++w) v += tbuf[w];
        tbuf[8] = v;
    }
    __syncthreads();
    s = tbuf[8];

    // pass 3: scale
    for (uint32_t j = threadIdx.x; j < cols; j += PRIM_BLOCK) {
        out[j] /= s;
    }
}

// ---------------------------------------------------------------------------
// LayerNorm / RMSNorm: one 16-thread CTA per row.
// ---------------------------------------------------------------------------

__global__ void prim_layernorm_kernel(vx_prim_norm_args_t* arg) {
    float* tbuf = (float*)__local_mem();
    const uint32_t row = blockIdx.x;
    const float* in = (const float*)(uintptr_t)arg->in + (size_t)row * arg->cols;
    const float* gamma = (const float*)(uintptr_t)arg->gamma;
    const float* beta = (const float*)(uintptr_t)arg->beta;
    float* out = (float*)(uintptr_t)arg->out + (size_t)row * arg->cols;
    const uint32_t cols = arg->cols;

    float s1 = 0.0f, s2 = 0.0f;
    for (uint32_t j = threadIdx.x; j < cols; j += PRIM_BLOCK) {
        s1 += in[j];
        s2 += in[j] * in[j];
    }
    s1 = warp_sum(s1);
    s2 = warp_sum(s2);
    if (threadIdx.x % warpSize == 0) {
        tbuf[threadIdx.x / warpSize] = s1;
        tbuf[8 + threadIdx.x / warpSize] = s2;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float v1 = 0.0f, v2 = 0.0f;
        for (uint32_t w = 0; w < PRIM_BLOCK / warpSize; ++w) {
            v1 += tbuf[w];
            v2 += tbuf[8 + w];
        }
        tbuf[16] = v1 / (float)cols;              // mean
        tbuf[17] = v2 / (float)cols;              // mean of squares
    }
    __syncthreads();
    const float mean = tbuf[16];
    const float m2 = tbuf[17];
    const float var = fmaxf(m2 - mean * mean, 0.0f);
    const float rstd = rsqrtf(var + arg->eps);

    for (uint32_t j = threadIdx.x; j < cols; j += PRIM_BLOCK) {
        out[j] = (in[j] - mean) * rstd * gamma[j] + beta[j];
    }
}

__global__ void prim_rmsnorm_kernel(vx_prim_rmsnorm_args_t* arg) {
    float* tbuf = (float*)__local_mem();
    const uint32_t row = blockIdx.x;
    const float* in = (const float*)(uintptr_t)arg->in + (size_t)row * arg->cols;
    const float* gamma = (const float*)(uintptr_t)arg->gamma;
    float* out = (float*)(uintptr_t)arg->out + (size_t)row * arg->cols;
    const uint32_t cols = arg->cols;

    float s2 = 0.0f;
    for (uint32_t j = threadIdx.x; j < cols; j += PRIM_BLOCK) {
        s2 += in[j] * in[j];
    }
    s2 = warp_sum(s2);
    if (threadIdx.x % warpSize == 0) {
        tbuf[threadIdx.x / warpSize] = s2;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = 0.0f;
        for (uint32_t w = 0; w < PRIM_BLOCK / warpSize; ++w) v += tbuf[w];
        tbuf[16] = rsqrtf(v / (float)cols + arg->eps);  // 1/rms
    }
    __syncthreads();
    const float inv_rms = tbuf[16];

    for (uint32_t j = threadIdx.x; j < cols; j += PRIM_BLOCK) {
        out[j] = in[j] * inv_rms * gamma[j];
    }
}
