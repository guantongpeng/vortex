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

// Vortex DL dnn kernels (plan P3-02 second tier): direct 2D convolution,
// windowed pooling and inference batch-norm (fused affine). NCHW FP32.
// Direct conv rather than implicit GEMM is deliberate at this node: the
// verification target is correctness on the models' small layers, and the
// tile/GEMM reuse happens through the TCU path later. One 4-warp CTA per
// (output channel, output row); LMEM stages the weight for that channel.

#include <hip/hip_runtime.h>
#include <math.h>

#include "dnn_args.h"

#define DNN_WMAX 16  // max kernel width staged in LMEM (kh*kw floats)

__global__ void dnn_conv2d_kernel(vx_dnn_conv_args_t* arg) {
    // grid: (wo_blocks?, co, n) — one CTA computes one (n, co, oy) row.
    const uint32_t n = blockIdx.z;
    const uint32_t co = blockIdx.y;
    const uint32_t oy = blockIdx.x;

    const uint32_t ci_group = arg->ci / arg->groups;
    const uint32_t group = co / (arg->co / arg->groups);
    const uint32_t ci_base = group * ci_group;
    const float* w = (const float*)(uintptr_t)arg->weight +
                     ((size_t)co * ci_group) * (arg->kh * arg->kw);
    float* stage = (float*)__local_mem();  // [ci][kh*kw]

    // Stage this output channel's weights (CI*KH*KW floats, 16 threads).
    const uint32_t wtotal = ci_group * arg->kh * arg->kw;
    for (uint32_t e = threadIdx.x; e < wtotal && e < ci_group * DNN_WMAX;
         e += 16) {
        stage[e] = w[e];
    }
    __syncthreads();
    const uint32_t wstaged = wtotal < ci_group * DNN_WMAX ? wtotal
                                                         : ci_group * DNN_WMAX;

    const float* in_base = (const float*)(uintptr_t)arg->in +
                           (size_t)n * arg->ci * arg->hi * arg->wi;
    float* out_row = (float*)(uintptr_t)arg->out +
                     (((size_t)n * arg->co + co) * arg->ho + oy) * arg->wo;
    const float bias = arg->has_bias
                           ? ((const float*)(uintptr_t)arg->bias)[co]
                           : 0.0f;

    // Output columns split across the CTA's threads.
    for (uint32_t ox = threadIdx.x; ox < arg->wo; ox += 16) {
        float acc = bias;
        uint32_t we = 0;
        for (uint32_t ci = 0; ci < ci_group; ++ci) {
            const float* in_ch = in_base + (size_t)(ci_base + ci) * arg->hi * arg->wi;
            for (uint32_t kh = 0; kh < arg->kh; ++kh) {
                const int32_t iy = (int32_t)(oy * arg->sh + kh * arg->dh) -
                                   (int32_t)arg->ph;
                if (iy < 0 || iy >= (int32_t)arg->hi) {
                    we += arg->kw;
                    continue;
                }
                for (uint32_t kw = 0; kw < arg->kw; ++kw) {
                    const int32_t ix = (int32_t)(ox * arg->sw + kw * arg->dw) -
                                       (int32_t)arg->pw;
                    if (ix >= 0 && ix < (int32_t)arg->wi && we < wstaged) {
                        acc += in_ch[(size_t)iy * arg->wi + ix] * stage[we];
                    }
                    ++we;
                }
            }
        }
        out_row[ox] = acc;
    }
}

__global__ void dnn_pool2d_kernel(vx_dnn_pool_args_t* arg) {
    const uint32_t n = blockIdx.z;
    const uint32_t c = blockIdx.y;
    const uint32_t oy = blockIdx.x;

    const float* in_ch = (const float*)(uintptr_t)arg->in +
                         (((size_t)n * arg->c) + c) * arg->hi * arg->wi;
    float* out_row = (float*)(uintptr_t)arg->out +
                     (((size_t)n * arg->c + c) * arg->ho + oy) * arg->wo;

    // Output columns split across the CTA's threads.
    for (uint32_t ox = threadIdx.x; ox < arg->wo; ox += 16) {
        // -INFINITY, not a finite sentinel: -3.4e38f dropped -inf inputs and
        // turned an all-NaN window into that sentinel instead of propagating
        // NaN (F10 in docs/mydocs/pytorch_plan.md).
        float acc = (arg->op == 0) ? -INFINITY : 0.0f;
        uint32_t count = 0;
        uint32_t include_count = 0;
        for (uint32_t kh = 0; kh < arg->kh; ++kh) {
            const int32_t iy = (int32_t)(oy * arg->sh + kh) - (int32_t)arg->ph;
            if (arg->op == 3 && iy >= -(int32_t)arg->ph &&
                iy < (int32_t)(arg->hi + arg->ph)) {
                for (uint32_t kw = 0; kw < arg->kw; ++kw) {
                    const int32_t ix = (int32_t)(ox * arg->sw + kw) -
                                       (int32_t)arg->pw;
                    if (ix >= -(int32_t)arg->pw &&
                        ix < (int32_t)(arg->wi + arg->pw)) {
                        ++include_count;
                    }
                }
            }
            if (iy < 0 || iy >= (int32_t)arg->hi) continue;
            for (uint32_t kw = 0; kw < arg->kw; ++kw) {
                const int32_t ix = (int32_t)(ox * arg->sw + kw) - (int32_t)arg->pw;
                if (ix < 0 || ix >= (int32_t)arg->wi) continue;
                float v = in_ch[(size_t)iy * arg->wi + ix];
                if (arg->op == 0) {
                    // NaN must win: plain `v > acc` (and fmaxf, which ignores
                    // NaN) would leave the accumulator at the sentinel.
                    if (v > acc || isnan(v)) acc = v;
                } else {
                    acc += v;
                }
                ++count;
            }
        }
        if (arg->op == 0) {
            out_row[ox] = acc;
        } else if (arg->op == 1 || arg->op == 2 || arg->op == 3) {
            const uint32_t divisor = arg->divisor
                                         ? arg->divisor
                                         : (arg->op == 1 ? count
                                                         : (arg->op == 3
                                                                ? include_count
                                                                : arg->kh * arg->kw));
            out_row[ox] = divisor ? acc / (float)divisor : 0.0f;
        }
    }
}

__global__ void dnn_bn_affine_kernel(vx_dnn_bn_args_t* arg) {
    const float* mean = (const float*)(uintptr_t)arg->mean;
    const float* var = (const float*)(uintptr_t)arg->var;
    const float* weight = (const float*)(uintptr_t)arg->weight;
    const float* bias = (const float*)(uintptr_t)arg->bias;
    // hw, not total/c: the latter is N*H*W and picks the right channel only at
    // N == 1.
    const uint32_t plane = arg->hw;
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < arg->total; idx += blockDim.x * gridDim.x) {
        const uint32_t ch = (idx / plane) % arg->c;
        const float x = ((const float*)(uintptr_t)arg->in)[idx];
        // 1/sqrt, matching what the CPU reference computes, rather than rsqrt
        const float rstd = 1.0f / sqrtf(var[ch] + arg->eps);
        // Optional affine: ATen's batch norm allows weight/bias to be absent,
        // and the caller signals that by leaving both pointers null rather
        // than by passing buffers of ones and zeros.
        const float w = arg->has_affine ? weight[ch] : 1.0f;
        const float b = arg->has_affine ? bias[ch] : 0.0f;
        ((float*)(uintptr_t)arg->out)[idx] = (x - mean[ch]) * rstd * w + b;
    }
}
