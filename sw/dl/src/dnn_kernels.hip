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

#include <hip/hip_runtime.h>
#include <math.h>

#include "dnn_args.h"

__global__ void dnn_conv2d_kernel(vx_dnn_conv_args_t* arg) {
  const uint32_t n = blockIdx.z;
  const uint32_t co = blockIdx.y;
  const uint32_t width_tiles = (arg->wo + 15) / 16;
  const uint32_t oy = blockIdx.x / width_tiles;
  const uint32_t ox = (blockIdx.x % width_tiles) * 16 + threadIdx.x;
  const uint32_t ci_group = arg->ci / arg->groups;
  const uint32_t ci_base = (co / (arg->co / arg->groups)) * ci_group;
  const uint32_t area = arg->kh * arg->kw;
  const uint32_t total = ci_group * area;
  const float* weights = (const float*)(uintptr_t)arg->weight + (size_t)co * total;
  const float* input = (const float*)(uintptr_t)arg->in +
                      ((size_t)n * arg->ci + ci_base) * arg->hi * arg->wi;
  float* stage = (float*)__local_mem();
  float acc = arg->has_bias ? ((const float*)(uintptr_t)arg->bias)[co] : 0.0f;
  for (uint32_t start = 0; start < total;) {
    const uint32_t count = total - start < VX_DNN_CONV_TILE
                               ? total - start : VX_DNN_CONV_TILE;
    for (uint32_t i = threadIdx.x; i < count; i += 16) {
      stage[i] = weights[start + i];
    }
    __syncthreads();
    if (ox < arg->wo) {
      for (uint32_t i = 0; i < count; ++i) {
        const uint32_t e = start + i;
        const uint32_t ci = e / area;
        const uint32_t kh = (e % area) / arg->kw;
        const uint32_t kw = e % arg->kw;
        const int64_t iy = (int64_t)oy * arg->sh + (int64_t)kh * arg->dh - arg->ph;
        const int64_t ix = (int64_t)ox * arg->sw + (int64_t)kw * arg->dw - arg->pw;
        if (iy >= 0 && iy < arg->hi && ix >= 0 && ix < arg->wi) {
          acc += input[((size_t)ci * arg->hi + iy) * arg->wi + ix] * stage[i];
        }
      }
    }
    // Tail lanes still participate before the next tile overwrites shared data.
    __syncthreads();
    start += count;
  }
  if (ox < arg->wo) {
    ((float*)(uintptr_t)arg->out)[(((size_t)n * arg->co + co) * arg->ho + oy)
                                    * arg->wo + ox] = acc;
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
            const int32_t iy = (int32_t)(oy * arg->sh + kh * arg->dh) -
                               (int32_t)arg->ph;
            if (arg->op == 3 && iy >= -(int32_t)arg->ph &&
                iy < (int32_t)(arg->hi + arg->ph)) {
                for (uint32_t kw = 0; kw < arg->kw; ++kw) {
                    const int32_t ix = (int32_t)(ox * arg->sw + kw * arg->dw) -
                                       (int32_t)arg->pw;
                    if (ix >= -(int32_t)arg->pw &&
                        ix < (int32_t)(arg->wi + arg->pw)) {
                        ++include_count;
                    }
                }
            }
            if (iy < 0 || iy >= (int32_t)arg->hi) continue;
            for (uint32_t kw = 0; kw < arg->kw; ++kw) {
                const int32_t ix = (int32_t)(ox * arg->sw + kw * arg->dw) -
                                   (int32_t)arg->pw;
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
