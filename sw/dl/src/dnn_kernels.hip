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
        int64_t first_y = (int64_t)oy * arg->sh - arg->ph;
        int64_t first_x = (int64_t)ox * arg->sw - arg->pw;
        if (first_y < 0) {
            first_y += ((-first_y + arg->dh - 1) / arg->dh) * arg->dh;
        }
        if (first_x < 0) {
            first_x += ((-first_x + arg->dw - 1) / arg->dw) * arg->dw;
        }
        int64_t max_index = first_y * arg->wi + first_x;
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
                    // Finite ties keep the first index; each NaN replaces it.
                    if (v > acc || isnan(v)) {
                        acc = v;
                        max_index = (int64_t)iy * arg->wi + ix;
                    }
                } else {
                    acc += v;
                }
                ++count;
            }
        }
        if (arg->op == 0) {
            out_row[ox] = acc;
            if (arg->indices) {
                ((int64_t*)(uintptr_t)arg->indices)[
                    (((size_t)n * arg->c + c) * arg->ho + oy) * arg->wo + ox] = max_index;
            }
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

static __device__ uint32_t resize_bound(int64_t i, uint32_t size) {
  return i < 0 ? 0 : (i >= size ? size - 1 : (uint32_t)i);
}

static __device__ float resize_cubic_weight(float x) {
  x = fabsf(x);
  if (x <= 1.0f) {
    return (1.25f * x - 2.25f) * x * x + 1.0f;
  }
  return ((-0.75f * x + 3.75f) * x - 6.0f) * x + 3.0f;
}

__global__ void dnn_resize_kernel(vx_dnn_resize_args_t* arg) {
  uint32_t spatial = 1;
  for (uint32_t d = 0; d < arg->ndim; ++d) {
    spatial *= arg->output_size[d];
  }
  const float* src = (const float*)(uintptr_t)arg->in +
                    (size_t)blockIdx.z * arg->strides[0] +
                    (size_t)blockIdx.y * arg->strides[1];
  float* dst = (float*)(uintptr_t)arg->out +
              ((size_t)blockIdx.z * arg->c + blockIdx.y) * spatial;
  for (uint32_t i = blockIdx.x; i < spatial; i += gridDim.x) {
    uint32_t coord[3] = {}, rem = i;
    for (uint32_t d = arg->ndim; d-- > 0;) {
      coord[d] = rem % arg->output_size[d];
      rem /= arg->output_size[d];
    }
    if (arg->mode < 2) {
      size_t offset = 0;
      for (uint32_t d = 0; d < arg->ndim; ++d) {
        int64_t index;
        if (arg->mode == 0 && arg->input_size[d] == arg->output_size[d]) {
          index = coord[d];
        } else if (arg->mode == 0 && (uint64_t)arg->input_size[d] * 2 == arg->output_size[d]) {
          index = coord[d] / 2;
        } else {
          index = (int64_t)floorf((coord[d] + (arg->mode == 1 ? 0.5f : 0.0f)) * arg->scales[d]);
        }
        offset += (size_t)resize_bound(index, arg->input_size[d]) * arg->strides[d + 2];
      }
      dst[i] = src[offset];
    } else {
      int64_t base[3] = {};
      float fraction[3] = {};
      for (uint32_t d = 0; d < arg->ndim; ++d) {
        float position = arg->align_corners ? coord[d] * arg->scales[d]
                          : (coord[d] + 0.5f) * arg->scales[d] - 0.5f;
        if (arg->mode == 2) {
          if (arg->input_size[d] == arg->output_size[d]) {
            position = coord[d];
          }
          position = fmaxf(position, 0.0f);
        }
        base[d] = (int64_t)floorf(position);
        fraction[d] = position - (float)base[d];
      }
      float result = 0.0f;
      if (arg->mode == 3) {
        for (uint32_t y = 0; y < 4; ++y) {
          float row = 0.0f;
          for (uint32_t x = 0; x < 4; ++x) {
            const size_t offset = (size_t)resize_bound(base[0] + y - 1, arg->input_size[0]) * arg->strides[2]
                                 + (size_t)resize_bound(base[1] + x - 1, arg->input_size[1]) * arg->strides[3];
            row += src[offset] * resize_cubic_weight((float)x - 1.0f - fraction[1]);
          }
          result += row * resize_cubic_weight((float)y - 1.0f - fraction[0]);
        }
      } else {
        for (uint32_t corner = 0; corner < (1u << arg->ndim); ++corner) {
          float weight = 1.0f;
          size_t offset = 0;
          for (uint32_t d = 0; d < arg->ndim; ++d) {
            const uint32_t upper = (corner >> d) & 1;
            weight *= upper ? fraction[d] : 1.0f - fraction[d];
            offset += (size_t)resize_bound(base[d] + upper, arg->input_size[d]) * arg->strides[d + 2];
          }
          result += weight * src[offset];
        }
      }
      dst[i] = result;
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
