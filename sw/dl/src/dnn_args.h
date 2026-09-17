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

#ifndef VORTEX_DL_DNN_ARGS_H
#define VORTEX_DL_DNN_ARGS_H

// Kernel argument blocks for the dnn kernels (plan P3-02 second tier).
// Conventions as blas_args.h: device-width pointers, NCHW fp32.

#include <stdint.h>

#if defined(__VORTEX__)
typedef uintptr_t vx_dl_ptr_t;
#elif defined(VX_CFG_XLEN) && VX_CFG_XLEN == 32
typedef uint32_t vx_dl_ptr_t;
#else
typedef uint64_t vx_dl_ptr_t;
#endif

// Direct 2D convolution, NCHW, dilation 1, groups 1.
//   out[n][co][oy][ox] = sum_{ci,kh,kw} in[n][ci][oy*sh + kh - ph][ox*sw + kw - pw] * w[co][ci][kh][kw] (+ b[co])
// Weight layout: [CO][CI][KH][KW]; pad symmetric ph x pw.
typedef struct {
    vx_dl_ptr_t in;
    vx_dl_ptr_t weight;
    vx_dl_ptr_t bias;   // may be 0 (no bias)
    vx_dl_ptr_t out;
    uint32_t n, ci, hi, wi;
    uint32_t co, ho, wo;
    uint32_t kh, kw, ph, pw, sh, sw;
    uint32_t has_bias;
} vx_dnn_conv_args_t;

// Windowed pooling over HxW: op 0 = max, 1 = avg (count_include_pad=false
// semantics: divisor counts in-bounds elements only when padding present).
typedef struct {
    vx_dl_ptr_t in;
    vx_dl_ptr_t out;
    uint32_t n, c, hi, wi, ho, wo;
    uint32_t kh, kw, ph, pw, sh, sw;
    uint32_t op;
} vx_dnn_pool_args_t;

// Inference batch norm as a fused per-channel affine:
//   y = (x - mean[c]) * (1/sqrt(var[c] + eps)) * weight[c] + bias[c]
//
// `hw` is the spatial span H*W and is passed explicitly rather than derived.
// Deriving it as total/c gives N*H*W, which is the right per-channel span only
// when N == 1 -- the bug torch-vortex's copy of this kernel had and fixed
// (F03 in docs/mydocs/pytorch_plan.md).
//
// `var` is the running variance and eps is applied here, so the caller does
// not have to make a host round-trip to take a square root per channel.
typedef struct {
    vx_dl_ptr_t in;
    vx_dl_ptr_t mean;     // [c]
    vx_dl_ptr_t var;      // [c] running variance
    vx_dl_ptr_t weight;   // [c]
    vx_dl_ptr_t bias;     // [c]
    vx_dl_ptr_t out;
    uint32_t total;       // n*c*hw
    uint32_t c;
    uint32_t hw;          // H*W, the per-channel span
    float eps;
} vx_dnn_bn_args_t;

#endif // VORTEX_DL_DNN_ARGS_H
