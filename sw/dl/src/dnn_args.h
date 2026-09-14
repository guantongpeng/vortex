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
//   y = (x - mean[c]) * rstd[c] * weight[c] + bias[c]
typedef struct {
    vx_dl_ptr_t in;
    vx_dl_ptr_t mean;     // [c]
    vx_dl_ptr_t rstd;     // [c] = 1/sqrt(var + eps)
    vx_dl_ptr_t weight;   // [c]
    vx_dl_ptr_t bias;     // [c]
    vx_dl_ptr_t out;
    uint32_t total;       // n*c*hi*wi
    uint32_t c;
} vx_dnn_bn_args_t;

#endif // VORTEX_DL_DNN_ARGS_H
