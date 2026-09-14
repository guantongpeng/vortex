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

#ifndef VORTEX_DL_PRIM_ARGS_H
#define VORTEX_DL_PRIM_ARGS_H

// Kernel argument blocks for the prim/norm kernels (plan P3-02).
// Same conventions as blas_args.h: device-width pointers, shared between
// the KMU image and the host library. Do not cache these fields into
// locals inside kernels (see the VOLT workaround note in blas_kernels.hip).

#include <stdint.h>

#if defined(__VORTEX__)
typedef uintptr_t vx_dl_ptr_t;
#elif defined(VX_CFG_XLEN) && VX_CFG_XLEN == 32
typedef uint32_t vx_dl_ptr_t;
#else
typedef uint64_t vx_dl_ptr_t;
#endif

typedef enum {
    VX_PRIM_RELU = 0,
    VX_PRIM_GELU = 1,
    VX_PRIM_SILU = 2,
    VX_PRIM_NEG  = 3,
} vx_prim_op_e;

// Elementwise unary over n FP32 values.
typedef struct {
    vx_dl_ptr_t in;
    vx_dl_ptr_t out;
    uint32_t n;
    uint32_t op;   // vx_prim_op_e
} vx_prim_unary_args_t;

// Block reduce over n FP32 values. op 0=sum, 1=max, 2=argmax (out is u32).
typedef struct {
    vx_dl_ptr_t in;
    vx_dl_ptr_t out;
    uint32_t n;
    uint32_t op;
} vx_prim_reduce_args_t;

// Per-row softmax over rows x cols FP32 (row-major, one CTA per row).
typedef struct {
    vx_dl_ptr_t in;
    vx_dl_ptr_t out;
    uint32_t rows;
    uint32_t cols;
} vx_prim_rowargs_t;

// LayerNorm: mean/var over the row, then scale/gamma + shift/beta.
typedef struct {
    vx_dl_ptr_t in;
    vx_dl_ptr_t gamma;
    vx_dl_ptr_t beta;
    vx_dl_ptr_t out;
    uint32_t rows;
    uint32_t cols;
    float eps;
    uint32_t pad;
} vx_prim_norm_args_t;

// RMSNorm: rms = sqrt(mean(x^2) + eps); out = x / rms * gamma (+beta opt).
typedef struct {
    vx_dl_ptr_t in;
    vx_dl_ptr_t gamma;
    vx_dl_ptr_t out;
    uint32_t rows;
    uint32_t cols;
    float eps;
    uint32_t pad;
} vx_prim_rmsnorm_args_t;

#endif // VORTEX_DL_PRIM_ARGS_H
