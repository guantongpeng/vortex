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

// Unary operations. The numbering is append-only: existing values stay put so
// an image and a host built at different times still agree.
//
// GELU is spelled out because it has two forms and torch's default is the erf
// one; an unsuffixed GELU that meant the tanh approximation is how a caller
// gets the wrong function without noticing.
typedef enum {
    VX_PRIM_RELU = 0,
    VX_PRIM_GELU_TANH = 1,   // was VX_PRIM_GELU; same value, clearer name
    VX_PRIM_SILU = 2,
    VX_PRIM_NEG  = 3,
    VX_PRIM_ABS = 4,
    VX_PRIM_EXP = 5,
    VX_PRIM_LOG = 6,
    VX_PRIM_SQRT = 7,
    VX_PRIM_RSQRT = 8,
    VX_PRIM_SIGMOID = 9,
    VX_PRIM_TANH = 10,
    VX_PRIM_RECIPROCAL = 11,
    VX_PRIM_GELU_ERF = 12,   // torch's default gelu
    VX_PRIM_COS = 19,
    VX_PRIM_SIN = 20,
} vx_prim_op_e;

// Elementwise unary over n FP32 values.
typedef struct {
    vx_dl_ptr_t in;
    vx_dl_ptr_t out;
    uint32_t n;
    uint32_t op;   // vx_prim_op_e
} vx_prim_unary_args_t;

// Binary elementwise operations over n FP32 values. The three addresses are
// independent so in-place callers can use dst == a or dst == b safely.
typedef enum {
    VX_PRIM_BIN_ADD = 0,
    VX_PRIM_BIN_SUB = 1,
    VX_PRIM_BIN_MUL = 2,
    VX_PRIM_BIN_DIV = 3,
    VX_PRIM_BIN_MAXIMUM = 4,
    VX_PRIM_BIN_MINIMUM = 5,
} vx_prim_binary_op_e;

typedef struct {
    vx_dl_ptr_t dst;
    vx_dl_ptr_t a;
    vx_dl_ptr_t b;
    uint32_t n;
    uint32_t op;   // vx_prim_binary_op_e
} vx_prim_binary_args_t;

// Scalar elementwise operation. `reverse` selects value <op> a, which is
// needed for scalar - tensor and scalar / tensor.
typedef struct {
    vx_dl_ptr_t dst;
    vx_dl_ptr_t a;
    float value;
    uint32_t n;
    uint32_t op;   // vx_prim_binary_op_e
    uint32_t reverse;
    uint32_t pad;
} vx_prim_scalar_args_t;

// Strided broadcast operation. The output is contiguous and the operand
// strides are measured in FP32 elements; zero repeats a broadcast dimension.
typedef struct {
    vx_dl_ptr_t dst;
    vx_dl_ptr_t a;
    vx_dl_ptr_t b;
    uint32_t op;   // vx_prim_binary_op_e
    uint32_t ndim;
    uint32_t total;
    uint32_t pad;
    uint32_t sizes[4];
    uint32_t a_strides[4];
    uint32_t b_strides[4];
} vx_prim_broadcast_args_t;

// Reduction operations, one row at a time. Append-only, same as the unary
// numbering, and for the same reason: the host picks the kernel's op with a
// subtraction, so a value that moved would select a different reduction
// without any diagnostic.
typedef enum {
    VX_PRIM_RED_SUM = 0,
    VX_PRIM_RED_MAX = 1,
    VX_PRIM_RED_ARGMAX = 2,   // out is uint32
    VX_PRIM_RED_MEAN = 3,
    // No MIN/ARGMIN yet. The kernel body for them is written, but seeding an
    // accumulator from arg->op makes VOLT drop the LMEM store and every
    // reduction in the image returns its seed -- measured, and reproduced by
    // adding nothing but that seed. See docs/mydocs/05_pytorch/torch_p5_10_argmax.md.
} vx_prim_reduce_op_e;

// Row-wise reduce over rows x cols FP32 (row-major, one CTA per row); out[r]
// is the reduction of row r. A reduction over a whole vector is rows = 1,
// which is also the only shape the previous single-CTA version could express
// -- hence the extra field rather than a second entry point.
//
// `values` is the extreme value for ARGMAX: the kernel already carries it
// while looking for the index, and torch's max(dim=) returns the pair, so
// writing it here is what keeps that one pass. Zero means the caller does not
// want it.
typedef struct {
    vx_dl_ptr_t in;
    vx_dl_ptr_t out;
    vx_dl_ptr_t values;   // arg ops only: rows floats, or 0
    uint32_t rows;
    uint32_t cols;
    uint32_t op;          // vx_prim_reduce_op_e
    uint32_t pad;
} vx_prim_reduce_args_t;

// The row-wise family over rows x cols FP32 (row-major, one CTA per row).
// All three share a kernel because they share both passes: a row max, then a
// sum of exponentials about it. What differs is only what is written out.
//
// LOGSUMEXP writes one float per row rather than cols of them, which is why
// the caller passes a rows-length buffer for it.
typedef enum {
    VX_PRIM_ROW_SOFTMAX = 0,
    VX_PRIM_ROW_LOG_SOFTMAX = 1,
    VX_PRIM_ROW_LOGSUMEXP = 2,
} vx_prim_row_op_e;

typedef struct {
    vx_dl_ptr_t in;
    vx_dl_ptr_t out;
    uint32_t rows;
    uint32_t cols;
    uint32_t op;   // vx_prim_row_op_e
    uint32_t pad;
} vx_prim_rowargs_t;

// LayerNorm: mean/var over the row, then scale/gamma + shift/beta.
//
// gamma and beta are optional: a null address means no affine, which is what
// F.layer_norm(x, shape) and nn.LayerNorm(elementwise_affine=False) ask for.
// Giving only one of them is a caller error, not a mode.
//
// mean and rstd are also optional outputs, one float per row:
// aten::native_layer_norm returns them and there is no second entry point that
// would compute them. rstd is 1/sqrt(var + eps), the reciprocal.
typedef struct {
    vx_dl_ptr_t in;
    vx_dl_ptr_t gamma;   // 0 for none
    vx_dl_ptr_t beta;    // 0 for none
    vx_dl_ptr_t out;
    vx_dl_ptr_t mean;    // rows floats, or 0
    vx_dl_ptr_t rstd;    // rows floats, or 0
    uint32_t rows;
    uint32_t cols;
    float eps;
    uint32_t pad;
} vx_prim_norm_args_t;

// RMSNorm: rms = sqrt(mean(x^2) + eps); out = x / rms * gamma.
typedef struct {
    vx_dl_ptr_t in;
    vx_dl_ptr_t gamma;   // 0 for none
    vx_dl_ptr_t out;
    uint32_t rows;
    uint32_t cols;
    float eps;
    uint32_t pad;
} vx_prim_rmsnorm_args_t;

#endif // VORTEX_DL_PRIM_ARGS_H
