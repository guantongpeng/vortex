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

#ifndef VORTEX_DL_PRIM_H
#define VORTEX_DL_PRIM_H

// Vortex DL prim/norm layer (plan P3-02): elementwise activations, block
// reductions, softmax, layernorm and rmsnorm over the vortex2.h ABI.
// Kernels are KMU images from sw/dl/src/prim_kernels.hip; the host layer
// mirrors vx_blas. All launches are async (sync via the queue).

#include <stdint.h>
#include <vortex2.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum vx_prim_status {
    VX_PRIM_OK = 0,
    VX_PRIM_ERR_NOT_INITIALIZED = 1,
    VX_PRIM_ERR_BAD_ARGS = 2,
    VX_PRIM_ERR_LAUNCH = 3,
    // init is idempotent only for the same device and image.
    VX_PRIM_ERR_ALREADY_INITIALIZED = 4,
} vx_prim_status;

typedef enum vx_prim_op {
    // Elementwise unary. The values are the kernel's own numbering verbatim
    // (src/prim_args.h), because the host passes this straight through;
    // prim_host.cpp pins every one of them.
    VX_PRIM_OP_RELU = 0,
    VX_PRIM_OP_GELU_TANH = 1,   // approximate="tanh"
    VX_PRIM_OP_SILU = 2,
    VX_PRIM_OP_NEG  = 3,
    VX_PRIM_OP_ABS = 4,
    VX_PRIM_OP_EXP = 5,
    VX_PRIM_OP_LOG = 6,
    VX_PRIM_OP_SQRT = 7,
    VX_PRIM_OP_RSQRT = 8,
    VX_PRIM_OP_SIGMOID = 9,
    VX_PRIM_OP_TANH = 10,
    VX_PRIM_OP_RECIPROCAL = 11,
    VX_PRIM_OP_GELU_ERF = 12,   // the default gelu, not the tanh one

    // Reductions, from these onwards. (op - VX_PRIM_OP_SUM) is the kernel's
    // VX_PRIM_RED_* numbering; prim_host.cpp pins that too.
    VX_PRIM_OP_SUM  = 13,
    VX_PRIM_OP_MAX  = 14,
    VX_PRIM_OP_ARGMAX = 15,     // index out; see vx_prim_index_reduce
    VX_PRIM_OP_MEAN = 16,
    VX_PRIM_OP_MIN = 17,
    VX_PRIM_OP_ARGMIN = 18,     // index out; see vx_prim_index_reduce
} vx_prim_op;

typedef enum vx_prim_binary_op {
    VX_PRIM_BINARY_ADD = 0,
    VX_PRIM_BINARY_SUB = 1,
    VX_PRIM_BINARY_MUL = 2,
    VX_PRIM_BINARY_DIV = 3,
    VX_PRIM_BINARY_MAXIMUM = 4,
    VX_PRIM_BINARY_MINIMUM = 5,
} vx_prim_binary_op;

vx_prim_status vx_prim_init(vx_device_h dev, const char* vxbin_path);
vx_prim_status vx_prim_finalize(void);

// Elementwise unary: out[i] = act(in[i]).
vx_prim_status vx_prim_unary(vx_queue_h q, vx_prim_op op,
                             uint64_t in, uint64_t out, uint32_t n);

vx_prim_status vx_prim_binary(vx_queue_h q, vx_prim_binary_op op,
                              uint64_t dst, uint64_t a, uint64_t b,
                              uint32_t n);

vx_prim_status vx_prim_scalar(vx_queue_h q, vx_prim_binary_op op,
                              uint64_t dst, uint64_t a, float value,
                              uint32_t n, uint32_t reverse);

vx_prim_status vx_prim_broadcast(vx_queue_h q, vx_prim_binary_op op,
                                 uint64_t dst, uint64_t a, uint64_t b,
                                 uint32_t total, uint32_t ndim,
                                 const uint32_t sizes[4],
                                 const uint32_t a_strides[4],
                                 const uint32_t b_strides[4]);

// Row-wise reduction over a rows x cols FP32 row-major buffer; out[r] is the
// reduction of row r. A whole-vector reduction is rows = 1. `out` is rows
// floats, and the op is one of SUM/MEAN/MAX/MIN -- the arg ops have their own
// entry point because their output is not floats.
vx_prim_status vx_prim_reduce(vx_queue_h q, vx_prim_op op,
                              uint64_t in, uint64_t out,
                              uint32_t rows, uint32_t cols);

// Row-wise maximum/minimum with its index, for VX_PRIM_OP_ARGMAX/ARGMIN.
// `indices` is rows uint32; `values` is rows floats and may be 0 if the caller
// wants only the index. ARGMAX ties go to the earliest element and a NaN beats
// every number, both of which match torch -- and because the value comes from
// the same pass as the index, values[r] is in[indices[r]].
vx_prim_status vx_prim_index_reduce(vx_queue_h q, vx_prim_op op, uint64_t in,
                                    uint64_t indices, uint64_t values,
                                    uint32_t rows, uint32_t cols);

// Row-wise softmax over rows x cols.
vx_prim_status vx_prim_softmax(vx_queue_h q, uint64_t in, uint64_t out,
                               uint32_t rows, uint32_t cols);

// The same two passes, written as log(x - max) - log(sum); out is rows x cols.
vx_prim_status vx_prim_log_softmax(vx_queue_h q, uint64_t in, uint64_t out,
                                   uint32_t rows, uint32_t cols);

// log(sum(exp(x - max))) + max, one value per row rather than cols of them.
vx_prim_status vx_prim_logsumexp(vx_queue_h q, uint64_t in, uint64_t out,
                                 uint32_t rows, uint32_t cols);

// LayerNorm, one CTA per row; gamma/beta are cols-length. Each may be 0
// independently: torch takes either alone, with the absent one meaning its
// identity (gamma 1, beta 0). Both 0 is the no-affine mode.
//
// mean and rstd are rows-length outputs, one per row, and may each be 0 if the
// caller does not want them. rstd is 1/sqrt(var + eps).
vx_prim_status vx_prim_layernorm(vx_queue_h q, uint64_t in, uint64_t gamma,
                                 uint64_t beta, uint64_t out, uint64_t mean,
                                 uint64_t rstd, uint32_t rows, uint32_t cols,
                                 float eps);

// RMSNorm, one CTA per row; gamma is cols-length and may be 0 (no affine).
vx_prim_status vx_prim_rmsnorm(vx_queue_h q, uint64_t in, uint64_t gamma,
                               uint64_t out, uint32_t rows, uint32_t cols,
                               float eps);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // VORTEX_DL_PRIM_H
