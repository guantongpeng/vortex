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
    VX_PRIM_OP_ARGMAX = 15,     // out is uint32
    VX_PRIM_OP_MEAN = 16,
} vx_prim_op;

vx_prim_status vx_prim_init(vx_device_h dev, const char* vxbin_path);
vx_prim_status vx_prim_finalize(void);

// Elementwise unary: out[i] = act(in[i]).
vx_prim_status vx_prim_unary(vx_queue_h q, vx_prim_op op,
                             uint64_t in, uint64_t out, uint32_t n);

// Row-wise reduction over a rows x cols FP32 row-major buffer; out[r] is the
// reduction of row r. A whole-vector reduction is rows = 1.
vx_prim_status vx_prim_reduce(vx_queue_h q, vx_prim_op op,
                              uint64_t in, uint64_t out,
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

// LayerNorm, one CTA per row; gamma/beta are cols-length. Either both are 0
// (no affine) or both are addresses -- one of the two is a caller error.
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
