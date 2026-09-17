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
    VX_PRIM_OP_RELU = 0,
    VX_PRIM_OP_GELU = 1,   // tanh approximation
    VX_PRIM_OP_SILU = 2,
    VX_PRIM_OP_NEG  = 3,
    VX_PRIM_OP_SUM  = 4,
    VX_PRIM_OP_MAX  = 5,
    VX_PRIM_OP_ARGMAX = 6, // out is uint32
} vx_prim_op;

vx_prim_status vx_prim_init(vx_device_h dev, const char* vxbin_path);
vx_prim_status vx_prim_finalize(void);

// Elementwise unary: out[i] = act(in[i]).
vx_prim_status vx_prim_unary(vx_queue_h q, vx_prim_op op,
                             uint64_t in, uint64_t out, uint32_t n);

// Reductions (single-CTA kernel over the whole vector).
vx_prim_status vx_prim_reduce(vx_queue_h q, vx_prim_op op,
                              uint64_t in, uint64_t out, uint32_t n);

// Row-wise softmax over rows x cols.
vx_prim_status vx_prim_softmax(vx_queue_h q, uint64_t in, uint64_t out,
                               uint32_t rows, uint32_t cols);

// LayerNorm / RMSNorm, one CTA per row; gamma/beta are cols-length.
vx_prim_status vx_prim_layernorm(vx_queue_h q, uint64_t in, uint64_t gamma,
                                 uint64_t beta, uint64_t out,
                                 uint32_t rows, uint32_t cols, float eps);
vx_prim_status vx_prim_rmsnorm(vx_queue_h q, uint64_t in, uint64_t gamma,
                               uint64_t out, uint32_t rows, uint32_t cols,
                               float eps);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // VORTEX_DL_PRIM_H
