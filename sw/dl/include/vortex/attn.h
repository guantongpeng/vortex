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

#ifndef VORTEX_DL_ATTN_H
#define VORTEX_DL_ATTN_H

// Vortex DL attention layer (plan P3 second tier): scaled dot-product
// attention forward over the vortex2.h ABI.
//
//   O[b][h][q][:] = softmax(scale * QK^T (+ causal mask)) V
//
// Tensors are B*H*L*D FP32 device buffers, row-major [b][h][l][d].
// Constraints: L <= 2048 (LMEM score staging), any B/H/D. The kernel is
// one 16-thread CTA per output row; see src/attn_args.h for the exact
// arg-block layout metadata.

#include <stdint.h>
#include <vortex2.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum vx_attn_status {
    VX_ATTN_OK = 0,
    VX_ATTN_ERR_NOT_INITIALIZED = 1,
    VX_ATTN_ERR_BAD_ARGS = 2,
    VX_ATTN_ERR_LAUNCH = 3,
    VX_ATTN_ERR_UNSUPPORTED = 4,
} vx_attn_status;

vx_attn_status vx_attn_init(vx_device_h dev, const char* vxbin_path);
vx_attn_status vx_attn_finalize(void);

// scale = 1/sqrt(D), computed in double on the host and passed to the
// kernel as FP32 (the device never calls sqrtf, so a double-precision
// reference can mirror the exact scale bit-for-bit). causal: 0 = full
// attention, nonzero = causal (row q attends columns j <= q). Async.
vx_attn_status vx_attn_forward(vx_queue_h q, uint64_t qmat, uint64_t kmat,
                               uint64_t vmat, uint64_t out,
                               uint32_t b, uint32_t h, uint32_t l,
                               uint32_t d, uint32_t causal);

// Explicit-scale variant (window attention / SAM-DINOv3 style callers
// pass their own scale). Same semantics otherwise.
vx_attn_status vx_attn_forward_scaled(vx_queue_h q, uint64_t qmat,
                                      uint64_t kmat, uint64_t vmat,
                                      uint64_t out, uint32_t b, uint32_t h,
                                      uint32_t l, uint32_t d, uint32_t causal,
                                      double scale);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // VORTEX_DL_ATTN_H
