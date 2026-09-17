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

#ifndef VORTEX_DL_SPARSE24_H
#define VORTEX_DL_SPARSE24_H

// Vortex DL 2:4 structured-sparsity layer (plan P6-02 Q5) over the
// vortex2.h ABI: magnitude top-2-of-4 pruning to a masked values array,
// and a metadata-driven sparse GEMM (dense f32 activations, FP32
// accumulate).
//
// Data layout (see sparse24_args.h for the full contract):
//   values: N x K f32 row-major, pruned slots exactly +0.0f;
//   mask:   N x ceil(K/4) uint8, 2 bits per element (1 = kept).

#include <stdint.h>
#include <vortex2.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum vx_sparse24_status {
    VX_SPARSE24_OK = 0,
    VX_SPARSE24_ERR_NOT_INITIALIZED = 1,
    VX_SPARSE24_ERR_BAD_ARGS = 2,
    VX_SPARSE24_ERR_LAUNCH = 3,
    // init is idempotent only for the same device and image.
    VX_SPARSE24_ERR_ALREADY_INITIALIZED = 4,
} vx_sparse24_status;

vx_sparse24_status vx_sparse24_init(vx_device_h dev, const char* vxbin_path);
vx_sparse24_status vx_sparse24_finalize(void);

// Prune dense weights (N x K f32) to values + mask in place on device.
// One launch, async; bit-deterministic (tie-break: lower index wins).
vx_sparse24_status vx_sparse24_prune(vx_queue_h q, uint64_t weights,
                                     uint64_t values, uint64_t mask,
                                     uint32_t n, uint32_t k);

// C[M][N] = act[M][K] x values[N][K]^T; the kernel consults the mask and
// skips pruned elements. out: M x N f32. Async.
vx_sparse24_status vx_sparse24_gemm(vx_queue_h q, uint64_t act,
                                    uint64_t values, uint64_t mask,
                                    uint64_t out, uint32_t m, uint32_t n,
                                    uint32_t k);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // VORTEX_DL_SPARSE24_H
