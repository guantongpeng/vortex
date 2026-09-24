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

#ifndef VORTEX_DL_MAMBA_H
#define VORTEX_DL_MAMBA_H

// Vortex DL Mamba selective scan host layer (plan P7 precursor): loads the
// mamba.vxbin KMU image and dispatches the scan kernel over the vortex2.h
// ABI. Simplified S6 software reference:
//   S[n] <- exp(A*dt) * S[n] + B[t][n] * x[t];  y[t] = sum_n C[t][n]*S[n]
// Layouts and arg-block metadata: see mamba_args.h (args.h here).

#include <stdint.h>
#include <vortex2.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum vx_mamba_status {
    VX_MAMBA_OK = 0,
    VX_MAMBA_ERR_NOT_INITIALIZED = 1,
    VX_MAMBA_ERR_BAD_ARGS = 2,
    VX_MAMBA_ERR_LAUNCH = 3,
    // init is idempotent only for the same device and image.
    VX_MAMBA_ERR_ALREADY_INITIALIZED = 4,
} vx_mamba_status;

vx_mamba_status vx_mamba_init(vx_device_h dev, const char* vxbin_path);
vx_mamba_status vx_mamba_finalize(void);

// One warp per (batch, channel); 16-thread CTAs (4 channels per CTA).
// All buffers FP32 device addresses with the layouts documented in
// mamba_args.h. Requires 1 <= dstate <= 8. Async (enqueue only).
vx_mamba_status vx_mamba_selective_scan(vx_queue_h q, uint64_t a,
                                        uint64_t dt, uint64_t b, uint64_t c,
                                        uint64_t x, uint64_t y,
                                        uint32_t batch, uint32_t channels,
                                        uint32_t seqlen, uint32_t dstate);

// Same recurrence and layouts, with an optional [batch, channels, dstate]
// initial state and a required final state. State size is not register-limited.
vx_mamba_status vx_mamba_selective_scan_state(
    vx_queue_h q, uint64_t a, uint64_t dt, uint64_t b, uint64_t c,
    uint64_t x, uint64_t y, uint64_t initial_state, uint64_t final_state,
    uint32_t batch, uint32_t channels, uint32_t seqlen, uint32_t dstate);

#ifdef __cplusplus
}
#endif

#endif // VORTEX_DL_MAMBA_H
