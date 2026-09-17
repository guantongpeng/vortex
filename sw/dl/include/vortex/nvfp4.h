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

#ifndef VORTEX_DL_NVFP4_H
#define VORTEX_DL_NVFP4_H

// NVFP4 software path (plan P6-02 Q4): FP4 E2M1 codes (two per byte) with
// per-group E4M3 scales and one per-tensor f32 scale;
// value = e2m1(code) * e4m3(scale) * tensor_scale.
// The hardware TCU NVFP4 variant gates on VX_TCU_DTYPE_NVFP4.

#include <stdint.h>
#include <vortex2.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VX_NVFP4_OK = 0,
    VX_NVFP4_ERR_BAD_ARGS = 1,
    VX_NVFP4_ERR_NOT_INITIALIZED = 2,
    VX_NVFP4_ERR_LAUNCH = 3,
    // init is idempotent only for the same device and image.
    VX_NVFP4_ERR_ALREADY_INITIALIZED = 4,
} vx_nvfp4_status;

vx_nvfp4_status vx_nvfp4_init(vx_device_h dev, const char* vxbin_path);
vx_nvfp4_status vx_nvfp4_finalize(void);

// packed: n*ceil(k/2) bytes; scales: n*ceil(k/group) e4m3 bytes.
vx_nvfp4_status vx_nvfp4_pack(vx_queue_h q, uint64_t weights,
                              uint64_t packed, uint64_t scales,
                              float tensor_scale, uint32_t n, uint32_t k,
                              uint32_t group);
vx_nvfp4_status vx_nvfp4_unpack(vx_queue_h q, uint64_t packed,
                                uint64_t scales, uint64_t out,
                                float tensor_scale, uint32_t n, uint32_t k,
                                uint32_t group);

// act: m*k fp16 storage; out m*n f32.
vx_nvfp4_status vx_nvfp4_gemm(vx_queue_h q, uint64_t act, uint64_t packed,
                              uint64_t scales, uint64_t out,
                              float tensor_scale, uint32_t m, uint32_t n,
                              uint32_t k, uint32_t group);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // VORTEX_DL_NVFP4_H
