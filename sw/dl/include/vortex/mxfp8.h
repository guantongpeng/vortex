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

#ifndef VORTEX_DL_MXFP8_H
#define VORTEX_DL_MXFP8_H

// MXFP8 (microscaling) software path (plan P6-02 Q4): e4m3 codes with
// per-group-of-K E8M0 power-of-two scales; value = code * 2^(e8m0-127).
// The hardware TCU MX variant gates on VX_TCU_DTYPE_MXFP8.

#include <stdint.h>
#include <vortex2.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VX_MXFP8_OK = 0,
    VX_MXFP8_ERR_NOT_INITIALIZED = 1,
    VX_MXFP8_ERR_BAD_ARGS = 2,
    VX_MXFP8_ERR_LAUNCH = 3,
} vx_mxfp8_status;

vx_mxfp8_status vx_mxfp8_init(vx_device_h dev, const char* vxbin_path);
vx_mxfp8_status vx_mxfp8_finalize(void);

// codes: rows*cols bytes; scales: rows*ceil(cols/group) bytes (E8M0).
vx_mxfp8_status vx_mxfp8_pack(vx_queue_h q, uint64_t src, uint64_t codes,
                              uint64_t scales, uint32_t rows, uint32_t cols,
                              uint32_t group);
vx_mxfp8_status vx_mxfp8_unpack(vx_queue_h q, uint64_t codes, uint64_t scales,
                                uint64_t out, uint32_t rows, uint32_t cols,
                                uint32_t group);

// act/w both MXFP8; out m*n f32 = dequant(act) x dequant(w)^T.
vx_mxfp8_status vx_mxfp8_gemm(vx_queue_h q, uint64_t act_codes,
                              uint64_t act_scales, uint64_t w_codes,
                              uint64_t w_scales, uint64_t out, uint32_t m,
                              uint32_t n, uint32_t k, uint32_t group);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // VORTEX_DL_MXFP8_H
