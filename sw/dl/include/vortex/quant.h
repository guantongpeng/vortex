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

#ifndef VORTEX_DL_QUANT_H
#define VORTEX_DL_QUANT_H

// Vortex DL quantization layer (plan P6-01): W4A16 weight-only packing
// (per-group scales) and INT8 W8A8 dynamic GEMM over the vortex2.h ABI.
//
// Metadata conventions (see quant_args.h for the exact layouts):
//   W4: signed [-8,7] nibbles, low nibble = even k; scales FP32 per
//       (row, group-of-K), scale = amax/7 (all-zero groups use 1.0).
//   W8A8: int8 activation (per-tensor scale) x int8 weight (per-row scale).

#include <stdint.h>
#include <vortex2.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum vx_quant_status {
    VX_QUANT_OK = 0,
    VX_QUANT_ERR_NOT_INITIALIZED = 1,
    VX_QUANT_ERR_BAD_ARGS = 2,
    VX_QUANT_ERR_LAUNCH = 3,
} vx_quant_status;

vx_quant_status vx_quant_init(vx_device_h dev, const char* vxbin_path);
vx_quant_status vx_quant_finalize(void);

// W4A16 -------------------------------------------------------------------

// weights: N*K f32 device buffer; packed: N*((K+1)/2) bytes; scales:
// N*ceil(K/group) f32. Two launches (scales then nibbles) — both async.
vx_quant_status vx_quant_pack_w4(vx_queue_h q, uint64_t weights,
                                 uint64_t packed, uint64_t scales,
                                 uint32_t n, uint32_t k, uint32_t group);

vx_quant_status vx_quant_unpack_w4(vx_queue_h q, uint64_t packed,
                                   uint64_t scales, uint64_t out,
                                   uint32_t n, uint32_t k, uint32_t group);

// act: M*K vx_fp16_t; out: M*N f32 = act x dequant(W)^T.
vx_quant_status vx_quant_gemm_w4a16(vx_queue_h q, uint64_t act,
                                    uint64_t packed, uint64_t scales,
                                    uint64_t out, uint32_t m, uint32_t n,
                                    uint32_t k, uint32_t group);

// INT8 W8A8 ---------------------------------------------------------------

// act: M*K int8 with per-tensor scale act_scale; weights: N*K int8 with
// per-row scales w_scale (N f32); out: M*N f32.
vx_quant_status vx_quant_gemm_w8a8(vx_queue_h q, uint64_t act,
                                   uint64_t weights, uint64_t w_scale,
                                   float act_scale, uint64_t out,
                                   uint32_t m, uint32_t n, uint32_t k);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // VORTEX_DL_QUANT_H
