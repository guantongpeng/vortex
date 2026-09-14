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

#ifndef VORTEX_DL_QUANT_ARGS_H
#define VORTEX_DL_QUANT_ARGS_H

// Kernel argument blocks for the quant kernels (plan P6-01).
// Same conventions as blas_args.h / prim_args.h.

#include <stdint.h>

#if defined(__VORTEX__)
typedef uintptr_t vx_dl_ptr_t;
#elif defined(VX_CFG_XLEN) && VX_CFG_XLEN == 32
typedef uint32_t vx_dl_ptr_t;
#else
typedef uint64_t vx_dl_ptr_t;
#endif

// Pack FP32 weights to int4 nibbles with per-group (along K) FP32 scales.
// weights: N x K row-major FP32 (the "W" of W4A16).
// packed:  N x ceil(K/2) bytes, low nibble = even k, high nibble = odd k,
//          signed [-8,7], little-endian nibble order.
// scales:  N x ceil(K/group) FP32, scale = amax/7 within each group
//          (zero group -> scale 1.0 to avoid div-by-zero).
typedef struct {
    vx_dl_ptr_t weights;
    vx_dl_ptr_t packed;
    vx_dl_ptr_t scales;
    uint32_t n;      // rows (output channels)
    uint32_t k;      // columns (reduction dim)
    uint32_t group;  // group size along K
} vx_quant_pack4_args_t;

// Dequantize int4 + scales back to FP32 (reference/fallback path and the
// pack->unpack bit-exactness test runs through it).
typedef struct {
    vx_dl_ptr_t packed;
    vx_dl_ptr_t scales;
    vx_dl_ptr_t out;   // N x K FP32
    uint32_t n;
    uint32_t k;
    uint32_t group;
} vx_quant_unpack4_args_t;

// W4A16 GEMM: C[N][M] = act[M][K](fp16 storage, f32 accumulate) x W[N][K]^T
// decompressed on the fly. act is M x K row-major FP16; W packed as above;
// C is M x N row-major FP32.
typedef struct {
    vx_dl_ptr_t act;     // M x K, vx_fp16_t
    vx_dl_ptr_t packed;  // N x ceil(K/2) int4
    vx_dl_ptr_t scales;  // N x ceil(K/group) f32
    vx_dl_ptr_t out;     // M x N f32
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t group;
} vx_quant_gemm_w4a16_args_t;

// INT8 W8A8 GEMM: act M x K int8 (per-tensor scale), W packed N x K int8
// (bytes, per-row scale). C = (a_q . w_q^T) * act_scale * w_scale[r].
typedef struct {
    vx_dl_ptr_t act;      // M x K int8
    vx_dl_ptr_t weights;  // N x K int8
    vx_dl_ptr_t w_scale;  // N f32 (per output channel)
    float act_scale;      // per-tensor activation scale
    uint32_t pad;
    vx_dl_ptr_t out;      // M x N f32
    uint32_t m;
    uint32_t n;
    uint32_t k;
} vx_quant_gemm_w8a8_args_t;

// FP8 W8A8 GEMM (plan P6-02 Q3 software path): act M x K and weights
// N x K stored as e4m3 (mode 0) or e5m2 (mode 1), FP32 accumulate, out
// M x N FP32. Per-tensor scales applied by the caller (dequant on host
// or a prim kernel); the kernel multiplies raw codes.
typedef struct {
    vx_dl_ptr_t act;
    vx_dl_ptr_t weights;
    vx_dl_ptr_t out;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t mode;
} vx_quant_gemm_fp8_args_t;

#endif // VORTEX_DL_QUANT_ARGS_H
