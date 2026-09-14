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

#ifndef VORTEX_DL_NVFP4_ARGS_H
#define VORTEX_DL_NVFP4_ARGS_H

// NVFP4 software path (plan P6-02 Q4) — format definition, kernel argument
// blocks, and exact bit-level conversions. Usable from host code and device
// kernels alike (pure integer arithmetic + IEEE f32 mul/div; no compiler
// _Float4/_Float16/_Float8 types, no compiler-rt helpers — those hang simx).
//
// ============================ DATA LAYOUT ==================================
//
// Weights W (N rows x K cols, row major) are stored as:
//
//   packed : N x ceil(K/2) bytes. Two FP4 E2M1 codes per byte, little-endian
//            nibble order: even column k = low nibble (bits 0..3), odd
//            column k = high nibble (bits 4..7). Row stride ceil(K/2) bytes.
//   scales : N x ceil(K/group) bytes of E4M3 (group = 16 in the tests; the
//            last group of a row may be partial when K % group != 0).
//   tscale : one f32 per tensor (a kernel scalar argument, not a buffer).
//
// Dequantized value:  v = e2m1_code * e4m3_scale * tscale.
//
// E2M1 (1 sign, 2 exp bits bias 1, 1 mantissa bit) — the 16 codes:
//   mag code: 0 0000 -> 0.0        4 0100 -> 2.0
//             1 0001 -> 0.5        5 0101 -> 3.0
//             2 0010 -> 1.0        6 0110 -> 4.0
//             3 0011 -> 1.5        7 0111 -> 6.0
//   codes 8..15 mirror 0..7 negated. No inf/NaN encodings.
//
// Quantization recipe (pack kernels; host reference mirrors op-for-op):
//   per group g of row r:      amax = max |w| (f32 compare)
//   s      = amax > 0 ? amax / 6.0f : 1.0f ;   s = s / tscale
//   scales[r][g] = E4M3(s)                       (RNE, see below)
//   E      = e4m3_to_f32(scales[r][g])           (exact)
//   q      = w / tscale ;  q = q / E             (two f32 divisions)
//   code   = f32_to_e2m1(q)                      (RNE among the 16 values)
// The two-level scale must keep s inside the E4M3 normal range
// [2^-6, 448]: choose tscale a power of two with 2^-6 <= amax/(6*tscale)
// for the tensor amax (the standard NVFP4 recipe). s below 2^-10
// underflows the scale to 0 (documented; the tests choose sane tscale).
//
// ======================= ROUNDING / TIE BEHAVIOR ==========================
//
// f32 -> E2M1: round-to-nearest with ties-to-even among the 16 representable
// magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}. "Even" = the code whose LSB
// (mantissa bit) is 0. Exact ties (midpoints .25, .75, 1.25, 1.75, 2.5, 3.5,
// 5 are all f32-representable) therefore resolve to 0, 1, 1, 2, 2, 4, 4.
// |x| >= 7.5 rounds up to 6 (saturation); values below 0.25 round to 0.
// +/-Inf and NaN saturate to +/-6 / +6 by sign bit only (documented).
//
// f32 -> E4M3 (scale encoding): RNE ties-to-even over the E4M3 grid,
// subnormals included (spacing 2^-9, min normal 2^-6), saturation to
// +/-448 (code 0x7e/0xfe); NaN/Inf map to 0x7f/0xff. NOTE: the shared
// header sw/dl/src/quant_fp8.h has a broken subnormal path (wrong shift,
// collapses [2^-10, 2^-6) onto the min normal) and returns the NaN code
// for magnitudes in (432, 512); the conversions below are self-contained
// and verified against an exhaustive brute-force reference in the test.
//
// ===================== ARG-BLOCK LAYOUT METADATA ==========================
//
// Device-width pointers (8 B on rv64/lp64), natural alignment. The VXKMDATA
// args_size MUST equal the real C sizeof of each struct:
//   vx_nvfp4_pack_args_t   : 3 ptr + 1 f32 + 3 u32 = 24 + 16      -> 40
//   vx_nvfp4_unpack_args_t : 3 ptr + 1 f32 + 3 u32 = 24 + 16      -> 40
//   vx_nvfp4_gemm_args_t   : 4 ptr + 1 f32 + 4 u32 = 32 + 20 = 52
//                            -> sizeof 56 (padded to align 8: 52 -> 56)
// (rv64 structs containing pointers are 8-byte aligned; the gemm struct's
// 4 trailing scalar bytes pad sizeof from 52 to 56. rv32: 28 / 28 / 36.)

#include <stdint.h>

#if defined(__VORTEX__)
typedef uintptr_t vx_dl_ptr_t;
#elif defined(VX_CFG_XLEN) && VX_CFG_XLEN == 32
typedef uint32_t vx_dl_ptr_t;
#else
typedef uint64_t vx_dl_ptr_t;
#endif

#ifdef __cplusplus
#define VX_NVFP4_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define VX_NVFP4_STATIC_ASSERT(cond, msg)
#endif

// ---- E2M1 <-> f32 (exact decode; RNE encode among the 16 values) ---------

static inline float vx_e2m1_to_f32(uint8_t c) {
    union { uint32_t u; float f; } out;
    uint32_t sign = ((uint32_t)(c & 0x8)) << 28;
    uint32_t e = (c >> 1) & 0x3;
    uint32_t m = c & 0x1;
    if (e == 0) {
        out.u = sign | (m ? 0x3f000000u : 0x0u);        // +-0.5 / +-0
    } else {
        out.u = sign | ((e - 1 + 127) << 23) | (m << 22);
    }
    return out.f;
}

static inline uint8_t vx_f32_to_e2m1(float x) {
    union { uint32_t u; float f; } v;
    v.f = x;
    uint32_t sign = (v.u >> 28) & 0x8;
    uint32_t b = (v.u >> 23) & 0xff;
    uint32_t man = v.u & 0x7fffff;
    if (b == 0xff) return (uint8_t)(sign | 0x7);        // inf/nan -> +-6
    int32_t e = (int32_t)b - 127 + 1;                    // e2m1 bias 1
    if (e >= 4) return (uint8_t)(sign | 0x7);            // |x| >= 8 -> 6
    if (e <= 0) {
        if (e < -1) return (uint8_t)sign;                // |x| < 0.25 -> 0
        // x in [0.25, 1): grid step 0.5 = subnormal code; RNE on q = round(2x)
        uint32_t man24 = man | 0x800000;
        uint32_t shift = (uint32_t)(23 - e);             // e=-1..0 -> 24, 23
        uint32_t q = man24 >> shift;
        uint32_t rem = man24 & ((1u << shift) - 1);
        uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (q & 1))) ++q;
        if (q & 0x2) return (uint8_t)(sign | 0x2);       // carry -> 1.0
        return (uint8_t)(sign | q);                      // 0 or 0.5
    }
    // x in [1, 8): 1 mantissa bit, RNE on the low 22 f32 mantissa bits
    uint32_t m = man >> 22;
    uint32_t rem = man & 0x3fffff;
    if (rem > 0x200000 || (rem == 0x200000 && (m & 1))) {
        ++m;
        if (m > 1) { m = 0; ++e; }                       // carry
    }
    if (e >= 4) return (uint8_t)(sign | 0x7);            // rounded >= 7.5 -> 6
    return (uint8_t)(sign | ((uint32_t)e << 1) | m);
}

// ---- E4M3 <-> f32 (exact decode; RNE encode, subnormals + saturation) ----
// Self-contained: independent of sw/dl/src/quant_fp8.h (see note above).

static inline float vx_nvfp4_e4m3_to_f32(uint8_t h) {
    union { uint32_t u; float f; } out;
    uint32_t sign = ((uint32_t)(h & 0x80)) << 24;
    uint32_t e = (h >> 3) & 0xf;
    uint32_t m = h & 0x7;
    if (e == 0xf && m == 0x7) {
        out.u = sign | 0x7fc00000;                       // nan
    } else if (e == 0) {
        if (m == 0) {
            out.u = sign;                                // +-0
        } else {
            uint32_t mm = m, e2 = 0;
            do { mm <<= 1; ++e2; } while (!(mm & 0x8));
            out.u = sign | ((121u - e2) << 23) | ((mm & 0x7) << 20);
        }
    } else {
        out.u = sign | ((e - 7 + 127) << 23) | (m << 20);
    }
    return out.f;
}

static inline uint8_t vx_nvfp4_f32_to_e4m3(float f) {
    union { uint32_t u; float f; } v;
    v.f = f;
    uint32_t x = v.u;
    uint32_t sign = (x >> 24) & 0x80;
    uint32_t b = (x >> 23) & 0xff;
    uint32_t man = x & 0x7fffff;
    if (b == 0xff) return (uint8_t)(sign | 0x7f);        // inf/nan -> nan code
    int32_t e = (int32_t)b - 127 + 7;                    // e4m3 bias 7
    if (e >= 16) return (uint8_t)(sign | 0x7e);          // |x| >= 512 -> 448
    if (e <= 0) {
        if (e < -3) return (uint8_t)sign;                // |x| < 2^-10 -> 0
        // x in [2^-10, 2^-6): subnormal grid, step 2^-9; RNE on q = round(x*2^9)
        uint32_t man24 = man | 0x800000;
        uint32_t shift = (uint32_t)(21 - e);             // e=-3..0 -> 24..21
        uint32_t q = man24 >> shift;
        uint32_t rem = man24 & ((1u << shift) - 1);
        uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (q & 1))) ++q;
        if (q & 0x8) return (uint8_t)(sign | 0x08);      // carry -> min normal
        return (uint8_t)(sign | q);
    }
    // x in [2^-6, 512): 3 mantissa bits, RNE on the low 20 f32 mantissa bits
    uint32_t m = man >> 20;
    uint32_t rem = man & 0xfffff;
    if (rem > 0x80000 || (rem == 0x80000 && (m & 1))) {
        ++m;
        if (m > 7) { m = 0; ++e; }                       // carry
    }
    // exp 15 + man 7 is the NaN code: saturate just below it (448)
    if (e >= 16 || (e == 15 && m == 7)) return (uint8_t)(sign | 0x7e);
    return (uint8_t)(sign | ((uint32_t)e << 3) | m);
}

// ---- Kernel argument blocks ----------------------------------------------

// Pack f32 weights -> E2M1 nibbles + per-group E4M3 scales (see layout and
// recipe at the top of this file). scales is written by nvfp4_scales_kernel
// first; nvfp4_pack_kernel reads it back (two launches, same arg block).
typedef struct {
    vx_dl_ptr_t weights;  // N x K row-major f32 (in)
    vx_dl_ptr_t packed;   // N x ceil(K/2) bytes (out)
    vx_dl_ptr_t scales;   // N x ceil(K/group) E4M3 bytes (out)
    float tscale;         // per-tensor f32 scale
    uint32_t n;           // rows (output channels)
    uint32_t k;           // columns (reduction dim)
    uint32_t group;       // group size along K (16 for NVFP4)
} vx_nvfp4_pack_args_t;

// Dequantize: out[r][c] = e2m1(code) * e4m3_to_f32(scales[r][c/group]) *
// tscale, evaluated left-to-right as two f32 multiplies.
typedef struct {
    vx_dl_ptr_t packed;   // N x ceil(K/2) bytes
    vx_dl_ptr_t scales;   // N x ceil(K/group) E4M3 bytes
    vx_dl_ptr_t out;      // N x K f32
    float tscale;
    uint32_t n;
    uint32_t k;
    uint32_t group;
} vx_nvfp4_unpack_args_t;

// W4A16-style GEMM: C[M][N] (f32) = act[M][K] (fp16 storage, converted to
// f32 at staging) x W[N][K]^T decoded on the fly at staging
// (w = (e2m1 * e4m3_scale) * tscale in f32), FP32 accumulation over K in
// ascending order. Tile shape mirrors quant_gemm_w4a16_kernel: 16x16 output
// tile per 4-warp (16-thread) CTA, K in chunks of 8, LMEM staging of the
// decompressed tiles as f32 (2 x 16*8*4 = 1024 bytes -> lmem_size 1024).
typedef struct {
    vx_dl_ptr_t act;      // M x K vx_fp16_t
    vx_dl_ptr_t packed;   // N x ceil(K/2) E2M1
    vx_dl_ptr_t scales;   // N x ceil(K/group) E4M3
    vx_dl_ptr_t out;      // M x N f32
    float tscale;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t group;
} vx_nvfp4_gemm_args_t;

#if !defined(VX_CFG_XLEN) || VX_CFG_XLEN == 64
VX_NVFP4_STATIC_ASSERT(sizeof(vx_nvfp4_pack_args_t) == 40,
                       "rv64 pack args must be 40 bytes");
VX_NVFP4_STATIC_ASSERT(sizeof(vx_nvfp4_unpack_args_t) == 40,
                       "rv64 unpack args must be 40 bytes");
VX_NVFP4_STATIC_ASSERT(sizeof(vx_nvfp4_gemm_args_t) == 56,
                       "rv64 gemm args: 4 ptr + 5 scalars = 52 -> sizeof 56");
#else
VX_NVFP4_STATIC_ASSERT(sizeof(vx_nvfp4_pack_args_t) == 28,
                       "rv32 pack args: 3 ptr + 4 scalars = 28");
VX_NVFP4_STATIC_ASSERT(sizeof(vx_nvfp4_unpack_args_t) == 28,
                       "rv32 unpack args: 3 ptr + 4 scalars = 28");
VX_NVFP4_STATIC_ASSERT(sizeof(vx_nvfp4_gemm_args_t) == 36, "rv32 gemm args 36");
#endif

#endif // VORTEX_DL_NVFP4_ARGS_H
