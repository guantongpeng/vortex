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

#ifndef VORTEX_DL_DTYPES_H
#define VORTEX_DL_DTYPES_H

// FP16 / BF16 storage types with bit-level conversions to/from FP32.
//
// Usable from both host code and device kernels (pure integer arithmetic,
// no compiler-rt half helpers — those hang on the current simx — and no
// _Float16, which the toolchain promotes through the same broken helpers).
// This mirrors how GPU half APIs work: storage type + explicit convert.
// GEMM kernels convert to FP32 at load time and accumulate in FP32.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint16_t vx_fp16_t;  // IEEE 754 binary16, storage only
typedef uint16_t vx_bf16_t;  // bfloat16 (IEEE 754 binary32 truncated), storage only

// ---- fp16 -> f32 (exact) -------------------------------------------------

static inline float vx_fp16_to_f32(vx_fp16_t h) {
    union {
        uint32_t u;
        float f;
    } out;
    uint32_t sign = ((uint32_t)(h & 0x8000)) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t man = h & 0x3FF;
    if (exp == 0) {
        if (man == 0) {
            out.u = sign;  // +/- 0
        } else {
            // subnormal half: value = man * 2^-24; normalize
            uint32_t e = 0;
            uint32_t m = man;
            do {
                m <<= 1;
                ++e;
            } while (!(m & 0x400));
            // 1.(m & 0x3FF) x 2^(-14 - e) -> biased exponent 113 - e
            out.u = sign | ((113u - e) << 23) | ((m & 0x3FF) << 13);
        }
    } else if (exp == 0x1F) {
        out.u = sign | 0x7F800000u | (man << 13);  // inf / NaN
    } else {
        out.u = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    return out.f;
}

// ---- f32 -> fp16 (round to nearest even) ---------------------------------

static inline vx_fp16_t vx_f32_to_fp16(float f) {
    union {
        uint32_t u;
        float f;
    } v;
    v.f = f;
    uint32_t x = v.u;
    uint32_t sign = (x >> 16) & 0x8000;
    uint32_t biased = (x >> 23) & 0xFF;
    uint32_t man = x & 0x7FFFFF;
    if (biased == 0xFF) {
        return (vx_fp16_t)(sign | 0x7C00 | (man ? (0x200 | (man >> 13)) : 0));
    }
    int32_t exp = (int32_t)biased - 127 + 15;
    if (exp >= 0x1F) {
        return (vx_fp16_t)(sign | 0x7C00);  // overflow -> inf
    }
    if (exp <= 0) {
        if (exp < -10) {
            return (vx_fp16_t)sign;  // underflow (|x| < half of smallest subnormal)
        }
        // subnormal half: step 2^-24
        uint32_t man24 = man | 0x800000;
        uint32_t shift = (uint32_t)(14 - exp);  // 14..24
        uint32_t half = man24 >> shift;
        uint32_t rem = man24 & ((1u << shift) - 1);
        uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half & 1))) {
            ++half;  // may carry into 0x400 == smallest normal, same encoding
        }
        return (vx_fp16_t)(sign | half);
    }
    uint32_t rounded = (man + 0x0FFF + ((man >> 13) & 1)) >> 13;
    if (rounded & 0x400) {  // mantissa carried into exponent
        ++exp;
        rounded = 0;
        if (exp >= 0x1F) {
            return (vx_fp16_t)(sign | 0x7C00);
        }
    }
    return (vx_fp16_t)(sign | ((uint32_t)exp << 10) | rounded);
}

// ---- bf16 <-> f32 (exact both ways; f32->bf16 truncates) ------------------

static inline float vx_bf16_to_f32(vx_bf16_t b) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = ((uint32_t)b) << 16;
    return v.f;
}

static inline vx_bf16_t vx_f32_to_bf16(float f) {
    union {
        uint32_t u;
        float f;
    } v;
    v.f = f;
    return (vx_bf16_t)(v.u >> 16);  // truncate; RNE is a TCU-path decision
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // VORTEX_DL_DTYPES_H
