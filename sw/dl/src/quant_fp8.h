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

#ifndef VORTEX_DL_QUANT_FP8_H
#define VORTEX_DL_QUANT_FP8_H

// FP8 (E4M3 / E5M2) storage formats with exact bit-level conversions
// (plan P6-02 Q3, software path). Same header-only philosophy as
// vortex/dtypes.h: storage type + explicit convert, no _Float8 compiler
// types, accumulate in FP32. The hardware TCU FP8 path (VX_TCU_DTYPE_FP8)
// plugs in behind the same ABI when a TCU-enabled build exists.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint8_t vx_fp8e4m3_t;  // 1s 4e 3m, bias 7,  no inf, nan = 0x7f/0xff
typedef uint8_t vx_fp8e5m2_t;  // 1s 5e 2m, bias 15, IEEE inf/nan

// ---- f32 -> e4m3 (round to nearest, saturate to +-448) ------------------

static inline vx_fp8e4m3_t vx_f32_to_e4m3(float f) {
    union { uint32_t u; float f; } v;
    v.f = f;
    uint32_t x = v.u;
    uint32_t sign = (x >> 24) & 0x80;
    uint32_t biased = (x >> 23) & 0xff;
    uint32_t man = x & 0x7fffff;

    if (biased == 0xff) return (vx_fp8e4m3_t)(sign | 0x7f);  // inf/nan -> nan
    int32_t exp = (int32_t)biased - 127 + 7;                  // fp8 bias
    if (exp >= 16) return (vx_fp8e4m3_t)(sign | 0x7e);        // saturate to 448
    if (exp <= 0) {
        if (exp < -3) return (vx_fp8e4m3_t)sign;              // underflow -> 0
        // subnormal: value = man(24b incl. implicit) * 2^(exp-3-... )
        uint32_t man24 = man | 0x800000;
        uint32_t shift = (uint32_t)(4 - exp);                  // 4..7
        uint32_t q = man24 >> shift;
        uint32_t rem = man24 & ((1u << shift) - 1);
        uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (q & 1))) ++q;
        if (q > 0x7) q = 0x8;                                  // carry to min normal
        if (q & 0x8) {                                         // becomes normal
            return (vx_fp8e4m3_t)(sign | (1u << 3) | 0);       // exp=1, man=0
        }
        return (vx_fp8e4m3_t)(sign | q);
    }
    // normal: 3 mantissa bits, RNE on the lower 20
    uint32_t m = man >> 20;
    uint32_t rem = man & 0xfffff;
    if (rem > 0x80000 || (rem == 0x80000 && (m & 1))) {
        ++m;
        if (m > 0x7) { m = 0; ++exp; }                         // carry
    }
    if (exp >= 16) return (vx_fp8e4m3_t)(sign | 0x7e);        // saturate
    return (vx_fp8e4m3_t)(sign | ((uint32_t)exp << 3) | m);
}

// ---- e4m3 -> f32 (exact) --------------------------------------------------

static inline float vx_e4m3_to_f32(vx_fp8e4m3_t h) {
    union { uint32_t u; float f; } out;
    uint32_t sign = ((uint32_t)(h & 0x80)) << 24;
    uint32_t exp = (h >> 3) & 0xf;
    uint32_t man = h & 0x7;
    if (exp == 0xf && man == 0x7) {
        out.u = sign | 0x7fc00000;                            // nan
    } else if (exp == 0) {
        if (man == 0) {
            out.u = sign;                                     // +-0
        } else {
            uint32_t e = 0, m = man;
            do { m <<= 1; ++e; } while (!(m & 0x8));
            out.u = sign | ((127u - 7u - e + 1u) << 23) | ((m & 0x7) << 20);
        }
    } else {
        out.u = sign | ((exp - 7 + 127) << 23) | (man << 20);
    }
    return out.f;
}

// ---- f32 -> e5m2 / e5m2 -> f32 (IEEE semantics) ---------------------------

static inline vx_fp8e5m2_t vx_f32_to_e5m2(float f) {
    union { uint32_t u; float f; } v;
    v.f = f;
    uint32_t x = v.u;
    uint32_t sign = (x >> 24) & 0x80;
    uint32_t biased = (x >> 23) & 0xff;
    uint32_t man = x & 0x7fffff;

    if (biased == 0xff) {
        // inf or nan
        return (vx_fp8e5m2_t)(sign | 0x7c | (man ? 0x2 : 0x0));
    }
    int32_t exp = (int32_t)biased - 127 + 15;
    if (exp >= 31) return (vx_fp8e5m2_t)(sign | 0x7c);        // inf
    if (exp <= 0) {
        if (exp < -2) return (vx_fp8e5m2_t)sign;
        uint32_t man24 = man | 0x800000;
        uint32_t shift = (uint32_t)(3 - exp);                  // 3..5
        uint32_t q = man24 >> shift;
        uint32_t rem = man24 & ((1u << shift) - 1);
        uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (q & 1))) ++q;
        if (q > 0x3) { q = 0; exp = 1; }
        else if (q & 0x0) { /* subnormal stays */ }
        if (exp == 1 && q == 0) {}                             // carried below
        return (vx_fp8e5m2_t)(sign | q);                       // exp=0, subnormal
    }
    uint32_t m = man >> 21;
    uint32_t rem = man & 0x1fffff;
    if (rem > 0x100000 || (rem == 0x100000 && (m & 1))) {
        ++m;
        if (m > 0x3) { m = 0; ++exp; }
    }
    if (exp >= 31) return (vx_fp8e5m2_t)(sign | 0x7c);
    return (vx_fp8e5m2_t)(sign | ((uint32_t)exp << 2) | m);
}

static inline float vx_e5m2_to_f32(vx_fp8e5m2_t h) {
    union { uint32_t u; float f; } out;
    uint32_t sign = ((uint32_t)(h & 0x80)) << 24;
    uint32_t exp = (h >> 2) & 0x1f;
    uint32_t man = h & 0x3;
    if (exp == 0x1f) {
        out.u = sign | 0x7f800000 | (man ? (0x400000 | (man << 21)) : 0);
    } else if (exp == 0) {
        if (man == 0) {
            out.u = sign;
        } else {
            uint32_t e = 0, m = man;
            do { m <<= 1; ++e; } while (!(m & 0x4));
            out.u = sign | ((127u - 15u - e + 1u) << 23) | ((m & 0x3) << 21);
        }
    } else {
        out.u = sign | ((exp - 15 + 127) << 23) | (man << 21);
    }
    return out.f;
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // VORTEX_DL_QUANT_FP8_H
