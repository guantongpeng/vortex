// VX_MXFP8 shared arg structs + MX format definition (plan P6-02 Q4).
//
// MXFP8 data-layout metadata (documented here once; kernels, host dispatch
// and the CPU reference all obey exactly this):
//
//   group size : GROUP = 32 elements along the K (column) axis, per row.
//   codes      : rows x cols bytes of E4M3 codes (1s 4e 3m, bias 7, max
//                normal 448, NaN 0x7f/0xff; conversions in quant_fp8.h).
//   scales     : rows x ceil(cols/GROUP) bytes of E8M0 scale exponents.
//                scale byte X encodes the power-of-two 2^(X-127) (bias 127,
//                no mantissa bits). X = 127 (scale 1.0) for an all-zero
//                group; X is clamped to [0,254] (0xff is the E8M0 NaN code).
//                X = 127 + ceil(log2(amax/448)), i.e. the smallest scale
//                power-of-two s with amax <= 448*s, so the group max maps to
//                at most the E4M3 max-normal magnitude 448.
//   element value = E4M3(code) * 2^(X-127)   (exact in FP32: the scale is a
//                power of two, so scaling never rounds).
//   Quantization of element w in a group with scale byte X:
//                code = f32_to_e4m3(w * 2^(127-X))   (RNE, saturate 448).
//
// Rounding-error bound (used by the dequant acceptance test): RNE of E4M3
// has half-ULP <= 16 at the top binade [256,448] and the pack scale keeps
// |w|/s <= 448, so |dequant(w) - w| <= 16*s exactly; for data whose group
// max stays <= 256*s the bound tightens to 8*s.
//
// Arg-block ABI (same conventions as quant_args.h): device-width pointers
// first, then u32 dims; on rv64 (8-byte align):
//   pack/dequant : 3 ptr + 3 u32 = 24 + 12 = 36 -> sizeof 40
//   gemm         : 5 ptr + 4 u32 = 40 + 16 = 56 -> sizeof 56
// VXKMDATA args_size MUST equal these sizeofs (the 8-byte tail padding of
// the pack/dequant struct is real: 40, not 36).

#ifndef VX_MXFP8_ARGS_H
#define VX_MXFP8_ARGS_H

#include <stdint.h>

#include "quant_fp8.h"  // bit-level e4m3 <-> f32 (sw/dl/src, read-only)

#if defined(__VORTEX__)
typedef uintptr_t vx_dl_ptr_t;
#elif defined(VX_CFG_XLEN) && VX_CFG_XLEN == 32
typedef uint32_t vx_dl_ptr_t;
#else
typedef uint64_t vx_dl_ptr_t;
#endif

// ---------------------------------------------------------------------------
// Kernel argument blocks
// ---------------------------------------------------------------------------

// Pack pass, FP32 -> MXFP8. Two launches share this struct (scales first,
// then codes; the codes kernel reads the scales from device memory):
//   src:    rows x cols FP32 row-major
//   codes:  rows x cols E4M3 bytes
//   scales: rows x ceil(cols/group) E8M0 bytes
typedef struct {
    vx_dl_ptr_t src;
    vx_dl_ptr_t codes;
    vx_dl_ptr_t scales;
    uint32_t rows;
    uint32_t cols;
    uint32_t group;
} vx_mxfp8_pack_args_t;

// Dequant pass, MXFP8 -> FP32 (codes/scales as above, out: rows x cols FP32).
typedef struct {
    vx_dl_ptr_t codes;
    vx_dl_ptr_t scales;
    vx_dl_ptr_t out;
    uint32_t rows;
    uint32_t cols;
    uint32_t group;
} vx_mxfp8_dequant_args_t;

// MXFP8 GEMM: out[M][N] f32 = A[M][K] x W[N][K]^T with both operands MXFP8
// (E4M3 codes + per-row E8M0 group scales along K). Decoding happens at
// LMEM staging time; accumulation is FP32. 16x16 output tile, 4-warp CTA.
typedef struct {
    vx_dl_ptr_t act_codes;    // M x K E4M3
    vx_dl_ptr_t act_scales;   // M x ceil(K/group) E8M0
    vx_dl_ptr_t w_codes;      // N x K E4M3
    vx_dl_ptr_t w_scales;     // N x ceil(K/group) E8M0
    vx_dl_ptr_t out;          // M x N f32
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t group;
} vx_mxfp8_gemm_args_t;

// The VXKMDATA args_size values must equal these sizeofs (see the ABI
// comment above): rv64 40/40/56, rv32 24/24/36.
static_assert(sizeof(vx_mxfp8_pack_args_t) ==
                  (sizeof(vx_dl_ptr_t) == 8 ? 40 : 24),
              "pack args_size mismatch");
static_assert(sizeof(vx_mxfp8_dequant_args_t) ==
                  (sizeof(vx_dl_ptr_t) == 8 ? 40 : 24),
              "dequant args_size mismatch");
static_assert(sizeof(vx_mxfp8_gemm_args_t) ==
                  (sizeof(vx_dl_ptr_t) == 8 ? 56 : 36),
              "gemm args_size mismatch");

// ---------------------------------------------------------------------------
// Bit-exact shared helpers (device kernels and the host CPU reference call
// the SAME code, so pack bit-exactness is by construction). Integer-only
// math: no libm, no float rounding anywhere except the RNE inside
// vx_f32_to_e4m3 itself.
// ---------------------------------------------------------------------------

// Smallest scale exponent X with 448 * 2^(X-127) >= amax (amax >= 0),
// i.e. X = 127 + ceil(log2(amax/448)), clamped to [0,254]. amax == 0
// (all-zero group) -> 127 = identity scale. Exact for every FP32 input
// including subnormals (normalized with a shift loop) and infinities
// (clamped to 254).
static inline uint8_t vx_mxfp8_e8m0(float amax) {
    union { uint32_t u; float f; } v;
    v.f = amax;
    uint32_t bits = v.u & 0x7fffffffu;  // fabs
    uint32_t be = (bits >> 23) & 0xffu;
    uint32_t frac = bits & 0x7fffffu;
    int32_t c;
    if (be == 0) {
        if (frac == 0) return 127;          // amax == 0
        // subnormal: amax = frac * 2^-149; smallest c with 448*2^(c+149)>=frac
        uint32_t t = 448;
        c = -149;
        while (t < frac) {
            t <<= 1;
            ++c;
        }
    } else {
        // amax = 1.frac * 2^(be-127) and 448*2^c = 1.75 * 2^(c+8):
        // c = be-127-8 is enough iff 1.frac <= 1.75 (frac <= 0x600000).
        c = (int32_t)be - 127 - 8 + (frac > 0x600000u ? 1 : 0);
    }
    int32_t x = 127 + c;
    if (x < 0) x = 0;
    if (x > 254) x = 254;
    return (uint8_t)x;
}

// Exact x * 2^p (p in [-2^31, 2^31)): normal results are exact because the
// scale is a power of two; overflow clamps to +-FLT_MAX (so the E4M3
// conversion saturates instead of producing a NaN), underflow truncates
// toward zero in the subnormal range. +-0 / inf / NaN pass through.
// Representation: value = m * 2^e with the 24-bit m in [2^23, 2^24)
// (implicit bit folded in), so a normal result has biased exponent e+150
// and a subnormal result is m >> (-149 - e).
static inline float vx_mxfp8_mul_pow2(float x, int32_t p) {
    union { uint32_t u; float f; } v;
    v.f = x;
    uint32_t sign = v.u & 0x80000000u;
    uint32_t be = (v.u >> 23) & 0xffu;
    uint32_t frac = v.u & 0x7fffffu;
    if ((be == 0 && frac == 0) || be == 0xffu) return x;
    uint32_t m;
    int32_t e;
    if (be == 0) {                    // subnormal input: normalize mantissa
        m = frac;
        int32_t s = 0;
        while (!(m & 0x800000u)) {
            m <<= 1;
            ++s;
        }
        e = -149 - s;                 // frac*2^-149 == (frac<<s)*2^(-149-s)
    } else {
        m = frac | 0x800000u;
        e = (int32_t)be - 150;        // value = m * 2^e
    }
    e += p;                           // value = m * 2^e, m in [2^23, 2^24)
    if (e >= 105) {                   // m*2^e >= 2^128: clamp to FLT_MAX
        v.u = sign | 0x7f7fffffu;
        return v.f;
    }
    if (e >= -149) {                  // normal result, exact
        v.u = sign | ((uint32_t)(e + 150) << 23) | (m & 0x7fffffu);
        return v.f;
    }
    int32_t shift = -149 - e;         // subnormal result (truncating)
    if (shift >= 24) {
        v.u = sign;
        return v.f;
    }
    v.u = sign | (m >> shift);
    return v.f;
}

// Dequant one element: E4M3 code scaled by its group's E8M0 byte.
static inline float vx_mxfp8_dequant(uint8_t code, uint8_t e8m0) {
    return vx_mxfp8_mul_pow2(vx_e4m3_to_f32(code), (int32_t)e8m0 - 127);
}

// Quantize one element against its group's E8M0 byte (RNE, saturate 448).
static inline uint8_t vx_mxfp8_quant(float w, uint8_t e8m0) {
    return vx_f32_to_e4m3(vx_mxfp8_mul_pow2(w, 127 - (int32_t)e8m0));
}

// ---------------------------------------------------------------------------
// Host API (implemented in host.cpp; at integration these prototypes move
// to sw/dl/include/vortex/mxfp8.h, same signatures).
// ---------------------------------------------------------------------------

// (host only — the device build has no vortex2.h ABI types)
#ifndef __VORTEX__
#ifdef __cplusplus
extern "C" {
#endif

typedef enum vx_mxfp8_status {
    VX_MXFP8_OK = 0,
    VX_MXFP8_ERR_NOT_INITIALIZED = 1,
    VX_MXFP8_ERR_BAD_ARGS = 2,
    VX_MXFP8_ERR_LAUNCH = 3,
} vx_mxfp8_status;

vx_mxfp8_status vx_mxfp8_init(vx_device_h dev, const char* vxbin_path);
vx_mxfp8_status vx_mxfp8_finalize(void);

// src: rows*cols f32 device buffer; codes: rows*cols bytes; scales:
// rows*ceil(cols/group) bytes. Two launches (scales then codes), async.
vx_mxfp8_status vx_mxfp8_pack(vx_queue_h q, uint64_t src, uint64_t codes,
                              uint64_t scales, uint32_t rows, uint32_t cols,
                              uint32_t group);

vx_mxfp8_status vx_mxfp8_unpack(vx_queue_h q, uint64_t codes, uint64_t scales,
                                 uint64_t out, uint32_t rows, uint32_t cols,
                                 uint32_t group);

// act_codes: M*K bytes, act_scales: M*ceil(K/group); w_codes/w_scales the
// same for N rows; out: M*N f32.
vx_mxfp8_status vx_mxfp8_gemm(vx_queue_h q, uint64_t act_codes,
                              uint64_t act_scales, uint64_t w_codes,
                              uint64_t w_scales, uint64_t out, uint32_t m,
                              uint32_t n, uint32_t k, uint32_t group);

#ifdef __cplusplus
} // extern "C"
#endif
#endif // !__VORTEX__

#endif // VX_MXFP8_ARGS_H
