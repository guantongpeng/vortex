// Philox 4x32-10 RNG kernel argument block (plan P3 first tier).
//
// Data layout metadata (must match the VXKMDATA sidecar args_size):
//   vx_dl_ptr_t out        device-width pointer (8 bytes on rv64, 4 on rv32)
//   uint32_t   offset_lo   low  word of the 64-bit counter offset
//   uint32_t   offset_hi   high word of the 64-bit counter offset
//   uint32_t   key0        Philox key word 0 (seed low  word)
//   uint32_t   key1        Philox key word 1 (seed high word)
//   uint32_t   n           number of f32 outputs to generate
//   uint32_t   pad         unused (keeps sizeof a multiple of 8 on rv64)
//
// sizeof: rv64 = 8 + 6*4 = 32 (align 8, no interior padding);
//         rv32 = 4 + 6*4 = 28 (align 4).
// The host packs the identical layout (device-width pointer).
//
// Counter convention (block = 4 consecutive f32 outputs):
//   block index b, b = 0 .. ceil(n/4)-1:
//     ctr64 = ((uint64_t)offset_hi << 32 | offset_lo) + b
//     ctr   = { (uint32_t)ctr64, (uint32_t)(ctr64 >> 32), 0, 0 }
//     key   = { key0, key1 }
//     out[b*4 + i] = (float)(w[i] >> 8) * 2^-24   for b*4+i < n,
//                    w = Philox4x32-10(ctr, key), i = 0..3.
// ctr words 2 and 3 are reserved (0) for a future stream/substream id.
//
// Uniform conversion choice: (bits >> 8) * 2^-24 truncates the low 8 bits
// (round-down) so the 24-bit integer maps bijectively onto the 24-bit fp32
// mantissa grid {0, 2^-24, ..., 1-2^-24}: every output is exact in fp32
// (int->float of a 24-bit value and scaling by a power of two are both
// exact IEEE ops), the result is strictly inside [0,1), and no rounding
// bias is introduced near 1.0 the way round-to-nearest of bits*2^-32
// would map two bit patterns onto 1.0.

#ifndef VORTEX_DL_RNG_ARGS_H
#define VORTEX_DL_RNG_ARGS_H

#include <stdint.h>

#if defined(__VORTEX__)
typedef uintptr_t vx_dl_ptr_t;
#elif defined(VX_CFG_XLEN) && VX_CFG_XLEN == 32
typedef uint32_t vx_dl_ptr_t;
#else
typedef uint64_t vx_dl_ptr_t;
#endif

// Fill out[0..n) with uniform [0,1) f32 from Philox4x32-10.
typedef struct {
    vx_dl_ptr_t out;      // n f32 values
    uint32_t offset_lo;   // counter offset, low  word (block units)
    uint32_t offset_hi;   // counter offset, high word (block units)
    uint32_t key0;        // key word 0
    uint32_t key1;        // key word 1
    uint32_t n;           // number of outputs
    uint32_t pad;
} vx_rng_uniform_args_t;

// Layout guard: the VXKMDATA sidecar args_size must equal this sizeof
// (rv64 = 32, rv32 = 28). All trailing members are u32 so no interior or
// tail padding can appear on either ABI.
#if defined(__cplusplus)
static_assert(sizeof(vx_rng_uniform_args_t) ==
                  sizeof(vx_dl_ptr_t) + 6 * sizeof(uint32_t),
              "vx_rng_uniform_args_t layout drifted from the metadata");
#endif

#endif // VORTEX_DL_RNG_ARGS_H
