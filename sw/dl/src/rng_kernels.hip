// Vortex DL RNG kernels (plan P3 first tier): Philox4x32-10 counter-based
// RNG -> uniform f32 in [0,1). KMU image compiled by hipcc-vortex with
// --kernel-lib=vortex2.
//
// Mapping: one warp (4-thread CTA on the warp=4 config; the kernel uses no
// cross-lane state so any warp size is fine) owns one 4-output Philox
// block. Every lane runs the full 10-round permutation of its block's
// counter (integer work only) and stores output word[lane]; the last block
// masks lanes whose global index reaches past n. No LMEM, no atomics, no
// float atomics. Arg fields are read directly through arg-> (VOLT -O3
// cached-arg-block miscompile workaround); only derived values live in
// locals.
//
// Algorithm: Random123 Philox4x32-10 (Salmon et al., SC'11):
//   round(ctr, key):
//     lo0/hi0 = 0xD2511F53 * ctr[0]   (32x32 -> 64-bit mulhilo)
//     lo1/hi1 = 0xCD9E8D57 * ctr[2]
//     out = { hi1^ctr[1]^key[0], lo1, hi0^ctr[3]^key[1], lo0 }
//   bumpkey: key[0] += 0x9E3779B9, key[1] += 0xBB67AE85
//   10 rounds = round, then 9x (bumpkey, round).
// Validated against the published vector Philox4x32-10(ctr=0, key=0) =
// {0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8} (see test.cpp).

#include <hip/hip_runtime.h>

#include "rng_args.h"

__global__ void rng_philox4x32_uniform_kernel(vx_rng_uniform_args_t* arg) {
    const uint32_t lane = threadIdx.x;
    // Grid-stride over 4-output blocks; unused-dim=1 and lmem=0 are the
    // host's job (canonical launch descriptor).
    for (uint32_t b = blockIdx.x; b * 4 < arg->n; b += gridDim.x) {
        const uint64_t ctr64 =
            (((uint64_t)arg->offset_hi << 32) | (uint64_t)arg->offset_lo) +
            (uint64_t)b;
        uint32_t c0 = (uint32_t)ctr64;
        uint32_t c1 = (uint32_t)(ctr64 >> 32);
        uint32_t c2 = 0, c3 = 0;  // words 2,3 reserved (stream id)
        uint32_t k0 = arg->key0;
        uint32_t k1 = arg->key1;
        for (uint32_t r = 0; r < 10; ++r) {
            if (r > 0) {
                k0 += 0x9E3779B9u;  // PHILOX_W32_0
                k1 += 0xBB67AE85u;  // PHILOX_W32_1
            }
            const uint64_t p0 = 0xD2511F53ull * (uint64_t)c0;  // PHILOX_M4x32_0
            const uint64_t p1 = 0xCD9E8D57ull * (uint64_t)c2;  // PHILOX_M4x32_1
            const uint32_t hi0 = (uint32_t)(p0 >> 32);
            const uint32_t lo0 = (uint32_t)p0;
            const uint32_t hi1 = (uint32_t)(p1 >> 32);
            const uint32_t lo1 = (uint32_t)p1;
            const uint32_t n0 = hi1 ^ c1 ^ k0;
            const uint32_t n1 = lo1;
            const uint32_t n2 = hi0 ^ c3 ^ k1;
            const uint32_t n3 = lo0;
            c0 = n0; c1 = n1; c2 = n2; c3 = n3;
        }
        const uint32_t w = lane == 0 ? c0 : lane == 1 ? c1 : lane == 2 ? c2 : c3;
        const uint32_t idx = b * 4 + lane;
        if (idx < arg->n) {
            // 24-bit round-down uniform: exact fp32 ops, value in [0, 1).
            ((float*)(uintptr_t)arg->out)[idx] = (float)(w >> 8) * (1.0f / 16777216.0f);
        }
    }
}
