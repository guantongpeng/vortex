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

#ifndef VORTEX_DL_RNG_H
#define VORTEX_DL_RNG_H

// Philox 4x32-10 counter-based RNG (plan P3 first tier): deterministic
// uniform fills from a (seed, offset) counter, matching Random123.

#include <stdint.h>
#include <vortex2.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum vx_rng_status {
    VX_RNG_OK = 0,
    VX_RNG_ERR_NOT_INITIALIZED = 1,
    VX_RNG_ERR_BAD_ARGS = 2,
    VX_RNG_ERR_LAUNCH = 3,
    // init is idempotent only for the same device and image.
    VX_RNG_ERR_ALREADY_INITIALIZED = 4,
} vx_rng_status;

vx_rng_status vx_rng_init(vx_device_h dev, const char* vxbin_path);
vx_rng_status vx_rng_finalize(void);

// Fill out[0..n) with uniform [0,1) f32 from Philox(counter = offset + i/4,
// key = seed). Reproducible host-side with the same algorithm.
vx_rng_status vx_rng_uniform_f32(vx_queue_h q, uint64_t out, uint32_t n,
                                 uint64_t seed, uint64_t offset);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // VORTEX_DL_RNG_H
