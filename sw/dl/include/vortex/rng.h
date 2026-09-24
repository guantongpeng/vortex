#ifndef VORTEX_DL_RNG_H
#define VORTEX_DL_RNG_H

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
    VX_RNG_ERR_ALREADY_INITIALIZED = 4,
} vx_rng_status;

vx_rng_status vx_rng_init(vx_device_h dev, const char* vxbin_path);
vx_rng_status vx_rng_finalize(void);
vx_rng_status vx_rng_uniform_f32(vx_queue_h q, uint64_t out, uint32_t n,
                                 uint64_t seed, uint64_t offset);

#ifdef __cplusplus
}
#endif

#endif
