// Philox 4x32-10 RNG host dispatch (plan P3 first tier). Self-contained
// layer in the style of sw/dl/src/quant_host.cpp: loads the rng.vxbin KMU
// image, packs the device-width arg block, launches with the canonical
// descriptor (ndim=3, unused dims 1, lmem 0). No dependency on
// libvortex_dl - only the read-only vortex runtime.

#include <stdint.h>
#include <vortex2.h>

#include "rng_args.h"

// --- public API (integration: move to include/vortex/rng.h) ---------------
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

// Fill out[0..n) with uniform [0,1) f32 values from Philox4x32-10 keyed by
// seed; the counter starts at offset (in 4-output blocks), so successive
// calls with the same seed and increasing offsets continue one stream.
// Async (queued on q).
vx_rng_status vx_rng_uniform_f32(vx_queue_h q, uint64_t out, uint32_t n,
                                 uint64_t seed, uint64_t offset);

#ifdef __cplusplus
}
#endif

// ---------------------------------------------------------------------------

namespace {

struct RngState {
    vx_device_h dev = nullptr;
    vx_module_h module = nullptr;
    vx_kernel_h uniform = nullptr;
};

RngState g_rng;

vx_rng_status launch_uniform(vx_queue_h q, const void* args, size_t args_size,
                             uint32_t grid_x, uint32_t block_x) {
    vx_launch_info_t li = {};
    li.struct_size = sizeof(li);
    li.kernel = g_rng.uniform;
    li.args_host = args;
    li.args_size = args_size;
    li.ndim = 3;
    li.grid_dim[0] = grid_x;  // one 4-thread warp per 4-output block
    li.grid_dim[1] = 1;
    li.grid_dim[2] = 1;
    li.block_dim[0] = block_x;
    li.block_dim[1] = 1;
    li.block_dim[2] = 1;
    li.lmem_size = 0;  // kernel uses no __local_mem()
    return vx_enqueue_launch(q, &li, 0, nullptr, nullptr) == VX_SUCCESS
               ? VX_RNG_OK
               : VX_RNG_ERR_LAUNCH;
}

} // namespace

vx_rng_status vx_rng_init(vx_device_h dev, const char* vxbin_path) {
    if (!dev || !vxbin_path) return VX_RNG_ERR_BAD_ARGS;
    if (g_rng.module) return VX_RNG_OK;
    if (vx_module_load_file(dev, vxbin_path, &g_rng.module) != VX_SUCCESS) {
        return VX_RNG_ERR_BAD_ARGS;
    }
    if (vx_module_get_kernel(g_rng.module, "rng_philox4x32_uniform_kernel",
                             &g_rng.uniform) != VX_SUCCESS) {
        return VX_RNG_ERR_BAD_ARGS;
    }
    g_rng.dev = dev;
    return VX_RNG_OK;
}

vx_rng_status vx_rng_finalize(void) {
    if (!g_rng.module) return VX_RNG_OK;
    if (g_rng.uniform) vx_kernel_release(g_rng.uniform);
    vx_module_release(g_rng.module);
    g_rng = RngState{};
    return VX_RNG_OK;
}

vx_rng_status vx_rng_uniform_f32(vx_queue_h q, uint64_t out, uint32_t n,
                                 uint64_t seed, uint64_t offset) {
    if (!g_rng.module) return VX_RNG_ERR_NOT_INITIALIZED;
    if (!out || n == 0) return VX_RNG_ERR_BAD_ARGS;
    vx_rng_uniform_args_t args = {};
    args.out = (vx_dl_ptr_t)out;
    args.offset_lo = (uint32_t)offset;
    args.offset_hi = (uint32_t)(offset >> 32);
    args.key0 = (uint32_t)seed;
    args.key1 = (uint32_t)(seed >> 32);
    args.n = n;
    args.pad = 0;
    return launch_uniform(q, &args, sizeof(args), (n + 3) / 4, 4);
}
