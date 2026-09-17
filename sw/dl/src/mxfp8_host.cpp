// VX_MXFP8 host dispatch (plan P6-02 Q4). Self-contained mirror of
// quant_host.cpp: loads the mxfp8.vxbin KMU image, packs device-width arg
// blocks, launches through vx_enqueue_launch with canonical dims (unused
// dims = 1). No libvortex_dl dependency.

#include <vortex2.h>
#include <string>

#include "mxfp8_args.h"

namespace {

struct Mxfp8State {
    vx_device_h dev = nullptr;
    vx_module_h module = nullptr;
    vx_kernel_h scales = nullptr;
    vx_kernel_h pack = nullptr;
    vx_kernel_h dequant = nullptr;
    vx_kernel_h gemm = nullptr;
    std::string path;
};

Mxfp8State g_mx;

vx_mxfp8_status launch1d(vx_queue_h q, vx_kernel_h k, const void* args,
                         size_t args_size, uint32_t grid_x,
                         uint32_t block_x) {
    vx_launch_info_t li = {};
    li.struct_size = sizeof(li);
    li.kernel = k;
    li.args_host = args;
    li.args_size = args_size;
    li.ndim = 3;
    li.grid_dim[0] = grid_x;
    li.grid_dim[1] = 1;
    li.grid_dim[2] = 1;
    li.block_dim[0] = block_x;
    li.block_dim[1] = 1;
    li.block_dim[2] = 1;
    li.lmem_size = 0;
    return vx_enqueue_launch(q, &li, 0, nullptr, nullptr) == VX_SUCCESS
               ? VX_MXFP8_OK
               : VX_MXFP8_ERR_LAUNCH;
}

} // namespace

vx_mxfp8_status vx_mxfp8_init(vx_device_h dev, const char* vxbin_path) {
    if (!dev || !vxbin_path) return VX_MXFP8_ERR_BAD_ARGS;
    // Idempotent only for the same device and image. It used to return OK
    // without looking at either, so a second caller with a different image
    // silently got the first caller's kernels.
    if (g_mx.module) {
        if (g_mx.dev != dev || g_mx.path != vxbin_path) {
            return VX_MXFP8_ERR_ALREADY_INITIALIZED;
        }
        return VX_MXFP8_OK;
    }
    if (vx_module_load_file(dev, vxbin_path, &g_mx.module) != VX_SUCCESS) {
        vx_mxfp8_finalize();
        return VX_MXFP8_ERR_BAD_ARGS;
    }
    struct {
        const char* name;
        vx_kernel_h* slot;
    } entries[] = {
        {"mxfp8_scales_kernel", &g_mx.scales},
        {"mxfp8_pack_kernel", &g_mx.pack},
        {"mxfp8_dequant_kernel", &g_mx.dequant},
        {"mxfp8_gemm_kernel", &g_mx.gemm},
    };
    for (auto& e : entries) {
        if (vx_module_get_kernel(g_mx.module, e.name, e.slot) != VX_SUCCESS) {
            vx_mxfp8_finalize();
            return VX_MXFP8_ERR_BAD_ARGS;
        }
    }
    g_mx.dev = dev;
    g_mx.path = vxbin_path;
    return VX_MXFP8_OK;
}

vx_mxfp8_status vx_mxfp8_finalize(void) {
    if (!g_mx.module) return VX_MXFP8_OK;
    vx_kernel_h ks[] = {g_mx.scales, g_mx.pack, g_mx.dequant, g_mx.gemm};
    for (vx_kernel_h k : ks) {
        if (k) vx_kernel_release(k);
    }
    vx_module_release(g_mx.module);
    g_mx = Mxfp8State{};
    return VX_MXFP8_OK;
}

vx_mxfp8_status vx_mxfp8_pack(vx_queue_h q, uint64_t src, uint64_t codes,
                              uint64_t scales, uint32_t rows, uint32_t cols,
                              uint32_t group) {
    if (!g_mx.module) return VX_MXFP8_ERR_NOT_INITIALIZED;
    if (!src || !codes || !scales || rows == 0 || cols == 0 || group == 0) {
        return VX_MXFP8_ERR_BAD_ARGS;
    }
    vx_mxfp8_pack_args_t args = {};
    args.src = (vx_dl_ptr_t)src;
    args.codes = (vx_dl_ptr_t)codes;
    args.scales = (vx_dl_ptr_t)scales;
    args.rows = rows;
    args.cols = cols;
    args.group = group;
    const uint32_t groups = (cols + group - 1) / group;
    // scales pass first (the codes pass reads them); single-warp CTAs.
    vx_mxfp8_status st = launch1d(q, g_mx.scales, &args, sizeof(args),
                                  (rows * groups + 3) / 4, 4);
    if (st != VX_MXFP8_OK) return st;
    return launch1d(q, g_mx.pack, &args, sizeof(args),
                    (rows * cols + 3) / 4, 4);
}

vx_mxfp8_status vx_mxfp8_unpack(vx_queue_h q, uint64_t codes, uint64_t scales,
                                 uint64_t out, uint32_t rows, uint32_t cols,
                                 uint32_t group) {
    if (!g_mx.module) return VX_MXFP8_ERR_NOT_INITIALIZED;
    if (!codes || !scales || !out || rows == 0 || cols == 0 || group == 0) {
        return VX_MXFP8_ERR_BAD_ARGS;
    }
    vx_mxfp8_dequant_args_t args = {};
    args.codes = (vx_dl_ptr_t)codes;
    args.scales = (vx_dl_ptr_t)scales;
    args.out = (vx_dl_ptr_t)out;
    args.rows = rows;
    args.cols = cols;
    args.group = group;
    return launch1d(q, g_mx.dequant, &args, sizeof(args),
                    (rows * cols + 3) / 4, 4);
}

vx_mxfp8_status vx_mxfp8_gemm(vx_queue_h q, uint64_t act_codes,
                              uint64_t act_scales, uint64_t w_codes,
                              uint64_t w_scales, uint64_t out, uint32_t m,
                              uint32_t n, uint32_t k, uint32_t group) {
    if (!g_mx.module) return VX_MXFP8_ERR_NOT_INITIALIZED;
    if (!act_codes || !act_scales || !w_codes || !w_scales || !out || m == 0 ||
        n == 0 || k == 0 || group == 0) {
        return VX_MXFP8_ERR_BAD_ARGS;
    }
    vx_mxfp8_gemm_args_t args = {};
    args.act_codes = (vx_dl_ptr_t)act_codes;
    args.act_scales = (vx_dl_ptr_t)act_scales;
    args.w_codes = (vx_dl_ptr_t)w_codes;
    args.w_scales = (vx_dl_ptr_t)w_scales;
    args.out = (vx_dl_ptr_t)out;
    args.m = m;
    args.n = n;
    args.k = k;
    args.group = group;
    // 2D grid: blockIdx.x -> N tiles, blockIdx.y -> M tiles; 16-thread CTA
    // (4 warps) with 1 KiB LMEM staging (lA [16][8] + lB [8][16] f32).
    vx_launch_info_t li = {};
    li.struct_size = sizeof(li);
    li.kernel = g_mx.gemm;
    li.args_host = &args;
    li.args_size = sizeof(args);
    li.ndim = 3;
    li.grid_dim[0] = (n + 15) / 16;
    li.grid_dim[1] = (m + 15) / 16;
    li.grid_dim[2] = 1;
    li.block_dim[0] = 16;
    li.block_dim[1] = 1;
    li.block_dim[2] = 1;
    li.lmem_size = 1024;
    return vx_enqueue_launch(q, &li, 0, nullptr, nullptr) == VX_SUCCESS
               ? VX_MXFP8_OK
               : VX_MXFP8_ERR_LAUNCH;
}
