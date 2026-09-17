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

// NVFP4 host dispatch layer (plan P6-02 Q4). Mirrors quant_host.cpp: loads
// the nvfp4.vxbin KMU image once, packs device-width arg blocks, launches
// through vx_enqueue_launch with canonical dims (ndim=3, unused dims = 1).
// Self-contained on purpose — no libvortex_dl edits; the integration moves
// this to sw/dl/src/nvfp4_host.cpp with the API in include/vortex/nvfp4.h
// (see integration.md).

#include <vortex2.h>
#include <string>

#include "nvfp4_args.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VX_NVFP4_OK = 0,
    VX_NVFP4_ERR_BAD_ARGS = 1,
    VX_NVFP4_ERR_NOT_INITIALIZED = 2,
    VX_NVFP4_ERR_LAUNCH = 3,
    VX_NVFP4_ERR_ALREADY_INITIALIZED = 4,
} vx_nvfp4_status;

vx_nvfp4_status vx_nvfp4_init(vx_device_h dev, const char* vxbin_path);
vx_nvfp4_status vx_nvfp4_finalize(void);
vx_nvfp4_status vx_nvfp4_pack(vx_queue_h q, uint64_t weights,
                              uint64_t packed, uint64_t scales,
                              float tensor_scale, uint32_t n, uint32_t k,
                              uint32_t group);
vx_nvfp4_status vx_nvfp4_unpack(vx_queue_h q, uint64_t packed,
                                uint64_t scales, uint64_t out,
                                float tensor_scale, uint32_t n, uint32_t k,
                                uint32_t group);
vx_nvfp4_status vx_nvfp4_gemm(vx_queue_h q, uint64_t act, uint64_t packed,
                              uint64_t scales, uint64_t out,
                              float tensor_scale, uint32_t m, uint32_t n,
                              uint32_t k, uint32_t group);

#ifdef __cplusplus
} // extern "C"
#endif

namespace {

struct Nvfp4State {
    vx_device_h dev = nullptr;
    vx_module_h module = nullptr;
    vx_kernel_h scales = nullptr;
    vx_kernel_h pack = nullptr;
    vx_kernel_h unpack = nullptr;
    vx_kernel_h gemm = nullptr;
    std::string path;
};

Nvfp4State g_nv;

vx_nvfp4_status launch(vx_queue_h q, vx_kernel_h k, const void* args,
                       size_t args_size, uint32_t grid_x, uint32_t block_x,
                       uint32_t lmem) {
    vx_launch_info_t li = {};
    li.struct_size = sizeof(li);
    li.kernel = k;
    li.args_host = args;
    li.args_size = args_size;
    li.ndim = 3;                 // canonical: unused grid/block dims stay 1
    li.grid_dim[0] = grid_x;
    li.grid_dim[1] = 1;
    li.grid_dim[2] = 1;
    li.block_dim[0] = block_x;
    li.block_dim[1] = 1;
    li.block_dim[2] = 1;
    li.lmem_size = lmem;
    return vx_enqueue_launch(q, &li, 0, nullptr, nullptr) == VX_SUCCESS
               ? VX_NVFP4_OK
               : VX_NVFP4_ERR_LAUNCH;
}

vx_nvfp4_status launch2d(vx_queue_h q, vx_kernel_h k, const void* args,
                         size_t args_size, uint32_t grid_x, uint32_t grid_y,
                         uint32_t block_x, uint32_t lmem) {
    vx_launch_info_t li = {};
    li.struct_size = sizeof(li);
    li.kernel = k;
    li.args_host = args;
    li.args_size = args_size;
    li.ndim = 3;
    li.grid_dim[0] = grid_x;
    li.grid_dim[1] = grid_y;
    li.grid_dim[2] = 1;
    li.block_dim[0] = block_x;
    li.block_dim[1] = 1;
    li.block_dim[2] = 1;
    li.lmem_size = lmem;
    return vx_enqueue_launch(q, &li, 0, nullptr, nullptr) == VX_SUCCESS
               ? VX_NVFP4_OK
               : VX_NVFP4_ERR_LAUNCH;
}

} // namespace

vx_nvfp4_status vx_nvfp4_init(vx_device_h dev, const char* vxbin_path) {
    if (!dev || !vxbin_path) return VX_NVFP4_ERR_BAD_ARGS;
    // Idempotent only for the same device and image.
    if (g_nv.module) {
        if (g_nv.dev != dev || g_nv.path != vxbin_path) {
            return VX_NVFP4_ERR_ALREADY_INITIALIZED;
        }
        return VX_NVFP4_OK;
    }
    if (vx_module_load_file(dev, vxbin_path, &g_nv.module) != VX_SUCCESS) {
        vx_nvfp4_finalize();
        return VX_NVFP4_ERR_BAD_ARGS;
    }
    struct {
        const char* name;
        vx_kernel_h* slot;
    } entries[] = {
        {"nvfp4_scales_kernel", &g_nv.scales},
        {"nvfp4_pack_kernel", &g_nv.pack},
        {"nvfp4_unpack_kernel", &g_nv.unpack},
        {"nvfp4_gemm_kernel", &g_nv.gemm},
    };
    for (auto& e : entries) {
        if (vx_module_get_kernel(g_nv.module, e.name, e.slot) != VX_SUCCESS) {
            vx_nvfp4_finalize();
            return VX_NVFP4_ERR_BAD_ARGS;
        }
    }
    g_nv.dev = dev;
    g_nv.path = vxbin_path;
    return VX_NVFP4_OK;
}

vx_nvfp4_status vx_nvfp4_finalize(void) {
    if (!g_nv.module) return VX_NVFP4_OK;
    vx_kernel_h ks[] = {g_nv.scales, g_nv.pack, g_nv.unpack, g_nv.gemm};
    for (vx_kernel_h k : ks) {
        if (k) vx_kernel_release(k);
    }
    vx_module_release(g_nv.module);
    g_nv = Nvfp4State{};
    return VX_NVFP4_OK;
}

vx_nvfp4_status vx_nvfp4_pack(vx_queue_h q, uint64_t weights,
                              uint64_t packed, uint64_t scales,
                              float tensor_scale, uint32_t n, uint32_t k,
                              uint32_t group) {
    if (!g_nv.module) return VX_NVFP4_ERR_NOT_INITIALIZED;
    if (!weights || !packed || !scales || n == 0 || k == 0 || group == 0) {
        return VX_NVFP4_ERR_BAD_ARGS;
    }
    vx_nvfp4_pack_args_t args = {};
    args.weights = (vx_dl_ptr_t)weights;
    args.packed = (vx_dl_ptr_t)packed;
    args.scales = (vx_dl_ptr_t)scales;
    args.tscale = tensor_scale;
    args.n = n;
    args.k = k;
    args.group = group;
    uint32_t groups = (k + group - 1) / group;
    // scales first (pack reads them back); single-warp elementwise CTAs.
    vx_nvfp4_status st = launch(q, g_nv.scales, &args, sizeof(args),
                                (n * groups + 3) / 4, 4, 0);
    if (st != VX_NVFP4_OK) return st;
    return launch(q, g_nv.pack, &args, sizeof(args),
                  (n * ((k + 1) / 2) + 3) / 4, 4, 0);
}

vx_nvfp4_status vx_nvfp4_unpack(vx_queue_h q, uint64_t packed,
                                uint64_t scales, uint64_t out,
                                float tensor_scale, uint32_t n, uint32_t k,
                                uint32_t group) {
    if (!g_nv.module) return VX_NVFP4_ERR_NOT_INITIALIZED;
    if (!packed || !scales || !out || n == 0 || k == 0 || group == 0) {
        return VX_NVFP4_ERR_BAD_ARGS;
    }
    vx_nvfp4_unpack_args_t args = {};
    args.packed = (vx_dl_ptr_t)packed;
    args.scales = (vx_dl_ptr_t)scales;
    args.out = (vx_dl_ptr_t)out;
    args.tscale = tensor_scale;
    args.n = n;
    args.k = k;
    args.group = group;
    return launch(q, g_nv.unpack, &args, sizeof(args), (n * k + 3) / 4, 4, 0);
}

vx_nvfp4_status vx_nvfp4_gemm(vx_queue_h q, uint64_t act, uint64_t packed,
                              uint64_t scales, uint64_t out,
                              float tensor_scale, uint32_t m, uint32_t n,
                              uint32_t k, uint32_t group) {
    if (!g_nv.module) return VX_NVFP4_ERR_NOT_INITIALIZED;
    if (!act || !packed || !scales || !out || m == 0 || n == 0 || k == 0 ||
        group == 0) {
        return VX_NVFP4_ERR_BAD_ARGS;
    }
    vx_nvfp4_gemm_args_t args = {};
    args.act = (vx_dl_ptr_t)act;
    args.packed = (vx_dl_ptr_t)packed;
    args.scales = (vx_dl_ptr_t)scales;
    args.out = (vx_dl_ptr_t)out;
    args.tscale = tensor_scale;
    args.m = m;
    args.n = n;
    args.k = k;
    args.group = group;
    // 2D grid: blockIdx.x -> N tiles, blockIdx.y -> M tiles (kernel maps
    // blockIdx.y to output rows); 16 threads = 4 warps, 1 KiB LMEM staging.
    return launch2d(q, g_nv.gemm, &args, sizeof(args), (n + 15) / 16,
                    (m + 15) / 16, 16, 1024);
}
