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

// Vortex DL quant host layer (plan P6-01). Mirrors blas/prim: loads the
// quant.vxbin KMU image, packs device-width arg blocks, launches through
// vx_enqueue_launch with canonical dims (unused = 1).

#include <vortex/quant.h>

#include "quant_args.h"

namespace {

struct QuantState {
    vx_device_h dev = nullptr;
    vx_module_h module = nullptr;
    vx_kernel_h scales4 = nullptr;
    vx_kernel_h pack4 = nullptr;
    vx_kernel_h unpack4 = nullptr;
    vx_kernel_h gemm_w4a16 = nullptr;
    vx_kernel_h gemm_w8a8 = nullptr;
};

QuantState g_q;

vx_quant_status launch(vx_queue_h q, vx_kernel_h k, const void* args,
                       size_t args_size, uint32_t grid_x, uint32_t block_x,
                       uint32_t lmem) {
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
    li.lmem_size = lmem;
    return vx_enqueue_launch(q, &li, 0, nullptr, nullptr) == VX_SUCCESS
               ? VX_QUANT_OK
               : VX_QUANT_ERR_LAUNCH;
}

} // namespace

vx_quant_status vx_quant_init(vx_device_h dev, const char* vxbin_path) {
    if (!dev || !vxbin_path) return VX_QUANT_ERR_BAD_ARGS;
    if (g_q.module) return VX_QUANT_OK;
    if (vx_module_load_file(dev, vxbin_path, &g_q.module) != VX_SUCCESS) {
        return VX_QUANT_ERR_BAD_ARGS;
    }
    struct {
        const char* name;
        vx_kernel_h* slot;
    } entries[] = {
        {"quant_scales4_kernel", &g_q.scales4},
        {"quant_pack4_kernel", &g_q.pack4},
        {"quant_unpack4_kernel", &g_q.unpack4},
        {"quant_gemm_w4a16_kernel", &g_q.gemm_w4a16},
        {"quant_gemm_w8a8_kernel", &g_q.gemm_w8a8},
    };
    for (auto& e : entries) {
        if (vx_module_get_kernel(g_q.module, e.name, e.slot) != VX_SUCCESS) {
            return VX_QUANT_ERR_BAD_ARGS;
        }
    }
    g_q.dev = dev;
    return VX_QUANT_OK;
}

vx_quant_status vx_quant_finalize(void) {
    if (!g_q.module) return VX_QUANT_OK;
    vx_kernel_h ks[] = {g_q.scales4, g_q.pack4, g_q.unpack4, g_q.gemm_w4a16,
                        g_q.gemm_w8a8};
    for (vx_kernel_h k : ks) {
        if (k) vx_kernel_release(k);
    }
    vx_module_release(g_q.module);
    g_q = QuantState{};
    return VX_QUANT_OK;
}

vx_quant_status vx_quant_pack_w4(vx_queue_h q, uint64_t weights,
                                 uint64_t packed, uint64_t scales,
                                 uint32_t n, uint32_t k, uint32_t group) {
    if (!g_q.module) return VX_QUANT_ERR_NOT_INITIALIZED;
    if (!weights || !packed || !scales || n == 0 || k == 0 ||
        group == 0 || group > k) {
        return VX_QUANT_ERR_BAD_ARGS;
    }
    vx_quant_pack4_args_t args = {};
    args.weights = (vx_dl_ptr_t)weights;
    args.packed = (vx_dl_ptr_t)packed;
    args.scales = (vx_dl_ptr_t)scales;
    args.n = n;
    args.k = k;
    args.group = group;
    uint32_t groups = (k + group - 1) / group;
    // scales first (pack reads them); single-warp CTAs, elementwise.
    vx_quant_status st = launch(q, g_q.scales4, &args, sizeof(args),
                                (n * groups + 3) / 4, 4, 0);
    if (st != VX_QUANT_OK) return st;
    return launch(q, g_q.pack4, &args, sizeof(args),
                  (n * ((k + 1) / 2) + 3) / 4, 4, 0);
}

vx_quant_status vx_quant_unpack_w4(vx_queue_h q, uint64_t packed,
                                   uint64_t scales, uint64_t out,
                                   uint32_t n, uint32_t k, uint32_t group) {
    if (!g_q.module) return VX_QUANT_ERR_NOT_INITIALIZED;
    if (!packed || !scales || !out || n == 0 || k == 0 || group == 0) {
        return VX_QUANT_ERR_BAD_ARGS;
    }
    vx_quant_unpack4_args_t args = {};
    args.packed = (vx_dl_ptr_t)packed;
    args.scales = (vx_dl_ptr_t)scales;
    args.out = (vx_dl_ptr_t)out;
    args.n = n;
    args.k = k;
    args.group = group;
    return launch(q, g_q.unpack4, &args, sizeof(args), (n * k + 3) / 4, 4, 0);
}

vx_quant_status vx_quant_gemm_w4a16(vx_queue_h q, uint64_t act,
                                    uint64_t packed, uint64_t scales,
                                    uint64_t out, uint32_t m, uint32_t n,
                                    uint32_t k, uint32_t group) {
    if (!g_q.module) return VX_QUANT_ERR_NOT_INITIALIZED;
    if (!act || !packed || !scales || !out || m == 0 || n == 0 || k == 0 ||
        group == 0) {
        return VX_QUANT_ERR_BAD_ARGS;
    }
    vx_quant_gemm_w4a16_args_t args = {};
    args.act = (vx_dl_ptr_t)act;
    args.packed = (vx_dl_ptr_t)packed;
    args.scales = (vx_dl_ptr_t)scales;
    args.out = (vx_dl_ptr_t)out;
    args.m = m;
    args.n = n;
    args.k = k;
    args.group = group;
    // 2D grid: blockIdx.x -> N tiles, blockIdx.y -> M tiles (the kernel
    // maps blockIdx.y to output rows / blockIdx.x to output columns).
    vx_launch_info_t li = {};
    li.struct_size = sizeof(li);
    li.kernel = g_q.gemm_w4a16;
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
               ? VX_QUANT_OK
               : VX_QUANT_ERR_LAUNCH;
}

vx_quant_status vx_quant_gemm_w8a8(vx_queue_h q, uint64_t act,
                                   uint64_t weights, uint64_t w_scale,
                                   float act_scale, uint64_t out,
                                   uint32_t m, uint32_t n, uint32_t k) {
    if (!g_q.module) return VX_QUANT_ERR_NOT_INITIALIZED;
    if (!act || !weights || !w_scale || !out || m == 0 || n == 0 || k == 0) {
        return VX_QUANT_ERR_BAD_ARGS;
    }
    vx_quant_gemm_w8a8_args_t args = {};
    args.act = (vx_dl_ptr_t)act;
    args.weights = (vx_dl_ptr_t)weights;
    args.w_scale = (vx_dl_ptr_t)w_scale;
    args.act_scale = act_scale;
    args.pad = 0;
    args.out = (vx_dl_ptr_t)out;
    args.m = m;
    args.n = n;
    args.k = k;
    // 2D grid: extra rows handled by the tile loop's masked staging.
    vx_launch_info_t li = {};
    li.struct_size = sizeof(li);
    li.kernel = g_q.gemm_w8a8;
    li.args_host = &args;
    li.args_size = sizeof(args);
    li.ndim = 3;
    li.grid_dim[0] = (n + 15) / 16;
    li.grid_dim[1] = (m + 15) / 16;  // W8A8 kernel maps blockIdx.y to M
    li.grid_dim[2] = 1;
    li.block_dim[0] = 16;
    li.block_dim[1] = 1;
    li.block_dim[2] = 1;
    li.lmem_size = 1024;
    return vx_enqueue_launch(q, &li, 0, nullptr, nullptr) == VX_SUCCESS
               ? VX_QUANT_OK
               : VX_QUANT_ERR_LAUNCH;
}
