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

// Vortex DL 2:4 structured-sparsity host layer (plan P6-02 Q5). Mirrors
// quant_host.cpp: loads the sparse24.vxbin KMU image, packs device-width
// arg blocks, launches through vx_enqueue_launch with canonical dims
// (ndim=3, unused grid/block dims = 1). Self-contained — does not touch
// libvortex_dl.

#include <vortex/sparse24.h>

#include "sparse24_args.h"

namespace {

struct Sparse24State {
    vx_device_h dev = nullptr;
    vx_module_h module = nullptr;
    vx_kernel_h prune = nullptr;
    vx_kernel_h gemm = nullptr;
};

Sparse24State g_s;

vx_sparse24_status launch(vx_queue_h q, vx_kernel_h k, const void* args,
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
               ? VX_SPARSE24_OK
               : VX_SPARSE24_ERR_LAUNCH;
}

} // namespace

vx_sparse24_status vx_sparse24_init(vx_device_h dev, const char* vxbin_path) {
    if (!dev || !vxbin_path) return VX_SPARSE24_ERR_BAD_ARGS;
    if (g_s.module) return VX_SPARSE24_OK;
    if (vx_module_load_file(dev, vxbin_path, &g_s.module) != VX_SUCCESS) {
        return VX_SPARSE24_ERR_BAD_ARGS;
    }
    struct {
        const char* name;
        vx_kernel_h* slot;
    } entries[] = {
        {"sparse24_prune_kernel", &g_s.prune},
        {"sparse24_gemm_kernel", &g_s.gemm},
    };
    for (auto& e : entries) {
        if (vx_module_get_kernel(g_s.module, e.name, e.slot) != VX_SUCCESS) {
            return VX_SPARSE24_ERR_BAD_ARGS;
        }
    }
    g_s.dev = dev;
    return VX_SPARSE24_OK;
}

vx_sparse24_status vx_sparse24_finalize(void) {
    if (!g_s.module) return VX_SPARSE24_OK;
    if (g_s.prune) vx_kernel_release(g_s.prune);
    if (g_s.gemm) vx_kernel_release(g_s.gemm);
    vx_module_release(g_s.module);
    g_s = Sparse24State{};
    return VX_SPARSE24_OK;
}

vx_sparse24_status vx_sparse24_prune(vx_queue_h q, uint64_t weights,
                                     uint64_t values, uint64_t mask,
                                     uint32_t n, uint32_t k) {
    if (!g_s.module) return VX_SPARSE24_ERR_NOT_INITIALIZED;
    if (!weights || !values || !mask || n == 0 || k == 0) {
        return VX_SPARSE24_ERR_BAD_ARGS;
    }
    vx_sparse24_prune_args_t args = {};
    args.weights = (vx_dl_ptr_t)weights;
    args.values = (vx_dl_ptr_t)values;
    args.mask = (vx_dl_ptr_t)mask;
    args.n = n;
    args.k = k;
    // One thread per (row, group-of-4); 16-thread CTAs, elementwise,
    // no local memory.
    const uint32_t total = n * ((k + 3) / 4);
    return launch(q, g_s.prune, &args, sizeof(args), (total + 15) / 16, 1,
                  16, 0);
}

vx_sparse24_status vx_sparse24_gemm(vx_queue_h q, uint64_t act,
                                    uint64_t values, uint64_t mask,
                                    uint64_t out, uint32_t m, uint32_t n,
                                    uint32_t k) {
    if (!g_s.module) return VX_SPARSE24_ERR_NOT_INITIALIZED;
    if (!act || !values || !mask || !out || m == 0 || n == 0 || k == 0) {
        return VX_SPARSE24_ERR_BAD_ARGS;
    }
    vx_sparse24_gemm_args_t args = {};
    args.act = (vx_dl_ptr_t)act;
    args.values = (vx_dl_ptr_t)values;
    args.mask = (vx_dl_ptr_t)mask;
    args.out = (vx_dl_ptr_t)out;
    args.m = m;
    args.n = n;
    args.k = k;
    // 2D grid: blockIdx.x -> N tiles, blockIdx.y -> M tiles; 16-thread
    // CTA (4 warps) over a 16x16 output tile; 1 KiB LMEM staging
    // (2 x 16 x 8 floats = 1024 bytes).
    return launch(q, g_s.gemm, &args, sizeof(args), (n + 15) / 16,
                  (m + 15) / 16, 16, 1024);
}
