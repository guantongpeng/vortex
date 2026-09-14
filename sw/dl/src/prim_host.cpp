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

// Vortex DL prim/norm host layer (plan P3-02). Mirrors blas_host.cpp:
// loads the KMU image, packs device-width arg blocks, launches through
// vx_enqueue_launch. Unused grid/block dims are always 1 (never 0 —
// see the P3-01 finding).

#include <vortex/prim.h>

#include <cstring>

#include "prim_args.h"

namespace {

struct PrimState {
    vx_device_h dev = nullptr;
    vx_module_h module = nullptr;
    vx_kernel_h unary = nullptr;
    vx_kernel_h reduce = nullptr;
    vx_kernel_h softmax = nullptr;
    vx_kernel_h layernorm = nullptr;
    vx_kernel_h rmsnorm = nullptr;
};

PrimState g_prim;

// Launch helper: canonical dims (unused = 1) + LMEM for row kernels.
vx_prim_status launch(vx_queue_h q, vx_kernel_h k, const void* args,
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
               ? VX_PRIM_OK
               : VX_PRIM_ERR_LAUNCH;
}

} // namespace

vx_prim_status vx_prim_init(vx_device_h dev, const char* vxbin_path) {
    if (!dev || !vxbin_path) return VX_PRIM_ERR_BAD_ARGS;
    if (g_prim.module) return VX_PRIM_OK;
    if (vx_module_load_file(dev, vxbin_path, &g_prim.module) != VX_SUCCESS) {
        return VX_PRIM_ERR_BAD_ARGS;
    }
    struct {
        const char* name;
        vx_kernel_h* slot;
    } entries[] = {
        {"prim_unary_kernel", &g_prim.unary},
        {"prim_reduce_kernel", &g_prim.reduce},
        {"prim_softmax_kernel", &g_prim.softmax},
        {"prim_layernorm_kernel", &g_prim.layernorm},
        {"prim_rmsnorm_kernel", &g_prim.rmsnorm},
    };
    for (auto& e : entries) {
        if (vx_module_get_kernel(g_prim.module, e.name, e.slot) != VX_SUCCESS) {
            return VX_PRIM_ERR_BAD_ARGS;
        }
    }
    g_prim.dev = dev;
    return VX_PRIM_OK;
}

vx_prim_status vx_prim_finalize(void) {
    if (!g_prim.module) return VX_PRIM_OK;
    vx_kernel_h ks[] = {g_prim.unary, g_prim.reduce, g_prim.softmax,
                        g_prim.layernorm, g_prim.rmsnorm};
    for (vx_kernel_h k : ks) {
        if (k) vx_kernel_release(k);
    }
    vx_module_release(g_prim.module);
    g_prim = PrimState{};
    return VX_PRIM_OK;
}

vx_prim_status vx_prim_unary(vx_queue_h q, vx_prim_op op,
                             uint64_t in, uint64_t out, uint32_t n) {
    if (!g_prim.module) return VX_PRIM_ERR_NOT_INITIALIZED;
    if (!in || !out || n == 0) return VX_PRIM_ERR_BAD_ARGS;
    if (op > VX_PRIM_OP_NEG) return VX_PRIM_ERR_BAD_ARGS;
    vx_prim_unary_args_t args = {};
    args.in = (vx_dl_ptr_t)in;
    args.out = (vx_dl_ptr_t)out;
    args.n = n;
    args.op = (uint32_t)op;
    // Single-warp CTAs for pure elementwise work (P2-03 constraint).
    return launch(q, g_prim.unary, &args, sizeof(args),
                  (n + 3) / 4, 4, 0);
}

vx_prim_status vx_prim_reduce(vx_queue_h q, vx_prim_op op,
                              uint64_t in, uint64_t out, uint32_t n) {
    if (!g_prim.module) return VX_PRIM_ERR_NOT_INITIALIZED;
    if (!in || !out || n == 0) return VX_PRIM_ERR_BAD_ARGS;
    if (op < VX_PRIM_OP_SUM) return VX_PRIM_ERR_BAD_ARGS;
    vx_prim_reduce_args_t args = {};
    args.in = (vx_dl_ptr_t)in;
    args.out = (vx_dl_ptr_t)out;
    args.n = n;
    args.op = (uint32_t)(op - VX_PRIM_OP_SUM);
    return launch(q, g_prim.reduce, &args, sizeof(args), 1, 16, 256);
}

vx_prim_status vx_prim_softmax(vx_queue_h q, uint64_t in, uint64_t out,
                               uint32_t rows, uint32_t cols) {
    if (!g_prim.module) return VX_PRIM_ERR_NOT_INITIALIZED;
    if (!in || !out || rows == 0 || cols == 0) return VX_PRIM_ERR_BAD_ARGS;
    vx_prim_rowargs_t args = {};
    args.in = (vx_dl_ptr_t)in;
    args.out = (vx_dl_ptr_t)out;
    args.rows = rows;
    args.cols = cols;
    return launch(q, g_prim.softmax, &args, sizeof(args), rows, 16, 256);
}

vx_prim_status vx_prim_layernorm(vx_queue_h q, uint64_t in, uint64_t gamma,
                                 uint64_t beta, uint64_t out,
                                 uint32_t rows, uint32_t cols, float eps) {
    if (!g_prim.module) return VX_PRIM_ERR_NOT_INITIALIZED;
    if (!in || !gamma || !beta || !out || rows == 0 || cols == 0) {
        return VX_PRIM_ERR_BAD_ARGS;
    }
    vx_prim_norm_args_t args = {};
    args.in = (vx_dl_ptr_t)in;
    args.gamma = (vx_dl_ptr_t)gamma;
    args.beta = (vx_dl_ptr_t)beta;
    args.out = (vx_dl_ptr_t)out;
    args.rows = rows;
    args.cols = cols;
    args.eps = eps;
    return launch(q, g_prim.layernorm, &args, sizeof(args), rows, 16, 256);
}

vx_prim_status vx_prim_rmsnorm(vx_queue_h q, uint64_t in, uint64_t gamma,
                               uint64_t out, uint32_t rows, uint32_t cols,
                               float eps) {
    if (!g_prim.module) return VX_PRIM_ERR_NOT_INITIALIZED;
    if (!in || !gamma || !out || rows == 0 || cols == 0) {
        return VX_PRIM_ERR_BAD_ARGS;
    }
    vx_prim_rmsnorm_args_t args = {};
    args.in = (vx_dl_ptr_t)in;
    args.gamma = (vx_dl_ptr_t)gamma;
    args.out = (vx_dl_ptr_t)out;
    args.rows = rows;
    args.cols = cols;
    args.eps = eps;
    return launch(q, g_prim.rmsnorm, &args, sizeof(args), rows, 16, 256);
}
