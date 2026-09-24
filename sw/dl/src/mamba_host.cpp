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

// Vortex DL Mamba selective scan host dispatch (plan P7 precursor).
// Mirrors quant_host.cpp: loads mamba.vxbin, looks up the kernel, packs the
// device-width arg block, launches with canonical dims (unused = 1).
// Self-contained scratch copy — integration moves this to sw/dl/src/
// mamba_host.cpp with the API in include/vortex/mamba.h.

#include <vortex/mamba.h>
#include <string>

#include "mamba_args.h"

namespace {

struct MambaState {
    vx_device_h dev = nullptr;
    vx_module_h module = nullptr;
    vx_kernel_h scan = nullptr;
    vx_kernel_h state = nullptr;
    std::string path;   // what init was called with
};

MambaState g_m;

vx_mamba_status launch(vx_queue_h q, vx_kernel_h k, const void* args,
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
               ? VX_MAMBA_OK
               : VX_MAMBA_ERR_LAUNCH;
}

} // namespace

vx_mamba_status vx_mamba_init(vx_device_h dev, const char* vxbin_path) {
    if (!dev || !vxbin_path) return VX_MAMBA_ERR_BAD_ARGS;
    // Idempotent only for the same device and image. It used to return OK
    // without looking at either, so a second caller with a different image
    // silently got the first caller's kernels.
    if (g_m.module) {
        if (g_m.dev != dev || g_m.path != vxbin_path) {
            return VX_MAMBA_ERR_ALREADY_INITIALIZED;
        }
        return VX_MAMBA_OK;
    }
    if (vx_module_load_file(dev, vxbin_path, &g_m.module) != VX_SUCCESS) {
        vx_mamba_finalize();
        return VX_MAMBA_ERR_BAD_ARGS;
    }
    if (vx_module_get_kernel(g_m.module, "mamba_scan_kernel", &g_m.scan) !=
        VX_SUCCESS) {
        vx_mamba_finalize();
        return VX_MAMBA_ERR_BAD_ARGS;
    }
    if (vx_module_get_kernel(g_m.module, "mamba_state_kernel", &g_m.state) != VX_SUCCESS) {
        vx_mamba_finalize();
        return VX_MAMBA_ERR_BAD_ARGS;
    }
    g_m.dev = dev;
    g_m.path = vxbin_path;
    return VX_MAMBA_OK;
}

vx_mamba_status vx_mamba_finalize(void) {
    if (!g_m.module) return VX_MAMBA_OK;
    if (g_m.scan) vx_kernel_release(g_m.scan);
    if (g_m.state) vx_kernel_release(g_m.state);
    vx_module_release(g_m.module);
    g_m = MambaState{};
    return VX_MAMBA_OK;
}

vx_mamba_status vx_mamba_selective_scan(vx_queue_h q, uint64_t a, uint64_t dt,
                                        uint64_t b, uint64_t c, uint64_t x,
                                        uint64_t y, uint32_t batch,
                                        uint32_t channels, uint32_t seqlen,
                                        uint32_t dstate) {
    if (!g_m.module) return VX_MAMBA_ERR_NOT_INITIALIZED;
    if (!a || !dt || !b || !c || !x || !y || batch == 0 || channels == 0 ||
        seqlen == 0 || dstate == 0 || dstate > 8) {
        return VX_MAMBA_ERR_BAD_ARGS;
    }
    // (batch * channels) must fit u32 indexing used by the kernel.
    if (batch > 0xffffffffu / channels) return VX_MAMBA_ERR_BAD_ARGS;
    vx_mamba_scan_args_t args = {};
    args.a = (vx_dl_ptr_t)a;
    args.dt = (vx_dl_ptr_t)dt;
    args.b = (vx_dl_ptr_t)b;
    args.c = (vx_dl_ptr_t)c;
    args.x = (vx_dl_ptr_t)x;
    args.y = (vx_dl_ptr_t)y;
    args.batch = batch;
    args.channels = channels;
    args.seqlen = seqlen;
    args.dstate = dstate;
    const uint32_t bc_total = batch * channels;
    return launch(q, g_m.scan, &args, sizeof(args), (bc_total + 3) / 4, 16, 64);
}

vx_mamba_status vx_mamba_selective_scan_state(
    vx_queue_h q, uint64_t a, uint64_t dt, uint64_t b, uint64_t c,
    uint64_t x, uint64_t y, uint64_t initial_state, uint64_t final_state,
    uint32_t batch, uint32_t channels, uint32_t seqlen, uint32_t dstate) {
    if (!g_m.module) return VX_MAMBA_ERR_NOT_INITIALIZED;
    if (!a || !dt || !b || !c || !x || !y || !final_state || !batch || !channels ||
        !seqlen || !dstate || batch > UINT32_MAX / channels ||
        (uint64_t)batch * channels * seqlen > UINT32_MAX ||
        (uint64_t)batch * channels * dstate > UINT32_MAX ||
        (uint64_t)seqlen * dstate > UINT32_MAX) {
        return VX_MAMBA_ERR_BAD_ARGS;
    }
    vx_mamba_state_args_t args = {};
    args.a = (vx_dl_ptr_t)a;
    args.dt = (vx_dl_ptr_t)dt;
    args.b = (vx_dl_ptr_t)b;
    args.c = (vx_dl_ptr_t)c;
    args.x = (vx_dl_ptr_t)x;
    args.y = (vx_dl_ptr_t)y;
    args.initial_state = (vx_dl_ptr_t)initial_state;
    args.final_state = (vx_dl_ptr_t)final_state;
    args.batch = batch;
    args.channels = channels;
    args.seqlen = seqlen;
    args.dstate = dstate;
    return launch(q, g_m.state, &args, sizeof(args), batch * channels, 16, 64);
}
