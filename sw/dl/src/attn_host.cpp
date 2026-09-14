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

// Vortex DL attention host layer (plan P3 second tier). Mirrors
// quant_host.cpp: loads the attn.vxbin KMU image, packs a device-width
// arg block, launches one 16-thread CTA per output row with canonical
// dims (unused = 1). Self-contained — links only the read-only vortex
// runtime, no libvortex_dl dependency.

#include <vortex/attn.h>

#include <math.h>

#include "attn_args.h"

namespace {

struct AttnState {
    vx_device_h dev = nullptr;
    vx_module_h module = nullptr;
    vx_kernel_h forward = nullptr;
};

AttnState g_a;

vx_attn_status launch_forward(vx_queue_h q, uint64_t qmat, uint64_t kmat,
                              uint64_t vmat, uint64_t out, uint32_t b,
                              uint32_t h, uint32_t l, uint32_t d,
                              uint32_t causal, float scale) {
    vx_attn_forward_args_t args = {};
    args.q = (vx_dl_ptr_t)qmat;
    args.k = (vx_dl_ptr_t)kmat;
    args.v = (vx_dl_ptr_t)vmat;
    args.o = (vx_dl_ptr_t)out;
    args.b = b;
    args.h = h;
    args.l = l;
    args.d = d;
    args.causal = causal;
    args.scale = scale;

    vx_launch_info_t li = {};
    li.struct_size = sizeof(li);
    li.kernel = g_a.forward;
    li.args_host = &args;
    li.args_size = sizeof(args);  // must equal VXKMDATA args_size
    li.ndim = 3;
    li.grid_dim[0] = b * h * l;   // one CTA per output row
    li.grid_dim[1] = 1;
    li.grid_dim[2] = 1;
    li.block_dim[0] = ATTN_BLOCK;
    li.block_dim[1] = 1;
    li.block_dim[2] = 1;
    li.lmem_size = ATTN_LMEM_BYTES;  // score staging + reduction slots
    return vx_enqueue_launch(q, &li, 0, nullptr, nullptr) == VX_SUCCESS
               ? VX_ATTN_OK
               : VX_ATTN_ERR_LAUNCH;
}

vx_attn_status check_forward_args(uint64_t qmat, uint64_t kmat, uint64_t vmat,
                                  uint64_t out, uint32_t b, uint32_t h,
                                  uint32_t l, uint32_t d, uint32_t causal,
                                  double scale) {
    if (!g_a.module) return VX_ATTN_ERR_NOT_INITIALIZED;
    if (!qmat || !kmat || !vmat || !out) return VX_ATTN_ERR_BAD_ARGS;
    if (b == 0 || h == 0 || l == 0 || d == 0) return VX_ATTN_ERR_BAD_ARGS;
    if (l > ATTN_MAX_SEQ) return VX_ATTN_ERR_UNSUPPORTED;
    if (causal > 1) return VX_ATTN_ERR_BAD_ARGS;
    if (!(scale > 0.0)) return VX_ATTN_ERR_BAD_ARGS;
    return VX_ATTN_OK;
}

} // namespace

vx_attn_status vx_attn_init(vx_device_h dev, const char* vxbin_path) {
    if (!dev || !vxbin_path) return VX_ATTN_ERR_BAD_ARGS;
    if (g_a.module) return VX_ATTN_OK;
    if (vx_module_load_file(dev, vxbin_path, &g_a.module) != VX_SUCCESS) {
        return VX_ATTN_ERR_BAD_ARGS;
    }
    if (vx_module_get_kernel(g_a.module, "attn_forward_kernel",
                             &g_a.forward) != VX_SUCCESS) {
        return VX_ATTN_ERR_BAD_ARGS;
    }
    g_a.dev = dev;
    return VX_ATTN_OK;
}

vx_attn_status vx_attn_finalize(void) {
    if (!g_a.module) return VX_ATTN_OK;
    if (g_a.forward) vx_kernel_release(g_a.forward);
    vx_module_release(g_a.module);
    g_a = AttnState{};
    return VX_ATTN_OK;
}

vx_attn_status vx_attn_forward(vx_queue_h q, uint64_t qmat, uint64_t kmat,
                               uint64_t vmat, uint64_t out, uint32_t b,
                               uint32_t h, uint32_t l, uint32_t d,
                               uint32_t causal) {
    // 1/sqrt(D) in double, cast once to the f32 the kernel consumes — the
    // double-precision test reference mirrors this exact value.
    const float scale = (float)(1.0 / sqrt((double)d));
    vx_attn_status st = check_forward_args(qmat, kmat, vmat, out, b, h, l, d,
                                           causal, (double)scale);
    if (st != VX_ATTN_OK) return st;
    return launch_forward(q, qmat, kmat, vmat, out, b, h, l, d, causal, scale);
}

vx_attn_status vx_attn_forward_scaled(vx_queue_h q, uint64_t qmat,
                                      uint64_t kmat, uint64_t vmat,
                                      uint64_t out, uint32_t b, uint32_t h,
                                      uint32_t l, uint32_t d, uint32_t causal,
                                      double scale) {
    vx_attn_status st = check_forward_args(qmat, kmat, vmat, out, b, h, l, d,
                                           causal, scale);
    if (st != VX_ATTN_OK) return st;
    return launch_forward(q, qmat, kmat, vmat, out, b, h, l, d, causal,
                          (float)scale);
}
