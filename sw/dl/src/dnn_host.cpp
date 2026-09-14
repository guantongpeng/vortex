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

// Vortex DL dnn host layer (plan P3-02 second tier). Mirrors blas/prim/
// quant: loads dnn.vxbin, packs device-width arg blocks, launches with
// canonical dims.

#include <vortex/dnn.h>

#include <initializer_list>

#include "dnn_args.h"

namespace {

struct DnnState {
    vx_device_h dev = nullptr;
    vx_module_h module = nullptr;
    vx_kernel_h conv2d = nullptr;
    vx_kernel_h pool2d = nullptr;
    vx_kernel_h bn = nullptr;
};

DnnState g_dnn;

vx_dnn_status launch3(vx_queue_h q, vx_kernel_h k, const void* args,
                      size_t args_size, uint32_t gx, uint32_t gy, uint32_t gz,
                      uint32_t lmem) {
    vx_launch_info_t li = {};
    li.struct_size = sizeof(li);
    li.kernel = k;
    li.args_host = args;
    li.args_size = args_size;
    li.ndim = 3;
    li.grid_dim[0] = gx;
    li.grid_dim[1] = gy;
    li.grid_dim[2] = gz;
    li.block_dim[0] = 16;
    li.block_dim[1] = 1;
    li.block_dim[2] = 1;
    li.lmem_size = lmem;
    return vx_enqueue_launch(q, &li, 0, nullptr, nullptr) == VX_SUCCESS
               ? VX_DNN_OK
               : VX_DNN_ERR_LAUNCH;
}

} // namespace

vx_dnn_status vx_dnn_init(vx_device_h dev, const char* vxbin_path) {
    if (!dev || !vxbin_path) return VX_DNN_ERR_BAD_ARGS;
    if (g_dnn.module) return VX_DNN_OK;
    if (vx_module_load_file(dev, vxbin_path, &g_dnn.module) != VX_SUCCESS) {
        return VX_DNN_ERR_BAD_ARGS;
    }
    struct {
        const char* name;
        vx_kernel_h* slot;
    } entries[] = {
        {"dnn_conv2d_kernel", &g_dnn.conv2d},
        {"dnn_pool2d_kernel", &g_dnn.pool2d},
        {"dnn_bn_affine_kernel", &g_dnn.bn},
    };
    for (auto& e : entries) {
        if (vx_module_get_kernel(g_dnn.module, e.name, e.slot) != VX_SUCCESS) {
            return VX_DNN_ERR_BAD_ARGS;
        }
    }
    g_dnn.dev = dev;
    return VX_DNN_OK;
}

vx_dnn_status vx_dnn_finalize(void) {
    if (!g_dnn.module) return VX_DNN_OK;
    for (vx_kernel_h k : {g_dnn.conv2d, g_dnn.pool2d, g_dnn.bn}) {
        if (k) vx_kernel_release(k);
    }
    vx_module_release(g_dnn.module);
    g_dnn = DnnState{};
    return VX_DNN_OK;
}

vx_dnn_status vx_dnn_conv2d(vx_queue_h q,
                            uint64_t in, uint64_t weight, uint64_t bias,
                            uint64_t out,
                            uint32_t n, uint32_t ci, uint32_t hi, uint32_t wi,
                            uint32_t co, uint32_t kh, uint32_t kw,
                            uint32_t ph, uint32_t pw, uint32_t sh, uint32_t sw) {
    if (!g_dnn.module) return VX_DNN_ERR_NOT_INITIALIZED;
    if (!in || !weight || !out || n == 0 || ci == 0 || hi == 0 || wi == 0 ||
        co == 0 || kh == 0 || kw == 0 || sh == 0 || sw == 0) {
        return VX_DNN_ERR_BAD_ARGS;
    }
    if (kh * kw > 32 || ci * kh * kw * 4 > 16384) {
        return VX_DNN_ERR_UNSUPPORTED;  // LMEM staging bound (DNN_WMAX=32)
    }
    const uint32_t ho = (hi + 2 * ph - kh) / sh + 1;
    const uint32_t wo = (wi + 2 * pw - kw) / sw + 1;
    vx_dnn_conv_args_t args = {};
    args.in = (vx_dl_ptr_t)in;
    args.weight = (vx_dl_ptr_t)weight;
    args.bias = (vx_dl_ptr_t)bias;
    args.out = (vx_dl_ptr_t)out;
    args.n = n;
    args.ci = ci;
    args.hi = hi;
    args.wi = wi;
    args.co = co;
    args.ho = ho;
    args.wo = wo;
    args.kh = kh;
    args.kw = kw;
    args.ph = ph;
    args.pw = pw;
    args.sh = sh;
    args.sw = sw;
    args.has_bias = bias != 0;
    return launch3(q, g_dnn.conv2d, &args, sizeof(args),
                   ho, co, n, ci * kh * kw * 4);
}

vx_dnn_status vx_dnn_pool2d(vx_queue_h q, uint64_t in, uint64_t out,
                            uint32_t n, uint32_t c,
                            uint32_t hi, uint32_t wi,
                            uint32_t kh, uint32_t kw,
                            uint32_t ph, uint32_t pw, uint32_t sh, uint32_t sw,
                            uint32_t op) {
    if (!g_dnn.module) return VX_DNN_ERR_NOT_INITIALIZED;
    if (!in || !out || n == 0 || c == 0 || hi == 0 || wi == 0 || kh == 0 ||
        kw == 0 || sh == 0 || sw == 0 || op > 1) {
        return VX_DNN_ERR_BAD_ARGS;
    }
    const uint32_t ho = (hi + 2 * ph - kh) / sh + 1;
    const uint32_t wo = (wi + 2 * pw - kw) / sw + 1;
    vx_dnn_pool_args_t args = {};
    args.in = (vx_dl_ptr_t)in;
    args.out = (vx_dl_ptr_t)out;
    args.n = n;
    args.c = c;
    args.hi = hi;
    args.wi = wi;
    args.ho = ho;
    args.wo = wo;
    args.kh = kh;
    args.kw = kw;
    args.ph = ph;
    args.pw = pw;
    args.sh = sh;
    args.sw = sw;
    args.op = op;
    return launch3(q, g_dnn.pool2d, &args, sizeof(args), ho, c, n, 0);
}

vx_dnn_status vx_dnn_bn_affine(vx_queue_h q, uint64_t in, uint64_t mean,
                               uint64_t rstd, uint64_t weight, uint64_t bias,
                               uint64_t out, uint32_t total, uint32_t c) {
    if (!g_dnn.module) return VX_DNN_ERR_NOT_INITIALIZED;
    if (!in || !mean || !rstd || !weight || !bias || !out || total == 0 ||
        c == 0 || total % c != 0) {
        return VX_DNN_ERR_BAD_ARGS;
    }
    vx_dnn_bn_args_t args = {};
    args.in = (vx_dl_ptr_t)in;
    args.mean = (vx_dl_ptr_t)mean;
    args.rstd = (vx_dl_ptr_t)rstd;
    args.weight = (vx_dl_ptr_t)weight;
    args.bias = (vx_dl_ptr_t)bias;
    args.out = (vx_dl_ptr_t)out;
    args.total = total;
    args.c = c;
    return launch3(q, g_dnn.bn, &args, sizeof(args),
                   (total + 3) / 4, 1, 1, 0);
}
