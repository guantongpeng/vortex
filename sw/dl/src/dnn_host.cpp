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
#include <string>
#include <cmath>

#include <initializer_list>

#include "dnn_args.h"

namespace {

struct DnnState {
    vx_device_h dev = nullptr;
    vx_module_h module = nullptr;
    vx_kernel_h conv2d = nullptr;
    vx_kernel_h pool2d = nullptr;
    vx_kernel_h bn = nullptr;
    vx_kernel_h resize = nullptr;
    std::string path;   // what init was called with
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
    // Idempotent only for the same device and image. It used to return OK
    // without looking at either, so a second caller with a different image
    // silently got the first caller's kernels.
    if (g_dnn.module) {
        if (g_dnn.dev != dev || g_dnn.path != vxbin_path) {
            return VX_DNN_ERR_ALREADY_INITIALIZED;
        }
        return VX_DNN_OK;
    }
    if (vx_module_load_file(dev, vxbin_path, &g_dnn.module) != VX_SUCCESS) {
        g_dnn.module = nullptr;
        return VX_DNN_ERR_BAD_ARGS;
    }
    struct {
        const char* name;
        vx_kernel_h* slot;
    } entries[] = {
        {"dnn_conv2d_kernel", &g_dnn.conv2d},
        {"dnn_pool2d_kernel", &g_dnn.pool2d},
        {"dnn_bn_affine_kernel", &g_dnn.bn},
        {"dnn_resize_kernel", &g_dnn.resize},
    };
    for (auto& e : entries) {
        if (vx_module_get_kernel(g_dnn.module, e.name, e.slot) != VX_SUCCESS) {
            // Fail closed. Leaving the module loaded would make every later
            // init return OK while a kernel slot stays null, and a null kernel
            // is the runtime's legacy escape hatch -- the launch would then
            // succeed with PC 0 rather than failing.
            vx_dnn_finalize();
            return VX_DNN_ERR_BAD_ARGS;
        }
    }
    g_dnn.dev = dev;
    g_dnn.path = vxbin_path;
    return VX_DNN_OK;
}

vx_dnn_status vx_dnn_finalize(void) {
    if (!g_dnn.module) return VX_DNN_OK;
    for (vx_kernel_h k : {g_dnn.conv2d, g_dnn.pool2d, g_dnn.bn, g_dnn.resize}) {
        if (k) vx_kernel_release(k);
    }
    vx_module_release(g_dnn.module);
    g_dnn = DnnState{};
    return VX_DNN_OK;
}

vx_dnn_status vx_dnn_conv2d_dilated(
    vx_queue_h q, uint64_t in, uint64_t weight, uint64_t bias, uint64_t out,
    uint32_t n, uint32_t ci, uint32_t hi, uint32_t wi, uint32_t co,
    uint32_t kh, uint32_t kw, uint32_t ph, uint32_t pw, uint32_t sh,
    uint32_t sw, uint32_t groups, uint32_t dh, uint32_t dw) {
    if (!g_dnn.module) return VX_DNN_ERR_NOT_INITIALIZED;
    if (!in || !weight || !out || n == 0 || ci == 0 || hi == 0 || wi == 0 ||
        co == 0 || kh == 0 || kw == 0 || sh == 0 || sw == 0 || groups == 0 ||
        dh == 0 || dw == 0 || ci % groups != 0 || co % groups != 0) {
        return VX_DNN_ERR_BAD_ARGS;
    }
    const uint64_t ekh = (uint64_t)(kh - 1) * dh + 1;
    const uint64_t ekw = (uint64_t)(kw - 1) * dw + 1;
    if (ekh > UINT32_MAX || ekw > UINT32_MAX ||
        (uint64_t)hi + 2ull * ph < ekh || (uint64_t)wi + 2ull * pw < ekw) {
        return VX_DNN_ERR_BAD_ARGS;
    }
    const uint64_t area = (uint64_t)kh * kw;
    if (area > UINT32_MAX || (ci / groups) > UINT32_MAX / area) {
        return VX_DNN_ERR_BAD_ARGS;
    }
    const uint64_t ho = ((uint64_t)hi + 2ull * ph - ekh) / sh + 1;
    const uint64_t wo = ((uint64_t)wi + 2ull * pw - ekw) / sw + 1;
    if (ho > UINT32_MAX || wo > UINT32_MAX - 15 ||
        ho * ((wo + 15) / 16) > UINT32_MAX) {
        return VX_DNN_ERR_BAD_ARGS;
    }
    uint64_t lmem = 0;
    if (vx_device_query(g_dnn.dev, VX_CAPS_LOCAL_MEM_SIZE, &lmem) != VX_SUCCESS ||
        lmem < VX_DNN_CONV_TILE * sizeof(float)) {
        return VX_DNN_ERR_UNSUPPORTED;
    }
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
    args.dh = dh;
    args.dw = dw;
    args.has_bias = bias != 0;
    args.groups = groups;
    return launch3(q, g_dnn.conv2d, &args, sizeof(args),
                   ho * ((wo + 15) / 16), co, n, VX_DNN_CONV_TILE * sizeof(float));
}

vx_dnn_status vx_dnn_conv2d(vx_queue_h q,
                            uint64_t in, uint64_t weight, uint64_t bias,
                            uint64_t out,
                            uint32_t n, uint32_t ci, uint32_t hi, uint32_t wi,
                            uint32_t co, uint32_t kh, uint32_t kw,
                            uint32_t ph, uint32_t pw, uint32_t sh, uint32_t sw,
                            uint32_t groups) {
    return vx_dnn_conv2d_dilated(q, in, weight, bias, out, n, ci, hi, wi, co,
                                 kh, kw, ph, pw, sh, sw, groups, 1, 1);
}

static vx_dnn_status pool2d_mode_impl(vx_queue_h q, uint64_t in, uint64_t out,
                                    uint32_t n, uint32_t c,
                                    uint32_t hi, uint32_t wi,
                                    uint32_t kh, uint32_t kw,
                                    uint32_t ph, uint32_t pw, uint32_t sh, uint32_t sw,
                                    uint32_t op, uint32_t divisor,
                                    uint32_t ceil_mode, uint32_t dh, uint32_t dw,
                                    uint64_t indices) {
    if (!g_dnn.module) return VX_DNN_ERR_NOT_INITIALIZED;
    if (!in || !out || n == 0 || c == 0 || hi == 0 || wi == 0 || kh == 0 ||
        kw == 0 || sh == 0 || sw == 0 ||
        (op != 0 && op != 1 && op != 2 && op != 3) || dh == 0 || dw == 0 ||
        (op == 0 && divisor != 0)) {
        return VX_DNN_ERR_BAD_ARGS;
    }
    const uint64_t ekh = (uint64_t)(kh - 1) * dh + 1;
    const uint64_t ekw = (uint64_t)(kw - 1) * dw + 1;
    if (ekh > UINT32_MAX || ekw > UINT32_MAX ||
        (uint64_t)hi + 2ull * ph < ekh || (uint64_t)wi + 2ull * pw < ekw) {
        return VX_DNN_ERR_BAD_ARGS;
    }
    const uint64_t nh = (uint64_t)hi + 2ull * ph - ekh;
    const uint64_t nw = (uint64_t)wi + 2ull * pw - ekw;
    const uint32_t ho = (uint32_t)((nh + (ceil_mode ? sh - 1 : 0)) / sh + 1);
    const uint32_t wo = (uint32_t)((nw + (ceil_mode ? sw - 1 : 0)) / sw + 1);
    if (ceil_mode) {
        // PyTorch drops a final window whose start is entirely in right/bottom
        // padding. Keep the correction in the host ABI rather than adding a
        // second kernel variant.
        const uint32_t hstart = (ho - 1) * sh;
        const uint32_t wstart = (wo - 1) * sw;
        const uint32_t hlimit = hi + ph;
        const uint32_t wlimit = wi + pw;
        const uint32_t adj_ho = hstart >= hlimit ? ho - 1 : ho;
        const uint32_t adj_wo = wstart >= wlimit ? wo - 1 : wo;
        if (adj_ho == 0 || adj_wo == 0) return VX_DNN_ERR_BAD_ARGS;
        vx_dnn_pool_args_t args = {};
        args.in = (vx_dl_ptr_t)in;
        args.out = (vx_dl_ptr_t)out;
        args.indices = (vx_dl_ptr_t)indices;
        args.n = n; args.c = c; args.hi = hi; args.wi = wi;
        args.ho = adj_ho; args.wo = adj_wo; args.kh = kh; args.kw = kw;
        args.ph = ph; args.pw = pw; args.sh = sh; args.sw = sw;
        args.dh = dh; args.dw = dw;
        args.op = op; args.divisor = divisor;
        return launch3(q, g_dnn.pool2d, &args, sizeof(args), adj_ho, c, n, 0);
    }
    vx_dnn_pool_args_t args = {};
    args.in = (vx_dl_ptr_t)in;
    args.out = (vx_dl_ptr_t)out;
    args.indices = (vx_dl_ptr_t)indices;
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
    args.dh = dh;
    args.dw = dw;
    args.op = op;
    args.divisor = divisor;
    return launch3(q, g_dnn.pool2d, &args, sizeof(args), ho, c, n, 0);
}

vx_dnn_status vx_dnn_pool2d_ex_mode(vx_queue_h q, uint64_t in, uint64_t out,
                                    uint32_t n, uint32_t c,
                                    uint32_t hi, uint32_t wi,
                                    uint32_t kh, uint32_t kw,
                                    uint32_t ph, uint32_t pw, uint32_t sh, uint32_t sw,
                                    uint32_t op, uint32_t divisor,
                                    uint32_t ceil_mode) {
    return pool2d_mode_impl(q, in, out, n, c, hi, wi, kh, kw, ph, pw, sh, sw,
                            op, divisor, ceil_mode, 1, 1, 0);
}

vx_dnn_status vx_dnn_pool2d_with_indices(
    vx_queue_h q, uint64_t in, uint64_t out, uint64_t indices,
    uint32_t n, uint32_t c, uint32_t hi, uint32_t wi,
    uint32_t kh, uint32_t kw, uint32_t ph, uint32_t pw,
    uint32_t sh, uint32_t sw, uint32_t ceil_mode, uint32_t dh, uint32_t dw) {
    return pool2d_mode_impl(q, in, out, n, c, hi, wi, kh, kw, ph, pw, sh, sw,
                            0, 0, ceil_mode, dh, dw, indices);
}

vx_dnn_status vx_dnn_pool2d_dilated(vx_queue_h q, uint64_t in, uint64_t out,
                                    uint32_t n, uint32_t c,
                                    uint32_t hi, uint32_t wi,
                                    uint32_t kh, uint32_t kw,
                                    uint32_t ph, uint32_t pw, uint32_t sh, uint32_t sw,
                                    uint32_t op, uint32_t divisor,
                                    uint32_t ceil_mode, uint32_t dh, uint32_t dw) {
    return pool2d_mode_impl(q, in, out, n, c, hi, wi, kh, kw, ph, pw, sh, sw,
                            op, divisor, ceil_mode, dh, dw, 0);
}

vx_dnn_status vx_dnn_pool2d_ex(vx_queue_h q, uint64_t in, uint64_t out,
                               uint32_t n, uint32_t c,
                               uint32_t hi, uint32_t wi,
                               uint32_t kh, uint32_t kw,
                               uint32_t ph, uint32_t pw, uint32_t sh, uint32_t sw,
                               uint32_t op, uint32_t divisor) {
    return vx_dnn_pool2d_ex_mode(q, in, out, n, c, hi, wi, kh, kw, ph, pw,
                                 sh, sw, op, divisor, 0);
}

vx_dnn_status vx_dnn_pool2d(vx_queue_h q, uint64_t in, uint64_t out,
                            uint32_t n, uint32_t c, uint32_t hi, uint32_t wi,
                            uint32_t kh, uint32_t kw, uint32_t ph, uint32_t pw,
                            uint32_t sh, uint32_t sw, uint32_t op) {
    return vx_dnn_pool2d_ex_mode(q, in, out, n, c, hi, wi, kh, kw, ph, pw,
                                 sh, sw, op, 0, 0);
}

vx_dnn_status vx_dnn_resize(
    vx_queue_h q, uint64_t in, uint64_t out, uint32_t n, uint32_t c,
    uint32_t ndim, const uint32_t* input_size, const uint32_t* output_size,
    const uint32_t* strides, const float* scales, uint32_t mode,
    uint32_t align_corners) {
  if (!g_dnn.module) {
    return VX_DNN_ERR_NOT_INITIALIZED;
  }
  if (!in || !out || !n || !c || ndim < 1 || ndim > 3 || mode > 3 ||
      (mode == 3 && ndim != 2) || align_corners > 1 ||
      !input_size || !output_size || !strides || !scales) {
    return VX_DNN_ERR_BAD_ARGS;
  }
  vx_dnn_resize_args_t args = {};
  args.in = (vx_dl_ptr_t)in;
  args.out = (vx_dl_ptr_t)out;
  args.n = n;
  args.c = c;
  args.ndim = ndim;
  args.mode = mode;
  args.align_corners = align_corners;
  uint64_t spatial = 1;
  for (uint32_t d = 0; d < ndim; ++d) {
    if (!input_size[d] || !output_size[d] || !std::isfinite(scales[d]) || scales[d] < 0 ||
        spatial > (UINT32_MAX - 15) / output_size[d]) {
      return VX_DNN_ERR_BAD_ARGS;
    }
    spatial *= output_size[d];
    args.input_size[d] = input_size[d];
    args.output_size[d] = output_size[d];
    args.scales[d] = scales[d];
  }
  for (uint32_t d = 0; d < ndim + 2; ++d) {
    args.strides[d] = strides[d];
  }
  return launch3(q, g_dnn.resize, &args, sizeof(args), 1, c, n, 0);
}

vx_dnn_status vx_dnn_resize_nearest2d(
    vx_queue_h q, uint64_t in, uint64_t out, uint32_t n, uint32_t c,
    uint32_t hi, uint32_t wi, uint32_t ho, uint32_t wo, uint32_t mode) {
  if (!ho || !wo || mode > 1 || (uint64_t)c * hi * wi > UINT32_MAX) {
    return VX_DNN_ERR_BAD_ARGS;
  }
  uint32_t input_size[] = {hi, wi}, output_size[] = {ho, wo};
  uint32_t strides[] = {c * hi * wi, hi * wi, wi, 1};
  float scales[] = {(float)hi / ho, (float)wi / wo};
  return vx_dnn_resize(q, in, out, n, c, 2, input_size, output_size, strides, scales, mode, 0);
}

vx_dnn_status vx_dnn_bn_affine(vx_queue_h q, uint64_t in, uint64_t mean,
                               uint64_t var, uint64_t weight, uint64_t bias,
                               uint64_t out, uint32_t n, uint32_t c,
                               uint32_t hw, float eps) {
    if (!g_dnn.module) return VX_DNN_ERR_NOT_INITIALIZED;
    if (!in || !mean || !var || !out || n == 0 || c == 0 || hw == 0) {
        return VX_DNN_ERR_BAD_ARGS;
    }
    // weight and bias are optional, but only as a pair: one without the other
    // is a caller mistake rather than a way to ask for no affine.
    if ((weight == 0) != (bias == 0)) return VX_DNN_ERR_BAD_ARGS;
    // hw is not derivable from total and c, which is the whole point of
    // passing it; check the product instead of trusting it.
    const uint32_t total = n * c * hw;
    if (n != 0 && total / n / c != hw) return VX_DNN_ERR_BAD_ARGS;
    vx_dnn_bn_args_t args = {};
    args.in = (vx_dl_ptr_t)in;
    args.mean = (vx_dl_ptr_t)mean;
    args.var = (vx_dl_ptr_t)var;
    args.weight = (vx_dl_ptr_t)weight;
    args.bias = (vx_dl_ptr_t)bias;
    args.out = (vx_dl_ptr_t)out;
    args.total = total;
    args.c = c;
    args.hw = hw;
    args.eps = eps;
    args.has_affine = (weight != 0) ? 1u : 0u;
    // launch3 hardcodes a 16-thread block, so the grid has to be sized for 16.
    // It was (total+3)/4 here, launching four times the CTAs the work needs.
    return launch3(q, g_dnn.bn, &args, sizeof(args),
                   (total + 15) / 16, 1, 1, 0);
}
