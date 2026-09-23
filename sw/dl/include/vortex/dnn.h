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

#ifndef VORTEX_DL_DNN_H
#define VORTEX_DL_DNN_H

// Vortex DL dnn layer (plan P3-02 second tier): direct NCHW FP32
// convolution, windowed pooling and inference batch-norm over the
// vortex2.h ABI. v1 constraints (enforced): groups >= 1,
// kh*kw <= 32, ci*kh*kw*4 <= local memory (16 KiB default).

#include <stdint.h>
#include <vortex2.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum vx_dnn_status {
    VX_DNN_OK = 0,
    VX_DNN_ERR_NOT_INITIALIZED = 1,
    VX_DNN_ERR_BAD_ARGS = 2,
    VX_DNN_ERR_UNSUPPORTED = 3,
    VX_DNN_ERR_LAUNCH = 4,
    // init is idempotent only for the same device and image; this is what a
    // second caller with different arguments gets, instead of being handed
    // the first caller's kernels.
    VX_DNN_ERR_ALREADY_INITIALIZED = 5,
} vx_dnn_status;

vx_dnn_status vx_dnn_init(vx_device_h dev, const char* vxbin_path);
vx_dnn_status vx_dnn_finalize(void);

// out[n][co][ho][wo]; ho = (hi + 2*ph - kh)/sh + 1, wo likewise.
// weight [co][ci][kh][kw], bias [co] or NULL (0 address).
vx_dnn_status vx_dnn_conv2d(vx_queue_h q,
                            uint64_t in, uint64_t weight, uint64_t bias,
                            uint64_t out,
                            uint32_t n, uint32_t ci, uint32_t hi, uint32_t wi,
                            uint32_t co, uint32_t kh, uint32_t kw,
                            uint32_t ph, uint32_t pw, uint32_t sh, uint32_t sw,
                            uint32_t groups);

// Dilation-aware entry point. The legacy spelling above is a compatibility
// wrapper for dh=dw=1; new callers use this entry point.
vx_dnn_status vx_dnn_conv2d_dilated(vx_queue_h q,
                                    uint64_t in, uint64_t weight, uint64_t bias,
                                    uint64_t out,
                                    uint32_t n, uint32_t ci, uint32_t hi, uint32_t wi,
                                    uint32_t co, uint32_t kh, uint32_t kw,
                                    uint32_t ph, uint32_t pw, uint32_t sh, uint32_t sw,
                                    uint32_t groups, uint32_t dh, uint32_t dw);

// op 0 = max, 1 = avg (count_include_pad = false),
// op 2 = avg (count_include_pad = true).
vx_dnn_status vx_dnn_pool2d(vx_queue_h q, uint64_t in, uint64_t out,
                            uint32_t n, uint32_t c,
                            uint32_t hi, uint32_t wi,
                            uint32_t kh, uint32_t kw,
                            uint32_t ph, uint32_t pw, uint32_t sh, uint32_t sw,
                            uint32_t op);

// Inference batch norm, fused affine. The kernel computes
// rstd = 1/sqrt(var + eps) itself, so the caller passes the running variance
// rather than making a host round-trip to take a square root per channel.
//
// `hw` is the per-channel spatial span H*W and is required, not derived:
// total/c gives N*H*W, which is the right span only when N == 1.
vx_dnn_status vx_dnn_bn_affine(vx_queue_h q, uint64_t in, uint64_t mean,
                               uint64_t var, uint64_t weight, uint64_t bias,
                               uint64_t out, uint32_t n, uint32_t c,
                               uint32_t hw, float eps);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // VORTEX_DL_DNN_H
