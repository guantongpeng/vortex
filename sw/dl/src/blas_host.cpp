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

// Vortex DL BLAS host library (plan P3-01): loads the KMU kernel image
// and dispatches tiled GEMM over the vortex2.h ABI. No private backend
// path; every launch goes through vx_enqueue_launch.

#include <vortex/blas.h>

#include <cstring>
#include <string>

#include "blas_args.h"

namespace {

struct BLASState {
    vx_device_h dev = nullptr;
    vx_module_h module = nullptr;
    vx_kernel_h kernels[3] = {nullptr, nullptr, nullptr};
    std::string path;   // what init was called with
};

BLASState g_state;

const char* kKernelNames[3] = {
    "sgemm_f32_kernel",
    "hgemm_f16_kernel",
    "bgemm_bf16_kernel",
};

const char* kVariantNames[3] = {
    "gemm_tile16_fpu_f32",
    "gemm_tile16_fpu_f16in_f32acc",
    "gemm_tile16_fpu_bf16in_f32acc",
};

} // namespace

vx_blas_status vx_blas_init(vx_device_h dev, const char* vxbin_path) {
    if (!dev || !vxbin_path) return VX_BLAS_ERR_BAD_ARGS;
    // Idempotent only for the same device and image. It used to return OK
    // without looking at either, so a second caller with a different image
    // silently got the first caller's kernels.
    if (g_state.module) {
        if (g_state.dev != dev || g_state.path != vxbin_path) {
            return VX_BLAS_ERR_ALREADY_INITIALIZED;
        }
        return VX_BLAS_OK;
    }
    if (vx_module_load_file(dev, vxbin_path, &g_state.module) != VX_SUCCESS) {
        g_state.module = nullptr;
        return VX_BLAS_ERR_BAD_ARGS;
    }
    for (int i = 0; i < 3; ++i) {
        if (vx_module_get_kernel(g_state.module, kKernelNames[i],
                                 &g_state.kernels[i]) != VX_SUCCESS) {
            // Fail closed. Leaving the module loaded would make every later
            // init return OK while a kernel slot stays null, and a null kernel
            // is the runtime's legacy escape hatch -- the launch would then
            // succeed with PC 0 rather than failing.
            vx_blas_finalize();
            return VX_BLAS_ERR_BAD_ARGS;
        }
    }
    g_state.dev = dev;
    g_state.path = vxbin_path;
    return VX_BLAS_OK;
}

vx_blas_status vx_blas_finalize(void) {
    if (!g_state.module) return VX_BLAS_OK;
    for (int i = 0; i < 3; ++i) {
        if (g_state.kernels[i]) vx_kernel_release(g_state.kernels[i]);
    }
    vx_module_release(g_state.module);
    g_state = BLASState{};
    return VX_BLAS_OK;
}

const char* vx_blas_kernel_name(vx_blas_dtype dt) {
    if ((int)dt < 0 || (int)dt > 2) return "invalid";
    return kVariantNames[dt];
}

vx_blas_status vx_blas_gemm(vx_queue_h q,
                            vx_blas_dtype dt,
                            uint32_t M, uint32_t N, uint32_t K,
                            float alpha, float beta,
                            uint64_t A, uint64_t B, uint64_t C) {
    if (!g_state.module) return VX_BLAS_ERR_NOT_INITIALIZED;
    if ((int)dt < 0 || (int)dt > 2) return VX_BLAS_ERR_BAD_ARGS;
    if (M == 0 || N == 0 || K == 0) return VX_BLAS_ERR_BAD_ARGS;
    if (!A || !B || !C) return VX_BLAS_ERR_BAD_ARGS;

    vx_blas_gemm_args_t args = {};
    args.A = (vx_blas_ptr_t)A;
    args.B = (vx_blas_ptr_t)B;
    args.C = (vx_blas_ptr_t)C;
    args.M = M;
    args.N = N;
    args.K = K;
    args.alpha = alpha;
    args.beta = beta;

    vx_launch_info_t launch = {};
    launch.struct_size = sizeof(launch);
    launch.kernel = g_state.kernels[dt];
    launch.args_host = &args;
    launch.args_size = sizeof(args);
    launch.ndim = 3;
    launch.grid_dim[0] = (N + 15) / 16;
    launch.grid_dim[1] = (M + 15) / 16;
    launch.grid_dim[2] = 1;
    // Unused dimensions must be 1, never 0: the KMU derives the CTA shape
    // from these fields and a zero collapses it (measured: kernels never
    // write back).
    launch.block_dim[0] = 16;  // 4 warps x 4 lanes on the default profile
    launch.block_dim[1] = 1;
    launch.block_dim[2] = 1;
    launch.lmem_size = 4 * 2 * 16 * 8;  // A+B tiles staged as FP32 (1 KiB)

    if (vx_enqueue_launch(q, &launch, 0, nullptr, nullptr) != VX_SUCCESS) {
        return VX_BLAS_ERR_LAUNCH;
    }
    return VX_BLAS_OK;
}
