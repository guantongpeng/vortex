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

// torch-vortex dnn kernels (plan P5-02): the ResNet op set as KMU images.
// Same shapes as sw/dl's dnn/blas kernels (one 4-warp CTA per row/tile,
// LMEM staging), with rv64-width pointer arg blocks (torch-vortex builds
// against an rv64 tree; see the package README).

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

// matmul lives in the DL library now (vx_blas_gemm).

// Argument blocks are defined once, for both the device and host compilers, in
// torch_kernel_args.h. Do not declare them locally: a second definition is
// exactly how the host/device layout drift started.
#include "torch_kernel_args.h"

// conv2d lives in the DL library now (vx_dnn_conv2d). Keeping a second
// copy here is what W3.1 exists to remove.

// pooling lives in the DL library now (vx_dnn_pool2d).

// batch norm lives in the DL library now (vx_dnn_bn_affine).

__global__ void tv_bias_add_kernel(bias_args_t* arg) {
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < arg->m * arg->n; idx += blockDim.x * gridDim.x) {
        const uint32_t j = idx % arg->n;
        ((float*)(uintptr_t)arg->out)[idx] +=
            ((const float*)(uintptr_t)arg->bias)[j];
    }
}
