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

// torch-vortex op kernels (plan P5-01). KMU image loaded by the C++
// extension; every entry takes the single-pointer arg-block ABI.
// v1 scope: FP32, contiguous, 1-D grid-stride over n elements.

#include <hip/hip_runtime.h>
#include <stdint.h>

// Argument blocks are defined once, for both the device and host compilers, in
// torch_kernel_args.h. Do not declare them locally: a second definition is
// exactly how the host/device layout drift started.
#include "torch_kernel_args.h"

__global__ void add_kernel(binary_args_t* arg) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < arg->n;
         i += blockDim.x * gridDim.x) {
        float* dst = (float*)(uintptr_t)arg->dst;
        dst[i] = ((const float*)(uintptr_t)arg->a)[i] +
                 ((const float*)(uintptr_t)arg->b)[i];
    }
}

__global__ void mul_kernel(binary_args_t* arg) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < arg->n;
         i += blockDim.x * gridDim.x) {
        float* dst = (float*)(uintptr_t)arg->dst;
        dst[i] = ((const float*)(uintptr_t)arg->a)[i] *
                 ((const float*)(uintptr_t)arg->b)[i];
    }
}

// max(x, 0) with PyTorch's semantics: NaN propagates, -0.0 is preserved.
// `x > 0.0f ? x : 0.0f` gets both wrong (NaN becomes 0, -0.0 becomes +0.0), and
// a NaN surviving a ReLU is a diagnostic signal rather than noise to erase.
__device__ inline float relu_f32(float x) {
    return x < 0.0f ? 0.0f : x;
}

__global__ void relu_kernel(fill_args_t* arg) {
    // in-place; reuses the fill layout (dst + n, value unused)
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < arg->n;
         i += blockDim.x * gridDim.x) {
        float* dst = (float*)(uintptr_t)arg->dst;
        dst[i] = relu_f32(dst[i]);
    }
}

// Non-inplace relu: separate source and destination, so aten::relu can return
// a fresh tensor without disturbing the input.
__global__ void relu_out_kernel(unary_args_t* arg) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < arg->n;
         i += blockDim.x * gridDim.x) {
        ((float*)(uintptr_t)arg->dst)[i] =
            relu_f32(((const float*)(uintptr_t)arg->a)[i]);
    }
}

__global__ void fill_kernel(fill_args_t* arg) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < arg->n;
         i += blockDim.x * gridDim.x) {
        ((float*)(uintptr_t)arg->dst)[i] = arg->value;
    }
}
