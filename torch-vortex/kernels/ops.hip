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

struct binary_args_t {
    uintptr_t dst;
    uintptr_t a;
    uintptr_t b;
    uint32_t n;
    uint32_t pad;
};

struct fill_args_t {
    uintptr_t dst;
    uint32_t n;
    float value;
    uint32_t pad;
};

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

__global__ void fill_kernel(fill_args_t* arg) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < arg->n;
         i += blockDim.x * gridDim.x) {
        ((float*)(uintptr_t)arg->dst)[i] = arg->value;
    }
}
