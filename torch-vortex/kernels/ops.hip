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
//
// The elementwise kernels are op-code driven -- one kernel per arity, with the
// operation selected by a field -- so adding an operator is a row in
// kernels/torch_kernel_args.h rather than another kernel. That keeps the
// surface small enough to keep honest, and it keeps every elementwise kernel
// in the simple shape VOLT compiles correctly (see the note on tv_mm_kernel
// for what happens when the shape stops being simple).

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

// Argument blocks and the op-code enums are defined once, for both the device
// and host compilers, in torch_kernel_args.h. Do not declare them locally: a
// second definition is exactly how the host/device layout drift started.
#include "torch_kernel_args.h"

__device__ inline float torch_binary_apply(float a, float b, uint32_t op) {
    switch (op) {
    case TORCH_BINARY_ADD:     return a + b;
    case TORCH_BINARY_SUB:     return a - b;
    case TORCH_BINARY_MUL:     return a * b;
    case TORCH_BINARY_DIV:     return a / b;
    // torch.maximum/minimum propagate NaN rather than ignoring it, unlike
    // fmaxf/fminf. `a + b` is NaN whenever either side is.
    case TORCH_BINARY_MAXIMUM: return (isnan(a) || isnan(b)) ? (a + b) : (a > b ? a : b);
    case TORCH_BINARY_MINIMUM: return (isnan(a) || isnan(b)) ? (a + b) : (a < b ? a : b);
    default:                   return a;
    }
}

__global__ void binary_op_kernel(binary_args_t* arg) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < arg->n;
         i += blockDim.x * gridDim.x) {
        ((float*)(uintptr_t)arg->dst)[i] =
            torch_binary_apply(((const float*)(uintptr_t)arg->a)[i],
                               ((const float*)(uintptr_t)arg->b)[i], arg->op);
    }
}

// dst = a <op> value, or value <op> a when `reverse` is set (scalar - tensor).
__global__ void scalar_op_kernel(scalar_args_t* arg) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < arg->n;
         i += blockDim.x * gridDim.x) {
        const float a = ((const float*)(uintptr_t)arg->a)[i];
        ((float*)(uintptr_t)arg->dst)[i] =
            arg->reverse ? torch_binary_apply(arg->value, a, arg->op)
                         : torch_binary_apply(a, arg->value, arg->op);
    }
}

__global__ void fill_kernel(fill_args_t* arg) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < arg->n;
         i += blockDim.x * gridDim.x) {
        ((float*)(uintptr_t)arg->dst)[i] = arg->value;
    }
}

// The index representation ATen wants, from the one the DL reduce produces.
// Pure conversion: one thread per index, grid-stride like everything here.
__global__ void widen_u32_i64_kernel(widen_args_t* arg) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < arg->n;
         i += blockDim.x * gridDim.x) {
        ((int64_t*)(uintptr_t)arg->dst)[i] =
            (int64_t)((const uint32_t*)(uintptr_t)arg->src)[i];
    }
}

// Elementwise copy between two arbitrarily strided buffers. Handles the cases
// hipMemcpy cannot: a transposed view on either side, a storage offset, and a
// broadcast (stride 0). Dims are walked from the innermost out, so a
// contiguous side costs a multiply per dimension rather than an address walk.
__global__ void copy_strided_kernel(copy_strided_args_t* arg) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < arg->total;
         i += blockDim.x * gridDim.x) {
        uint32_t rem = i, dst_off = 0, src_off = 0;
        for (uint32_t d = arg->ndim; d-- > 0;) {
            const uint32_t c = rem % arg->sizes[d];
            rem /= arg->sizes[d];
            dst_off += c * arg->dst_strides[d];
            src_off += c * arg->src_strides[d];
        }
        ((float*)(uintptr_t)arg->dst)[dst_off] =
            ((const float*)(uintptr_t)arg->src)[src_off];
    }
}

// a <op> b with broadcasting. Strides of 0 are the broadcast dimensions; the
// output is contiguous, so its index is the linear one. Same decomposition as
// the copy kernel, which is the shape VOLT handles correctly.
__global__ void broadcast_op_kernel(broadcast_op_args_t* arg) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < arg->total;
         i += blockDim.x * gridDim.x) {
        uint32_t rem = i, a_off = 0, b_off = 0;
        for (uint32_t d = arg->ndim; d-- > 0;) {
            const uint32_t c = rem % arg->sizes[d];
            rem /= arg->sizes[d];
            a_off += c * arg->a_strides[d];
            b_off += c * arg->b_strides[d];
        }
        ((float*)(uintptr_t)arg->dst)[i] =
            torch_binary_apply(((const float*)(uintptr_t)arg->a)[a_off],
                               ((const float*)(uintptr_t)arg->b)[b_off], arg->op);
    }
}
