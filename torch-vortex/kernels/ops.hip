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
#include <vortex/dtypes.h>

// Argument blocks and the op-code enums are defined once, for both the device
// and host compilers, in torch_kernel_args.h. Do not declare them locally: a
// second definition is exactly how the host/device layout drift started.
#include "torch_kernel_args.h"

__global__ void fill_kernel(fill_args_t* arg) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < arg->n;
         i += blockDim.x * gridDim.x) {
        ((float*)(uintptr_t)arg->dst)[i] = arg->value;
    }
}

__global__ void nll_loss_kernel(nll_loss_args_t* arg) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float total = 0.0f;
    float total_weight = 0.0f;
    for (uint32_t row = 0; row < arg->rows; ++row) {
        const int64_t target = ((const int64_t*)(uintptr_t)arg->target)[row];
        if (target == arg->ignore_index) {
            if (arg->reduction == 0) ((float*)(uintptr_t)arg->dst)[row] = 0.0f;
            continue;
        }
        if (target < 0 || target >= (int64_t)arg->classes) {
            *reinterpret_cast<uint32_t*>((uintptr_t)arg->invalid) = 1;
            if (arg->reduction == 0) ((float*)(uintptr_t)arg->dst)[row] = 0.0f;
            continue;
        }
        const float w = arg->weight
                            ? ((const float*)(uintptr_t)arg->weight)[target]
                            : 1.0f;
        const float value = -((const float*)(uintptr_t)arg->input)
                                 [(size_t)row * arg->classes + target] * w;
        if (arg->reduction == 0) ((float*)(uintptr_t)arg->dst)[row] = value;
        total += value;
        total_weight += w;
    }
    if (arg->reduction == 1) {
        ((float*)(uintptr_t)arg->dst)[0] =
            total_weight == 0.0f ? NAN : total / total_weight;
    } else if (arg->reduction == 2) {
        ((float*)(uintptr_t)arg->dst)[0] = total;
    }
    ((float*)(uintptr_t)arg->total_weight)[0] = total_weight;
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

// Copy one contiguous input into a concatenation or stack output. The host
// supplies the logical extents so the kernel does not need tensor metadata.
__global__ void cat_kernel(cat_args_t* arg) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < arg->total;
         i += blockDim.x * gridDim.x) {
        uint32_t rem = i;
        const uint32_t inner = rem % arg->inner;
        rem /= arg->inner;
        const uint32_t coord = rem % arg->src_dim;
        const uint32_t outer = rem / arg->src_dim;
        const uint32_t dst_index =
            ((outer * arg->out_dim) + arg->dst_offset + coord) * arg->inner + inner;
        ((float*)(uintptr_t)arg->dst)[dst_index] =
            ((const float*)(uintptr_t)arg->src)[i];
    }
}

__device__ inline int64_t torch_index_value(index_args_t* arg, uint32_t i) {
    return arg->index_type == 0
               ? (int64_t)((const int32_t*)(uintptr_t)arg->index)[i]
               : ((const int64_t*)(uintptr_t)arg->index)[i];
}

// Gather and scatter share the same coordinate walk. The invalid flag lets
// the host report an out-of-range device index without reading the whole
// index tensor back before the operation.
__global__ void gather_kernel(index_args_t* arg) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < arg->total;
         i += blockDim.x * gridDim.x) {
        uint32_t rem = i;
        uint32_t src_off = 0;
        const int64_t index = torch_index_value(arg, i);
        if (index < 0 || index >= (int64_t)arg->dim_size) {
            *reinterpret_cast<uint32_t*>((uintptr_t)arg->invalid) = 1;
            continue;
        }
        for (uint32_t d = arg->ndim; d-- > 0;) {
            const uint32_t coord = rem % arg->sizes[d];
            rem /= arg->sizes[d];
            src_off += (d == arg->dim ? (uint32_t)index : coord) * arg->strides[d];
        }
        ((float*)(uintptr_t)arg->dst)[i] =
            ((const float*)(uintptr_t)arg->src)[src_off];
    }
}

__global__ void scatter_kernel(index_args_t* arg) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < arg->total;
         i += blockDim.x * gridDim.x) {
        uint32_t rem = i;
        uint32_t dst_off = 0;
        const int64_t index = torch_index_value(arg, i);
        if (index < 0 || index >= (int64_t)arg->dim_size) {
            *reinterpret_cast<uint32_t*>((uintptr_t)arg->invalid) = 1;
            continue;
        }
        for (uint32_t d = arg->ndim; d-- > 0;) {
            const uint32_t coord = rem % arg->sizes[d];
            rem /= arg->sizes[d];
            dst_off += (d == arg->dim ? (uint32_t)index : coord) * arg->strides[d];
        }
        ((float*)(uintptr_t)arg->dst)[dst_off] =
            ((const float*)(uintptr_t)arg->src)[i];
    }
}

__global__ void index_add_kernel(index_add_args_t* arg) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < arg->total;
         i += blockDim.x * gridDim.x) {
        uint32_t rem = i;
        uint32_t self_off = 0;
        uint32_t row = 0;
        for (uint32_t d = arg->ndim; d-- > 0;) {
            const uint32_t coord = rem % arg->sizes[d];
            rem /= arg->sizes[d];
            if (d == arg->dim) row = coord;
            self_off += coord * arg->self_strides[d];
        }

        float acc = 0.0f;
        for (uint32_t j = 0; j < arg->index_count; ++j) {
            const int64_t index =
                arg->index_type == 0
                    ? (int64_t)((const int32_t*)(uintptr_t)arg->index)[j]
                    : ((const int64_t*)(uintptr_t)arg->index)[j];
            if (index < 0 || index >= (int64_t)arg->dim_size) {
                *reinterpret_cast<uint32_t*>((uintptr_t)arg->invalid) = 1;
                continue;
            }
            if ((uint32_t)index != row) continue;

            uint32_t src_rem = i;
            uint32_t src_off = 0;
            for (uint32_t d = arg->ndim; d-- > 0;) {
                const uint32_t coord = src_rem % arg->sizes[d];
                src_rem /= arg->sizes[d];
                src_off += (d == arg->dim ? j : coord) * arg->src_strides[d];
            }
            acc += ((const float*)(uintptr_t)arg->src)[src_off];
        }
        ((float*)(uintptr_t)arg->dst)[i] =
            ((const float*)(uintptr_t)arg->self)[self_off] + arg->alpha * acc;
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
        float value;
        switch (arg->src_type) {
        case TORCH_COPY_F32:
            value = ((const float*)(uintptr_t)arg->src)[src_off];
            break;
        case TORCH_COPY_F64:
            value = (float)((const double*)(uintptr_t)arg->src)[src_off];
            break;
        case TORCH_COPY_F16:
            value = vx_fp16_to_f32(((const vx_fp16_t*)(uintptr_t)arg->src)[src_off]);
            break;
        case TORCH_COPY_BF16:
            value = vx_bf16_to_f32(((const vx_bf16_t*)(uintptr_t)arg->src)[src_off]);
            break;
        case TORCH_COPY_I32:
            value = (float)((const int32_t*)(uintptr_t)arg->src)[src_off];
            break;
        case TORCH_COPY_I64:
            value = (float)((const int64_t*)(uintptr_t)arg->src)[src_off];
            break;
        case TORCH_COPY_BOOL:
            value = (float)((const uint8_t*)(uintptr_t)arg->src)[src_off];
            break;
        default:
            value = 0.0f;
            break;
        }
        switch (arg->dst_type) {
        case TORCH_COPY_F32:
            ((float*)(uintptr_t)arg->dst)[dst_off] = value;
            break;
        case TORCH_COPY_F64:
            ((double*)(uintptr_t)arg->dst)[dst_off] = (double)value;
            break;
        case TORCH_COPY_F16:
            ((vx_fp16_t*)(uintptr_t)arg->dst)[dst_off] = vx_f32_to_fp16(value);
            break;
        case TORCH_COPY_BF16:
            ((vx_bf16_t*)(uintptr_t)arg->dst)[dst_off] = vx_f32_to_bf16(value);
            break;
        case TORCH_COPY_I32:
            ((int32_t*)(uintptr_t)arg->dst)[dst_off] = (int32_t)value;
            break;
        case TORCH_COPY_I64:
            ((int64_t*)(uintptr_t)arg->dst)[dst_off] = (int64_t)value;
            break;
        case TORCH_COPY_BOOL:
            ((uint8_t*)(uintptr_t)arg->dst)[dst_off] = value != 0.0f;
            break;
        default:
            break;
        }
    }
}
