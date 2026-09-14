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

// Vortex DL LLM support kernels (plan P7): RoPE, SwiGLU, embedding
// gather, KV-cache append. All four are elementwise / row-copy kernels:
// grid-stride loops over single-warp (4-thread) CTAs, no __local_mem,
// no atomics, no cross-thread data sharing. Arg-block fields are read
// directly through arg-> at every use — nothing from the arg block is
// ever cached into a local (the VOLT -O3 cached-arg miscompile
// workaround).

#include <hip/hip_runtime.h>
#include <math.h>

#include "llm_args.h"

// ---------------------------------------------------------------------------
// Rotary position embedding (in place, theta base 1e4, interleaved pairs)
// ---------------------------------------------------------------------------

__global__ void llm_rope_kernel(vx_llm_rope_args_t* arg) {
    // One thread owns one (position, pair) rotation; both halves of the
    // pair are read and written by the same thread, so updating x in
    // place is race-free.
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < arg->seq * (arg->dim / 2); idx += blockDim.x * gridDim.x) {
        const uint32_t p = idx / (arg->dim / 2);
        const uint32_t i = idx % (arg->dim / 2);
        const float freq =
            powf(10000.0f, (-2.0f * (float)i) / (float)arg->dim);
        const float angle = (float)p * freq;
        const float c = cosf(angle);
        const float s = sinf(angle);
        const size_t base = (size_t)p * arg->dim + 2u * i;
        const float x0 =
            ((const float*)(uintptr_t)arg->x)[base];
        const float x1 =
            ((const float*)(uintptr_t)arg->x)[base + 1];
        ((float*)(uintptr_t)arg->x)[base] = x0 * c - x1 * s;
        ((float*)(uintptr_t)arg->x)[base + 1] = x0 * s + x1 * c;
    }
}

// ---------------------------------------------------------------------------
// SwiGLU: silu gate times up projection
// ---------------------------------------------------------------------------

__global__ void llm_swiglu_kernel(vx_llm_swiglu_args_t* arg) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < arg->n;
         i += blockDim.x * gridDim.x) {
        const float x = ((const float*)(uintptr_t)arg->a)[i];
        const float y = ((const float*)(uintptr_t)arg->b)[i];
        // silu, same expression as prim VX_PRIM_SILU
        const float gate = x / (1.0f + expf(-x));
        ((float*)(uintptr_t)arg->out)[i] = gate * y;
    }
}

// ---------------------------------------------------------------------------
// Embedding gather: f32 rows selected by u32 ids
// ---------------------------------------------------------------------------

__global__ void llm_embedding_kernel(vx_llm_embedding_args_t* arg) {
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < arg->n * arg->dim; idx += blockDim.x * gridDim.x) {
        const uint32_t t = idx / arg->dim;
        const uint32_t d = idx % arg->dim;
        const uint32_t id = ((const uint32_t*)(uintptr_t)arg->ids)[t];
        ((float*)(uintptr_t)arg->out)[idx] =
            ((const float*)(uintptr_t)arg->table)[(size_t)id * arg->dim + d];
    }
}

// ---------------------------------------------------------------------------
// KV-cache append: copy new K rows into the cache at a base offset
// ---------------------------------------------------------------------------

__global__ void llm_kv_append_kernel(vx_llm_kv_append_args_t* arg) {
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < arg->new_tokens * arg->dim; idx += blockDim.x * gridDim.x) {
        const uint32_t r = idx / arg->dim;
        const uint32_t d = idx % arg->dim;
        ((float*)(uintptr_t)arg->cache)[(size_t)(arg->base + r) * arg->dim +
                                        d] =
            ((const float*)(uintptr_t)arg->new_k)[idx];
    }
}
