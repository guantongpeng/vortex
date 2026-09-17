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

#ifndef VORTEX_DL_LLM_H
#define VORTEX_DL_LLM_H

// Vortex DL LLM support-ops layer (plan P7, Llama precursor): RoPE,
// SwiGLU, embedding gather and KV-cache append over the vortex2.h ABI.
// The host layer loads the llm.vxbin KMU image and launches with
// canonical dims (unused grid/block dims = 1); all four kernels are
// grid-stride single-warp CTAs with no local memory.

#include <stdint.h>
#include <vortex2.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum vx_llm_status {
    VX_LLM_OK = 0,
    VX_LLM_ERR_NOT_INITIALIZED = 1,
    VX_LLM_ERR_BAD_ARGS = 2,
    VX_LLM_ERR_LAUNCH = 3,
    // init is idempotent only for the same device and image.
    VX_LLM_ERR_ALREADY_INITIALIZED = 4,
} vx_llm_status;

vx_llm_status vx_llm_init(vx_device_h dev, const char* vxbin_path);
vx_llm_status vx_llm_finalize(void);

// Rotary position embedding, in place on x[seq][dim] f32 (theta base
// 10000, interleaved pairs). An odd trailing column is left untouched.
vx_llm_status vx_llm_rope(vx_queue_h q, uint64_t x, uint32_t seq,
                          uint32_t dim);

// out[i] = a[i] * sigmoid(a[i]) * b[i]  (silu gate x up projection).
vx_llm_status vx_llm_swiglu(vx_queue_h q, uint64_t a, uint64_t b,
                            uint64_t out, uint32_t n);

// out[t*dim + d] = table[ids[t]*dim + d]; ids: n u32 row indices
// (caller guarantees ids[t] < vocab — not bounds-checked on device).
vx_llm_status vx_llm_embedding(vx_queue_h q, uint64_t table, uint64_t ids,
                               uint64_t out, uint32_t dim, uint32_t n);

// cache[(base + r)*dim + d] = new_k[r*dim + d] for r < new_tokens; rows
// outside [base, base+new_tokens) are untouched. Fails if
// base + new_tokens > capacity.
vx_llm_status vx_llm_kv_append(vx_queue_h q, uint64_t cache, uint64_t new_k,
                               uint32_t capacity, uint32_t base,
                               uint32_t new_tokens, uint32_t dim);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // VORTEX_DL_LLM_H
