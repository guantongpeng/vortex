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

#ifndef VORTEX_DL_LLM_ARGS_H
#define VORTEX_DL_LLM_ARGS_H

// Kernel argument blocks for the LLM support kernels (plan P7:
// RoPE, SwiGLU, embedding gather, KV-cache append).
// Same conventions as quant_args.h / prim_args.h.
//
// Data-layout metadata (VXKMDATA "args_size" must equal the C sizeof of
// the struct; device pointers are device-width, so sizes are XLEN-dependent):
//
//   kernel                 fields                    rv64 sizeof   rv32 sizeof
//   llm_rope_kernel        1 ptr + 2 u32             16            12
//   llm_swiglu_kernel      3 ptr + 1 u32             32 (24+4 pad) 16
//   llm_embedding_kernel   3 ptr + 2 u32             32            20
//   llm_kv_append_kernel   2 ptr + 3 u32             32 (16+12 pad) 20
//
// rv64 structs hold 8-byte pointers, so trailing u32 runts are padded to
// the 8-byte alignment (e.g. 2 ptr + 3 u32 = 16+12 = 28 -> sizeof 32).
// rv32 (XLEN=32, 4-byte align) packs without padding. All data tensors
// are row-major FP32 unless noted; ids are uint32 row indices.

#include <stdint.h>

#if defined(__VORTEX__)
typedef uintptr_t vx_dl_ptr_t;
#elif defined(VX_CFG_XLEN) && VX_CFG_XLEN == 32
typedef uint32_t vx_dl_ptr_t;
#else
typedef uint64_t vx_dl_ptr_t;
#endif

// Rotary position embedding, in place on x[seq][dim] f32 (one head,
// flattened). For p in [0,seq), i in [0, dim/2):
//   freq = 10000^(-2i/dim), a = p*freq,
//   (x[2i], x[2i+1]) <- (x[2i]*cos a - x[2i+1]*sin a,
//                        x[2i]*sin a + x[2i+1]*cos a)
// One thread owns a whole pair, so the in-place update is race-free.
// If dim is odd the trailing column is left untouched.
typedef struct {
    vx_dl_ptr_t x;   // seq x dim f32, rotated in place
    uint32_t seq;
    uint32_t dim;
} vx_llm_rope_args_t;

// SwiGLU: out[i] = silu(a[i]) * b[i] with silu(x) = x/(1+e^-x).
// a and b are two aligned activations (gate / up projections).
typedef struct {
    vx_dl_ptr_t a;    // n f32 gate input
    vx_dl_ptr_t b;    // n f32 up input
    vx_dl_ptr_t out;  // n f32
    uint32_t n;
} vx_llm_swiglu_args_t;

// Embedding gather: out[t][d] = table[ids[t]][d]. table is vocab x dim
// f32 rows; ids are uint32 row indices (caller guarantees < vocab —
// the kernel does not bounds-check the vocabulary).
typedef struct {
    vx_dl_ptr_t table;  // vocab x dim f32
    vx_dl_ptr_t ids;    // n u32
    vx_dl_ptr_t out;    // n x dim f32
    uint32_t dim;
    uint32_t n;
} vx_llm_embedding_args_t;

// KV-cache append: cache[base + r][d] = new_k[r][d] for r in
// [0, new_tokens). Pure row copy; rows outside [base, base+new_tokens)
// are never written. capacity is host-side validation only.
typedef struct {
    vx_dl_ptr_t cache;      // capacity x dim f32
    vx_dl_ptr_t new_k;      // new_tokens x dim f32
    uint32_t base;          // first cache row to overwrite
    uint32_t new_tokens;
    uint32_t dim;
} vx_llm_kv_append_args_t;

#endif // VORTEX_DL_LLM_ARGS_H
