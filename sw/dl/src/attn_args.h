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

#ifndef VORTEX_DL_ATTN_ARGS_H
#define VORTEX_DL_ATTN_ARGS_H

// Kernel argument blocks for the attention kernels (plan P3 second tier,
// SAM-DINOv3 precursor). Conventions as blas_args.h / quant_args.h:
// device-width pointers, FP32 in / FP32 out, little-endian.
//
// DATA LAYOUT METADATA (single kernel: attn_forward_kernel)
// ---------------------------------------------------------------
// All four tensors are row-major FP32 with identical shape B*H*L*D
// elements, laid out [b][h][l][d] (batch, head, sequence, head-dim):
//
//   q : B*H*L*D f32   queries,   row r = (b*H + h)*L + l
//   k : B*H*L*D f32   keys,      row r = (b*H + h)*L + j
//   v : B*H*L*D f32   values,    row r = (b*H + h)*L + j
//   o : B*H*L*D f32   output,    row r = (b*H + h)*L + q
//
//   b, h, l, d : u32 shape fields (batch, heads, seq len, head dim)
//   causal      : u32 flag; 0 = full attention (row q attends every
//                j in [0, L)), nonzero = causal (row q attends j <= q)
//   scale       : f32 softmax scale. The host computes 1/sqrt(D) in
//                double and casts to f32; the device never calls
//                sqrtf so the reference can mirror the exact f32
//                scale bit-for-bit.
//
// Arg-block size (VXKMDATA args_size must equal the C sizeof):
//   rv64 (XLEN=64): 4 ptrs (8B each) = 32; b,h,l,d,causal = 5*4 = 20;
//                   scale (f32) at offset 52 -> sizeof 56 (align 8).
//   rv32 (XLEN=32): 4 ptrs (4B) + 5*4 + 4 = 40 (align 4).
//
// Launch shape: one 16-thread (4-warp) CTA per output row
// (grid.x = B*H*L, block.x = 16, unused dims = 1), lmem_size =
// ATTN_LMEM_BYTES: the CTA stages the row's L logits/probabilities in
// local memory (score[ATTN_MAX_SEQ]) plus 16 f32 reduction slots.
// Supported: L <= ATTN_MAX_SEQ (2048; 8256 of the 16384 LMEM bytes),
// any D (threads stride d by 16).

#include <stdint.h>

#if defined(__VORTEX__)
typedef uintptr_t vx_dl_ptr_t;
#elif defined(VX_CFG_XLEN) && VX_CFG_XLEN == 32
typedef uint32_t vx_dl_ptr_t;
#else
typedef uint64_t vx_dl_ptr_t;
#endif

#define ATTN_BLOCK 16        // threads per CTA (4 warps)
#define ATTN_MAX_SEQ 2048    // LMEM score-buffer capacity (kv positions)
#define ATTN_LMEM_FLOATS (ATTN_MAX_SEQ + ATTN_BLOCK)  // scores + red slots
#define ATTN_LMEM_BYTES (ATTN_LMEM_FLOATS * 4)

typedef struct {
    vx_dl_ptr_t q;      // B*H*L*D f32 [b][h][l][d]
    vx_dl_ptr_t k;      // B*H*L*D f32
    vx_dl_ptr_t v;      // B*H*L*D f32
    vx_dl_ptr_t o;      // B*H*L*D f32 (output)
    uint32_t b;         // batch
    uint32_t h;         // heads
    uint32_t l;         // sequence length
    uint32_t d;         // head dimension
    uint32_t causal;    // 0 = full, nonzero = causal (j <= q)
    float scale;        // 1/sqrt(D) from host double, cast to f32
} vx_attn_forward_args_t;

// The host packs this block, so the host sizeof IS the VXKMDATA
// args_size; the device sees the identical field order/types at the
// same XLEN. (No assert under __VORTEX__: trailing padding depends on
// the device ABI and the host-side asserts below pin the packed size.)
#if !defined(__VORTEX__)
#if defined(VX_CFG_XLEN) && VX_CFG_XLEN == 32
static_assert(sizeof(vx_attn_forward_args_t) == 40, "attn args rv32 layout");
#else
static_assert(sizeof(vx_attn_forward_args_t) == 56, "attn args rv64 layout");
#endif
#endif

#endif // VORTEX_DL_ATTN_ARGS_H
