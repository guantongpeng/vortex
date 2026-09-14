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

#ifndef VORTEX_DL_SPARSE24_ARGS_H
#define VORTEX_DL_SPARSE24_ARGS_H

// Kernel argument blocks for the 2:4 structured-sparsity kernels
// (plan P6-02 Q5). Same conventions as quant_args.h / blas_args.h.
//
// ---------------------------------------------------------------------------
// 2:4 DATA LAYOUT (canonical simple form, N x K row-major weight matrix)
// ---------------------------------------------------------------------------
// values:  N x K FP32, row-major. Kept elements carry their original FP32
//          bits; pruned elements are exactly +0.0f. Row stride = K.
// mask:    N x ceil(K/4) uint8, row-major. Byte b of a row covers the
//          consecutive-4 group starting at column 4*b. Each element owns a
//          2-bit little-endian field: field i = bits [2i, 2i+1] of the byte
//          for element i of the group (i = k mod 4). Field value 1 = kept,
//          0 = pruned (2/3 reserved). Trailing fields of a ragged final
//          group (K not divisible by 4) are 0.
// Semantics: in every consecutive group of 4 along K, the 2 largest
//          magnitudes are kept; ties break to the LOWER index (deterministic
//          both on device and in the host mirror). A ragged tail group of
//          g < 4 elements keeps its top-min(2, g).
// Sparsity: with K divisible by 4, exactly half of all elements are pruned.
//
// Arg-block sizeof (device-width pointers, C struct layout):
//   prune: 3 ptr + 2 u32 -> rv64: 24 + 8  = 32 (align 8, no pad)
//                            rv32: 12 + 8  = 20
//   gemm:  4 ptr + 3 u32 -> rv64: 32 + 12 = 44 -> pad to align 8 = 48
//                            rv32: 16 + 12 = 28
// VXKMDATA args_size MUST equal these (see meta.json / integration.md).

#include <stdint.h>

#if defined(__VORTEX__)
typedef uintptr_t vx_dl_ptr_t;
#elif defined(VX_CFG_XLEN) && VX_CFG_XLEN == 32
typedef uint32_t vx_dl_ptr_t;
#else
typedef uint64_t vx_dl_ptr_t;
#endif

// Prune dense FP32 weights to the 2:4 form (values + mask), one thread per
// (row, group-of-4); output is bit-deterministic.
typedef struct {
    vx_dl_ptr_t weights;  // N x K f32 dense, row-major (in)
    vx_dl_ptr_t values;   // N x K f32, pruned slots = +0.0f (out)
    vx_dl_ptr_t mask;     // N x ceil(K/4) uint8 (out)
    uint32_t n;           // rows (output channels)
    uint32_t k;           // columns (reduction dim)
} vx_sparse24_prune_args_t;

// Sparse GEMM: C[M][N] = A[M][K] (dense f32) x values[N][K]^T, FP32
// accumulate, 16x16 output tile / 4-warp CTA / LMEM staging (same shape as
// quant_gemm_w4a16_kernel). The weight-staging path consults the 2-bit mask
// fields on device and never loads a pruned element.
typedef struct {
    vx_dl_ptr_t act;     // M x K f32 dense, row-major
    vx_dl_ptr_t values;  // N x K f32, pruned slots +0.0f
    vx_dl_ptr_t mask;    // N x ceil(K/4) uint8
    vx_dl_ptr_t out;     // M x N f32, row-major
    uint32_t m;
    uint32_t n;
    uint32_t k;
} vx_sparse24_gemm_args_t;

#endif // VORTEX_DL_SPARSE24_ARGS_H
