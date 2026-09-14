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

#ifndef VORTEX_DL_MAMBA_ARGS_H
#define VORTEX_DL_MAMBA_ARGS_H

// Kernel argument block for the Mamba selective scan (plan P7 precursor,
// simplified S6 software reference). Same conventions as quant_args.h.
//
// == Data layout metadata ===================================================
// VXKMDATA "args_size" MUST equal the C sizeof of the struct on the target:
//   rv64 (vortex64): 6 device pointers (8 B each = 48) + 4 u32 (= 16)
//                    -> sizeof = 64 (already 8-byte aligned).
//   rv32 (vortex32): 6 device pointers (4 B each = 24) + 4 u32 (= 16)
//                    -> sizeof = 40 (align 4).
// All tensors are FP32, row-major; batch B, channels C, seqlen T, state
// dimension N (d_state) with 1 <= N <= 8:
//   a  : A      [B*C]     per-(b,c) SSM scale, negative (A < 0)
//   dt :        [B*C*T]   per-(b,c,t) positive timestep
//   b  : Bmat   [T*N]     input projection, shared across (b, c)
//   c  : Cmat   [T*N]     output projection, shared across (b, c)
//   x  :        [B*C*T]   per-(b,c) input sequence
//   y  :        [B*C*T]   output, written by the kernel
// Recurrence (initial state S = 0):
//   S[n] <- exp(A * dt) * S[n] + Bmat[t][n] * x[t]
//   y[t]  = sum_n Cmat[t][n] * S[n]
// ===========================================================================

#include <stdint.h>

#if defined(__VORTEX__)
typedef uintptr_t vx_dl_ptr_t;
#elif defined(VX_CFG_XLEN) && VX_CFG_XLEN == 32
typedef uint32_t vx_dl_ptr_t;
#else
typedef uint64_t vx_dl_ptr_t;
#endif

typedef struct {
    vx_dl_ptr_t a;         // A  [B*C]
    vx_dl_ptr_t dt;        //    [B*C*T]
    vx_dl_ptr_t b;         // B  [T*N], shared
    vx_dl_ptr_t c;         // C  [T*N], shared
    vx_dl_ptr_t x;         //    [B*C*T]
    vx_dl_ptr_t y;         //    [B*C*T], output
    uint32_t batch;        // B
    uint32_t channels;     // C
    uint32_t seqlen;       // T
    uint32_t dstate;       // N (1..8; kernel is register-unrolled for N<=8)
} vx_mamba_scan_args_t;

#endif // VORTEX_DL_MAMBA_ARGS_H
