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

#ifndef VORTEX_DL_BLAS_ARGS_H
#define VORTEX_DL_BLAS_ARGS_H

// Kernel argument block shared by the device image and the host library.
// Pointers are DEVICE-width integers: the host is 64-bit but an rv32
// device reads 32-bit pointers, so the host packs with the width it was
// configured for (VX_CFG_XLEN from the build tree) and the device sees
// its native uintptr_t. Never memcpy a host pointer struct verbatim.

#include <stdint.h>

#if defined(__VORTEX__)
// Device compile (hipcc-vortex defines __VORTEX__): native width.
typedef uintptr_t vx_blas_ptr_t;
#elif defined(VX_CFG_XLEN) && VX_CFG_XLEN == 32
typedef uint32_t vx_blas_ptr_t;
#else
typedef uint64_t vx_blas_ptr_t;
#endif

typedef struct {
    vx_blas_ptr_t A;
    vx_blas_ptr_t B;
    vx_blas_ptr_t C;
    uint32_t M;
    uint32_t N;
    uint32_t K;
    float alpha;
    float beta;
    uint32_t transb;  // 0: B[K][N], 1: B[N][K]
} vx_blas_gemm_args_t;

#endif // VORTEX_DL_BLAS_ARGS_H
