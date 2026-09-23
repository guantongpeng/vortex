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

#ifndef VORTEX_DL_BLAS_H
#define VORTEX_DL_BLAS_H

// Vortex DL BLAS layer (plan P3-01): GEMM dispatch over the canonical
// vortex2.h ABI. The kernels are KMU images compiled by hipcc-vortex from
// sw/dl/src/blas_kernels.hip (multi-warp CTAs, LMEM tile staging); this
// host library loads the image and launches it.
//
// v1 dispatch: every dtype runs the tiled FPU kernel (FP32 math; FP16/BF16
// are storage formats converted at load time, accumulating in FP32 — the
// same correctness shape W4A16 will use). A TCU path slots in behind
// vx_blas_dispatch once a TCU-dtype capability ID exists in vortex2.h;
// until then TCU selection is a compile-time configuration of the image.

#include <stddef.h>
#include <stdint.h>
#include <vortex2.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum vx_blas_status {
    VX_BLAS_OK = 0,
    VX_BLAS_ERR_NOT_INITIALIZED = 1,
    VX_BLAS_ERR_BAD_ARGS = 2,
    VX_BLAS_ERR_UNSUPPORTED = 3,
    VX_BLAS_ERR_LAUNCH = 4,
    // init is idempotent only for the same device and image; this is what a
    // second caller with different arguments gets, instead of being handed
    // the first caller's kernels.
    VX_BLAS_ERR_ALREADY_INITIALIZED = 5,
} vx_blas_status;

typedef enum vx_blas_dtype {
    VX_BLAS_F32 = 0,
    VX_BLAS_F16 = 1,  // vx_fp16_t storage, FP32 accumulate
    VX_BLAS_BF16 = 2, // vx_bf16_t storage, FP32 accumulate
} vx_blas_dtype;

// Load the BLAS kernel image. `vxbin_path` points at the .vxbin built from
// sw/dl/src/blas_kernels.hip (the tests' Makefile builds it). One handle
// per device; release with vx_blas_finalize.
vx_blas_status vx_blas_init(vx_device_h dev, const char* vxbin_path);
vx_blas_status vx_blas_finalize(void);

// Row-major GEMM: C[M][N] = alpha * A[M][K] * B[K][N] + beta * C[M][N].
// A, B, C are device buffer addresses (vx_buffer_address values).
// Launches on `q` and returns without waiting — sync via the queue.
vx_blas_status vx_blas_gemm(vx_queue_h q,
                            vx_blas_dtype dt,
                            uint32_t M, uint32_t N, uint32_t K,
                            float alpha, float beta,
                            uint64_t A, uint64_t B, uint64_t C);

// Same row-major GEMM with an optional transposed B storage. When transb is
// non-zero, B is supplied as [N][K] and the kernel consumes B^T directly;
// no temporary materialisation is needed by linear layers.
vx_blas_status vx_blas_gemm_ex(vx_queue_h q,
                                vx_blas_dtype dt,
                                uint32_t M, uint32_t N, uint32_t K,
                                float alpha, float beta,
                                uint64_t A, uint64_t B, uint64_t C,
                                uint32_t transb);

// Report which kernel variant the last dispatch selected (for test output
// and future perf attribution: "fallback" vs "tcu").
const char* vx_blas_kernel_name(vx_blas_dtype dt);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // VORTEX_DL_BLAS_H
