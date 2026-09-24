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

// Vortex DL Mamba selective scan kernel (plan P7 precursor, simplified S6).
// One warp (4 threads on this config) per (batch, channel); the N=8 state
// vector is split across the warp lanes (2 f32 registers per lane) and the
// scan runs sequentially over t. The per-timestep dot product y = C . S is
// reduced across lanes through an LMEM staging buffer + __syncthreads (the
// KMU-path-blessed reduction; no float atomics, no shuffles).
//
// VOLT workaround honored: arg-> fields are read directly at each use and
// are NEVER cached into local const variables.

#include <hip/hip_runtime.h>
#include <math.h>

#include "mamba_args.h"

#define MAMBA_LANES 4                     // warp size (VX_CFG_NUM_THREADS = 4)
#define MAMBA_MAX_N 8                     // register-unrolled d_state bound
#define MAMBA_N_PER_LANE (MAMBA_MAX_N / MAMBA_LANES)

__global__ void mamba_scan_kernel(vx_mamba_scan_args_t* arg) {
    // LMEM staging: one partial dot product per thread (16 f32 = 64 B).
    float* red = (float*)__local_mem();
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid % MAMBA_LANES;
    const uint32_t warp = tid / MAMBA_LANES;
    // Global (batch, channel) index owned by this warp.
    const uint32_t bc =
        blockIdx.x * (blockDim.x / MAMBA_LANES) + warp;

    // State registers: lane owns states n with n % 4 == lane, n < dstate.
    float s[MAMBA_N_PER_LANE] = {0.0f, 0.0f};

    for (uint32_t t = 0; t < arg->seqlen; ++t) {
        float partial = 0.0f;
        // Active warps advance the recurrence; idle warps (tail of the last
        // CTA) contribute zeros but MUST still reach both __syncthreads.
        if (bc < arg->batch * arg->channels) {
            const float decay =
                expf(((const float*)(uintptr_t)arg->a)[bc] *
                     ((const float*)(uintptr_t)arg->dt)
                         [(size_t)bc * arg->seqlen + t]);
            const float xv = ((const float*)(uintptr_t)arg->x)
                [(size_t)bc * arg->seqlen + t];
#pragma unroll
            for (uint32_t i = 0; i < MAMBA_N_PER_LANE; ++i) {
                const uint32_t n = lane + i * MAMBA_LANES;
                if (n < arg->dstate) {
                    s[i] = decay * s[i] +
                           ((const float*)(uintptr_t)arg->b)
                               [(size_t)t * arg->dstate + n] *
                           xv;
                    partial += ((const float*)(uintptr_t)arg->c)
                                   [(size_t)t * arg->dstate + n] *
                               s[i];
                }
            }
        }
        red[tid] = partial;
        __syncthreads();
        if (lane == 0 && bc < arg->batch * arg->channels) {
            ((float*)(uintptr_t)arg->y)[(size_t)bc * arg->seqlen + t] =
                red[warp * MAMBA_LANES + 0] + red[warp * MAMBA_LANES + 1] +
                red[warp * MAMBA_LANES + 2] + red[warp * MAMBA_LANES + 3];
        }
        __syncthreads();  // red[] reused next timestep: avoid WAR hazard
    }
}

__global__ void mamba_state_kernel(vx_mamba_state_args_t* arg) {
    const uint32_t bc = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    float* state = (float*)(uintptr_t)arg->final_state + (size_t)bc * arg->dstate;
    float* partials = (float*)__local_mem();
    for (uint32_t n = tid; n < arg->dstate; n += 16) {
        state[n] = arg->initial_state
            ? ((const float*)(uintptr_t)arg->initial_state)[(size_t)bc * arg->dstate + n]
            : 0.0f;
    }
    for (uint32_t t = 0; t < arg->seqlen; ++t) {
        const float decay = expf(((const float*)(uintptr_t)arg->a)[bc] *
            ((const float*)(uintptr_t)arg->dt)[(size_t)bc * arg->seqlen + t]);
        const float xv = ((const float*)(uintptr_t)arg->x)[(size_t)bc * arg->seqlen + t];
        float partial = 0.0f;
        for (uint32_t n = tid; n < arg->dstate; n += 16) {
            const float next = decay * state[n] +
                ((const float*)(uintptr_t)arg->b)[(size_t)t * arg->dstate + n] * xv;
            state[n] = next;
            partial += ((const float*)(uintptr_t)arg->c)[(size_t)t * arg->dstate + n] * next;
        }
        partials[tid] = partial;
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for (uint32_t i = 0; i < 16; ++i) total += partials[i];
            ((float*)(uintptr_t)arg->y)[(size_t)bc * arg->seqlen + t] = total;
        }
        __syncthreads();
    }
}
