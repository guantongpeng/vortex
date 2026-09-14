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

// Vortex DL attention kernels (plan P3 second tier, SAM-DINOv3
// precursor): scaled dot-product attention forward.
//
//   O[b][h][q][:] = softmax(scale * Q[b][h][q][:] . K[b][h]'^T) . V[b][h]
//
// Shipped variant: TWO-PASS row attention with the row's L logits /
// probabilities staged in LMEM ("fused-softmax-in-LMEM"): pass 1 computes
// s_j = scale * <Q_q, K_j> f32 and the block max; pass 2 overwrites the
// staged scores with p_j = expf(s_j - m) and reduces the block sum; pass 3
// accumulates the p-weighted V rows and normalizes. Staging the probs in
// LMEM (instead of recomputing the dots a third time) keeps the K/V reads
// at one Q-row pass plus one V pass per output row.
//
// Shape: one 16-thread (4-warp) CTA per output row (b,h,q); kv positions
// are strided across threads in passes 1-2, head dims in pass 3.
// Conventions honored: no arg-field caching into locals (VOLT -O3
// workaround — scalar arg fields are read through arg-> at each use;
// only per-row ADDRESS derivations follow the prim_softmax precedent),
// no float atomics, cross-warp reduction via __syncthreads + LMEM.

#include <hip/hip_runtime.h>
#include <math.h>

#include "attn_args.h"

static __device__ float attn_warp_sum(float x) {
    for (uint32_t off = warpSize / 2; off > 0; off >>= 1) {
        x += __shfl_down(x, (int)off, warpSize);
    }
    return x;
}

static __device__ float attn_warp_max(float x) {
    for (uint32_t off = warpSize / 2; off > 0; off >>= 1) {
        x = fmaxf(x, __shfl_down(x, (int)off, warpSize));
    }
    return x;
}

__global__ void attn_forward_kernel(vx_attn_forward_args_t* arg) {
    if (arg->l == 0 || arg->l > ATTN_MAX_SEQ || arg->d == 0) return;

    float* score = (float*)__local_mem();  // [ATTN_MAX_SEQ]: s_j then p_j
    float* red = score + ATTN_MAX_SEQ;     // [4] warp partials, [8] broadcast

    const uint32_t t = threadIdx.x;
    const uint32_t row = blockIdx.x;       // = (b*H + h)*L + q
    const uint32_t jmax = (arg->causal != 0) ? (row % arg->l)   // causal: q
                                             : (arg->l - 1);    // full
    // Per-row base addresses (single derivation, prim_softmax precedent).
    const float* Qrow = (const float*)(uintptr_t)arg->q + (size_t)row * arg->d;
    const float* Kb = (const float*)(uintptr_t)arg->k +
                      (size_t)(row / arg->l) * arg->l * arg->d;
    const float* Vb = (const float*)(uintptr_t)arg->v +
                      (size_t)(row / arg->l) * arg->l * arg->d;
    float* Orow = (float*)(uintptr_t)arg->o + (size_t)row * arg->d;

    // pass 1: logits (kv stride) + running max ----------------------------
    float m = -INFINITY;
    for (uint32_t j = t; j <= jmax; j += ATTN_BLOCK) {
        const float* Kj = Kb + (size_t)j * arg->d;
        float dot = 0.0f;
        for (uint32_t dd = 0; dd < arg->d; ++dd) {
            dot += Qrow[dd] * Kj[dd];
        }
        const float s = dot * arg->scale;
        score[j] = s;
        m = fmaxf(m, s);
    }
    m = attn_warp_max(m);
    if (t % warpSize == 0) red[t / warpSize] = m;
    __syncthreads();
    if (t == 0) {
        float v = red[0];
        for (uint32_t w = 1; w < ATTN_BLOCK / warpSize; ++w) {
            v = fmaxf(v, red[w]);
        }
        red[ATTN_BLOCK / 2] = v;
    }
    __syncthreads();
    m = red[ATTN_BLOCK / 2];

    // pass 2: p_j = expf(s_j - m) in place + block sum --------------------
    float psum = 0.0f;
    for (uint32_t j = t; j <= jmax; j += ATTN_BLOCK) {
        const float p = expf(score[j] - m);
        score[j] = p;
        psum += p;
    }
    psum = attn_warp_sum(psum);
    if (t % warpSize == 0) red[t / warpSize] = psum;
    __syncthreads();
    if (t == 0) {
        float v = red[0];
        for (uint32_t w = 1; w < ATTN_BLOCK / warpSize; ++w) {
            v += red[w];
        }
        red[ATTN_BLOCK / 2] = v;
    }
    __syncthreads();
    const float denom = red[ATTN_BLOCK / 2];

    // pass 3: O[q][d] = (sum_j p_j * V[j][d]) / sum_j p_j (d stride) -------
    for (uint32_t dd = t; dd < arg->d; dd += ATTN_BLOCK) {
        float acc = 0.0f;
        for (uint32_t j = 0; j <= jmax; ++j) {
            acc += score[j] * Vb[(size_t)j * arg->d + dd];
        }
        Orow[dd] = acc / denom;
    }
}
