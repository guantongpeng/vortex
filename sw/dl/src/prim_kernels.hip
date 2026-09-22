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

// Vortex DL prim/norm kernels (plan P3-02). KMU image compiled by
// hipcc-vortex with --kernel-lib=vortex2.
//
// Shapes follow the P2-03 findings: elementwise work is grid-stride over
// single-warp CTAs (no cross-warp state); reductions/softmax/norm use one
// 16-thread multi-warp CTA per row with LMEM staging + __syncthreads,
// which the KMU path supports. No float atomics anywhere (the ZACAS
// livelock finding). Arg fields are read directly, never cached into
// locals (the VOLT -O3 miscompile workaround).

#include <hip/hip_runtime.h>
#include <math.h>

#include "prim_args.h"

#define PRIM_BLOCK 16
#define PRIM_LMEM_FLOATS 64  // staging for per-warp partials

// ---------------------------------------------------------------------------
// Elementwise unary
// ---------------------------------------------------------------------------

__global__ void prim_unary_kernel(vx_prim_unary_args_t* arg) {
    const float* in = (const float*)(uintptr_t)arg->in;
    float* out = (float*)(uintptr_t)arg->out;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < arg->n;
         i += blockDim.x * gridDim.x) {
        float x = in[i];
        float y;
        switch (arg->op) {
        case VX_PRIM_RELU:
            // max(x, 0) with PyTorch's semantics: NaN propagates and -0.0
            // keeps its sign. `x > 0.0f ? x : 0.0f` turns NaN into 0 and
            // -0.0 into +0.0 (F01 in docs/mydocs/pytorch_plan.md).
            y = x < 0.0f ? 0.0f : x;
            break;
        case VX_PRIM_GELU_TANH:
            y = 0.5f * x * (1.0f + tanhf(
                0.7978845608028654f * (x + 0.044715f * x * x * x)));
            break;
        case VX_PRIM_GELU_ERF:
            // torch's default gelu; not the same function as the tanh form
            y = 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
            break;
        // fabsf, not `x < 0.0f ? -x : x`: the latter returns -0.0 for -0.0,
        // while torch.abs(-0.0) is +0.0.
        case VX_PRIM_ABS:        y = fabsf(x);              break;
        case VX_PRIM_EXP:        y = expf(x);               break;
        case VX_PRIM_LOG:        y = logf(x);               break;
        case VX_PRIM_SQRT:       y = sqrtf(x);              break;
        case VX_PRIM_RSQRT:      y = 1.0f / sqrtf(x);       break;
        case VX_PRIM_SIGMOID:    y = 1.0f / (1.0f + expf(-x)); break;
        case VX_PRIM_TANH:       y = tanhf(x);              break;
        case VX_PRIM_RECIPROCAL: y = 1.0f / x;              break;
        case VX_PRIM_SILU:
            y = x / (1.0f + expf(-x));
            break;
        default:  // VX_PRIM_NEG
            y = -x;
            break;
        }
        out[i] = y;
    }
}

// ---------------------------------------------------------------------------
// Row reduce: one 16-thread CTA per row, over that row's `cols` values.
// Multi-warp CTA with LMEM staging; out[r] = f(row r).
// ---------------------------------------------------------------------------

static __device__ float warp_sum(float x) {
    for (uint32_t off = warpSize / 2; off > 0; off >>= 1) {
        x += __shfl_down(x, (int)off, warpSize);
    }
    return x;
}

// max with NaN winning. fmaxf returns the non-NaN operand, so a NaN anywhere
// in the row used to reduce to whatever the other values were -- and both
// torch.amax and torch.max propagate it.
static __device__ float nan_max(float a, float b) {
    return (isnan(a) || isnan(b)) ? NAN : (a >= b ? a : b);
}

static __device__ float warp_max(float x) {
    for (uint32_t off = warpSize / 2; off > 0; off >>= 1) {
        x = nan_max(x, __shfl_down(x, (int)off, warpSize));
    }
    return x;
}

// The better of two (value, index) candidates: NaN beats every number, and
// among equals the smaller index wins -- which is what torch.argmax returns
// for a tie.
//
// Two NaNs are an equality, and that needs saying: NaN != NaN is true, so a
// plain `if (v != bv) return v > bv;` sends two NaNs down the value compare,
// gets false, and keeps whichever candidate happened to be on the left. The
// result is not the first NaN but the last one in the merge order, which is
// measurable: for [1, NaN, NaN, NaN] this returned index 2 where torch
// returns 1, and it returned 16 for a 21-wide row whose first element is the
// NaN, where torch returns 0.
static __device__ bool argmax_better(float v, uint32_t i, float bv, uint32_t bi) {
    if (isnan(v) != isnan(bv)) return isnan(v);
    if (isnan(v)) return i < bi;          // both NaN: the earlier index wins
    if (v != bv) return v > bv;
    return i < bi;
}

__global__ void prim_reduce_kernel(vx_prim_reduce_args_t* arg) {
    float* tbuf = (float*)__local_mem();
    const uint32_t lane = threadIdx.x % warpSize;
    const uint32_t wid = threadIdx.x / warpSize;
    const uint32_t row = blockIdx.x;
    const float* in = (const float*)(uintptr_t)arg->in + (size_t)row * arg->cols;
    const uint32_t cols = arg->cols;
    const uint32_t nwarp = PRIM_BLOCK / warpSize;

    float acc = (arg->op == VX_PRIM_RED_MAX) ? -INFINITY : 0.0f;
    float seen = -INFINITY;  // argmax: best value this lane has seen
    uint32_t seen_idx = 0;

    for (uint32_t i = threadIdx.x; i < cols; i += PRIM_BLOCK) {
        const float x = in[i];
        if (arg->op == VX_PRIM_RED_SUM || arg->op == VX_PRIM_RED_MEAN) {
            acc += x;
        } else {
            // The same compare serves the value and the index, so the two
            // cannot disagree about which element won. `isnan(x) && !isnan(seen)`
            // rather than `|| isnan(x)`: with the latter, a lane that already
            // holds a NaN adopts every later NaN as well and ends up reporting
            // the last one instead of the first.
            if (x > seen || (isnan(x) && !isnan(seen))) {
                seen = x;
                seen_idx = i;
            }
            acc = nan_max(acc, x);
        }
    }

    if (arg->op == VX_PRIM_RED_ARGMAX) {
        for (uint32_t off = warpSize / 2; off > 0; off >>= 1) {
            const float ov = __shfl_down(seen, (int)off, warpSize);
            const uint32_t oi = __shfl_down(seen_idx, (int)off, warpSize);
            if (argmax_better(ov, oi, seen, seen_idx)) {
                seen = ov;
                seen_idx = oi;
            }
        }
        uint32_t* ubuf = (uint32_t*)tbuf;
        if (lane == 0) {
            tbuf[wid] = seen;
            ubuf[PRIM_LMEM_FLOATS / 2 + wid] = seen_idx;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float bv = tbuf[0];
            uint32_t bi = ubuf[PRIM_LMEM_FLOATS / 2];
            for (uint32_t w = 1; w < nwarp; ++w) {
                const float v = tbuf[w];
                const uint32_t i2 = ubuf[PRIM_LMEM_FLOATS / 2 + w];
                if (argmax_better(v, i2, bv, bi)) {
                    bv = v;
                    bi = i2;
                }
            }
            ((uint32_t*)(uintptr_t)arg->out)[row] = bi;
            // The extreme this pass already visited, which torch's max(dim=)
            // returns alongside the index. Writing it here is what keeps that
            // a single pass over the row.
            if (arg->values) ((float*)(uintptr_t)arg->values)[row] = bv;
        }
        return;
    }

    acc = (arg->op == VX_PRIM_RED_MAX) ? warp_max(acc) : warp_sum(acc);
    if (lane == 0) {
        tbuf[wid] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = tbuf[0];
        for (uint32_t w = 1; w < nwarp; ++w) {
            v = (arg->op == VX_PRIM_RED_MAX) ? nan_max(v, tbuf[w]) : v + tbuf[w];
        }
        if (arg->op == VX_PRIM_RED_MEAN) v /= (float)cols;
        ((float*)(uintptr_t)arg->out)[row] = v;
    }
}

// ---------------------------------------------------------------------------
// The softmax family: one 16-thread CTA per row.
//
// softmax, log_softmax and logsumexp differ only in what pass 3 writes:
// a normalised distribution, its logarithm, or the single log-sum-exp per row.
// The max shift in pass 1 is what keeps them stable, and it is the same shift
// for all three, so they are one kernel with a selector rather than three
// copies of the same two passes.
// ---------------------------------------------------------------------------

__global__ void prim_softmax_kernel(vx_prim_rowargs_t* arg) {
    float* tbuf = (float*)__local_mem();
    const uint32_t row = blockIdx.x;
    const float* in = (const float*)(uintptr_t)arg->in + (size_t)row * arg->cols;
    float* out = (float*)(uintptr_t)arg->out + (size_t)row * arg->cols;
    const uint32_t cols = arg->cols;

    // pass 1: row max, NaN-aware.
    //
    // For softmax and log_softmax the NaN-ness of this value does not matter:
    // a NaN in the row reaches the output through pass 2's exponential and the
    // sum whatever the max was. It matters for logsumexp, whose pass 3 takes
    // an infinite max as the answer itself -- `isinf(m)` has to mean "the row
    // really is infinite", and a NaN-blind max reports +inf for [NaN, inf],
    // where torch answers NaN. Making the max NaN-aware is what keeps that
    // test meaningful, and it changes no result for the other two.
    float m = -INFINITY;
    for (uint32_t j = threadIdx.x; j < cols; j += PRIM_BLOCK) {
        m = nan_max(m, in[j]);
    }
    m = warp_max(m);
    if (threadIdx.x % warpSize == 0) {
        tbuf[threadIdx.x / warpSize] = m;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = tbuf[0];
        for (uint32_t w = 1; w < PRIM_BLOCK / warpSize; ++w) {
            v = nan_max(v, tbuf[w]);
        }
        tbuf[8] = v;  // broadcast slot
    }
    __syncthreads();
    m = tbuf[8];

    // pass 2: exp(x - max) and its sum. logsumexp keeps no per-element values,
    // so it does not write here at all -- and its `out` is rows floats, not
    // rows*cols, so writing would run off the end of the buffer.
    float s = 0.0f;
    for (uint32_t j = threadIdx.x; j < cols; j += PRIM_BLOCK) {
        float e = expf(in[j] - m);
        if (arg->op != VX_PRIM_ROW_LOGSUMEXP) out[j] = e;
        s += e;
    }
    s = warp_sum(s);
    if (threadIdx.x % warpSize == 0) {
        tbuf[threadIdx.x / warpSize] = s;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = tbuf[0];
        for (uint32_t w = 1; w < PRIM_BLOCK / warpSize; ++w) v += tbuf[w];
        tbuf[8] = v;
    }
    __syncthreads();
    s = tbuf[8];

    // pass 3: what this op is for
    if (arg->op == VX_PRIM_ROW_LOGSUMEXP) {
        // log(sum) + max, the inverse of the shift.
        //
        // An infinite row max is the answer itself: every element is that
        // infinity or below it, so the log-sum-exp of the row is it. The
        // general path gives NaN here instead -- the shift inf - inf is NaN
        // and it poisons the sum -- and torch returns the infinity, so the
        // two disagree on exactly the rows where the answer is trivial.
        if (threadIdx.x == 0) {
            ((float*)(uintptr_t)arg->out)[row] = isinf(m) ? m : logf(s) + m;
        }
        return;
    }
    if (arg->op == VX_PRIM_ROW_LOG_SOFTMAX) {
        // (x - max) - log(sum). Subtracting in this order rather than taking
        // log(softmax) keeps the small values that softmax would round away.
        const float ls = logf(s);
        for (uint32_t j = threadIdx.x; j < cols; j += PRIM_BLOCK) {
            out[j] = (in[j] - m) - ls;
        }
        return;
    }
    for (uint32_t j = threadIdx.x; j < cols; j += PRIM_BLOCK) {
        out[j] /= s;
    }
}

// ---------------------------------------------------------------------------
// LayerNorm / RMSNorm: one 16-thread CTA per row.
// ---------------------------------------------------------------------------

__global__ void prim_layernorm_kernel(vx_prim_norm_args_t* arg) {
    float* tbuf = (float*)__local_mem();
    const uint32_t row = blockIdx.x;
    const float* in = (const float*)(uintptr_t)arg->in + (size_t)row * arg->cols;
    const float* gamma = (const float*)(uintptr_t)arg->gamma;
    const float* beta = (const float*)(uintptr_t)arg->beta;
    float* out = (float*)(uintptr_t)arg->out + (size_t)row * arg->cols;
    const uint32_t cols = arg->cols;

    // Both passes work in coordinates shifted by the row's first element.
    //
    // The deviations have to be measured from something that can represent
    // them. For a row of large values the mean cannot: [1e7, 1e7+1, 1e7+2,
    // 1e7+3] has an exact mean of 1e7+1.5, which is not a float32 at ulp 1,
    // so the deviations from the rounded mean are [-2,-1,0,1] instead of
    // [-1.5,-0.5,0.5,1.5] and the variance comes out 1.5 where it is 1.25 --
    // an 8% error in rstd, and worse as the spread shrinks relative to the
    // magnitude. Subtracting one element first is exact, because the shifted
    // values are small, so the sums are conditioned on the row's spread
    // rather than on its magnitude.
    const float anchor = in[0];

    // pass 1: the mean, in shifted coordinates
    float s1 = 0.0f;
    for (uint32_t j = threadIdx.x; j < cols; j += PRIM_BLOCK) {
        s1 += in[j] - anchor;
    }
    s1 = warp_sum(s1);
    if (threadIdx.x % warpSize == 0) {
        tbuf[threadIdx.x / warpSize] = s1;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float v1 = 0.0f;
        for (uint32_t w = 0; w < PRIM_BLOCK / warpSize; ++w) v1 += tbuf[w];
        tbuf[16] = v1 / (float)cols;              // mean, shifted
    }
    __syncthreads();
    const float mean_shift = tbuf[16];

    // pass 2: the variance, as the mean squared deviation from that mean.
    //
    // It used to be E[x^2] - mean^2 in one pass, which is the same number in
    // exact arithmetic and a different one in float32: for a row around 1e6
    // both terms are ~1e12 and their difference is ~1, so every significant
    // digit cancels and the result clamps to 0. Two passes cost one more read
    // of a row that is already in cache, and Welford -- what ATen uses -- is
    // not available without a cross-thread recurrence.
    float s2 = 0.0f;
    for (uint32_t j = threadIdx.x; j < cols; j += PRIM_BLOCK) {
        const float d = (in[j] - anchor) - mean_shift;
        s2 += d * d;
    }
    s2 = warp_sum(s2);
    if (threadIdx.x % warpSize == 0) {
        tbuf[threadIdx.x / warpSize] = s2;
    }
    __syncthreads();
    const float mean = anchor + mean_shift;
    if (threadIdx.x == 0) {
        float v2 = 0.0f;
        for (uint32_t w = 0; w < PRIM_BLOCK / warpSize; ++w) v2 += tbuf[w];
        const float var = v2 / (float)cols;
        const float rstd = rsqrtf(var + arg->eps);
        tbuf[17] = rstd;
        if (arg->mean) ((float*)(uintptr_t)arg->mean)[row] = mean;
        if (arg->rstd) ((float*)(uintptr_t)arg->rstd)[row] = rstd;
    }
    __syncthreads();
    const float rstd = tbuf[17];

    // The affine is applied in shifted coordinates too: (x - anchor) -
    // mean_shift is the deviation, and it is the accurate one. gamma and beta
    // are independent options -- torch takes either alone, with the other
    // standing for its identity -- so each is tested where it is read.
    if (gamma) {
        for (uint32_t j = threadIdx.x; j < cols; j += PRIM_BLOCK) {
            const float d = (in[j] - anchor) - mean_shift;
            const float s = d * rstd * gamma[j];
            out[j] = beta ? s + beta[j] : s;
        }
    } else {
        for (uint32_t j = threadIdx.x; j < cols; j += PRIM_BLOCK) {
            const float d = (in[j] - anchor) - mean_shift;
            const float s = d * rstd;
            out[j] = beta ? s + beta[j] : s;
        }
    }
}

__global__ void prim_rmsnorm_kernel(vx_prim_rmsnorm_args_t* arg) {
    float* tbuf = (float*)__local_mem();
    const uint32_t row = blockIdx.x;
    const float* in = (const float*)(uintptr_t)arg->in + (size_t)row * arg->cols;
    const float* gamma = (const float*)(uintptr_t)arg->gamma;
    float* out = (float*)(uintptr_t)arg->out + (size_t)row * arg->cols;
    const uint32_t cols = arg->cols;

    float s2 = 0.0f;
    for (uint32_t j = threadIdx.x; j < cols; j += PRIM_BLOCK) {
        s2 += in[j] * in[j];
    }
    s2 = warp_sum(s2);
    if (threadIdx.x % warpSize == 0) {
        tbuf[threadIdx.x / warpSize] = s2;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = 0.0f;
        for (uint32_t w = 0; w < PRIM_BLOCK / warpSize; ++w) v += tbuf[w];
        tbuf[16] = rsqrtf(v / (float)cols + arg->eps);  // 1/rms
    }
    __syncthreads();
    const float inv_rms = tbuf[16];

    // gamma is optional: torch.rms_norm(x, shape) and nn.RMSNorm without a
    // weight pass none, which is a mode rather than a caller error.
    if (gamma) {
        for (uint32_t j = threadIdx.x; j < cols; j += PRIM_BLOCK) {
            out[j] = in[j] * inv_rms * gamma[j];
        }
    } else {
        for (uint32_t j = threadIdx.x; j < cols; j += PRIM_BLOCK) {
            out[j] = in[j] * inv_rms;
        }
    }
}
