// Native HIPVortex warp reduction (plan P2-03): __shfl_down tree reduction
// across many single-warp blocks.
//
// IMPORTANT (measured, see p2_03 doc): in the legacy spawn model
// (libvortex.a) __syncthreads() for multi-warp blocks is unreliable —
// the barrier id is local_group_id (0 for every group when a block spans
// multiple warps), warps iterate their groups independently, and the
// spawn machinery's own join collides with the same barrier. Depending
// on timing the outcome is a deadlock, a lost LMEM update, or an
// accidental pass. Multi-warp CTA semantics belong to the KMU path
// (vortex2 + host launch); this hostless test therefore uses one warp
// per block, which needs no cross-warp synchronization at all.

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <math.h>

__global__ void reduce_kernel(const float* in, float* out, int n) {
    // one warp per block: pure warp-shuffle reduction, no cross-warp state
    uint32_t gsize = blockDim.x * gridDim.x;
    float acc = 0.0f;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (uint32_t)n;
         i += gsize) {
        acc += in[i];
    }
    for (uint32_t off = warpSize / 2; off > 0; off >>= 1) {
        acc += __shfl_down(acc, (int)off, warpSize);
    }
    if (threadIdx.x == 0) {
        out[blockIdx.x] = acc;
    }
}

int main() {
    const int N = 1024;
    float* in = (float*)hipMalloc(N * sizeof(float));
    float* out = (float*)hipMalloc(256 * sizeof(float));
    if (!in || !out) {
        printf("hipMalloc failed\n");
        return 1;
    }

    float ref = 0.0f;
    for (int i = 0; i < N; ++i) {
        in[i] = (float)(i % 97) * 0.25f;
        ref += in[i];
    }

    int threads = warpSize;         // one warp per block
    int blocks = (N + threads - 1) / threads;
    hipLaunchKernelGGL(reduce_kernel, dim3(blocks), dim3(threads), 0, 0,
                       in, out, N);
    hipDeviceSynchronize();
    float got = 0.0f;
    for (int b = 0; b < blocks; ++b) got += out[b];

    printf("reduce: ref=%.3f got=%.3f blocks=%d warpSize=%d\n",
           (double)ref, (double)got, blocks, warpSize);
    float tol = 1e-2f * fabsf(ref) + 1e-3f;
    if (fabsf(got - ref) <= tol) {
        printf("PASSED\n");
        return 0;
    }
    printf("FAILED\n");
    return 1;
}
