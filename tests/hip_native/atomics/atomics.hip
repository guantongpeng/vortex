// Native HIPVortex atomics (plan P2-03): int atomicAdd histogram across
// multi-warp blocks — the RVA hardware AMO path.
//
// Run from a build tree whose simulator was built with
// -DVX_CFG_EXT_A_ENABLE, and pass the same flag via CONFIGS so the gate
// below matches the image:
//   make run-simx CONFIGS=-DVX_CFG_EXT_A_ENABLE
// (SimX hardcodes VX_CSR_MISA to 0, so the runtime query cannot be used.)
//
// float atomicAdd is deliberately NOT exercised here: its CAS-loop
// fallback livelocks when several lanes contend on the same address
// (measured: 4 lanes x 1 increment never terminates), because without
// ZACAS there is no true compare-and-swap and lr/sc reservations
// invalidate each other. It needs -DVX_CFG_EXT_ZACAS_ENABLE hardware to
// be usable; see the p2_03 doc.

#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void histogram_kernel(const int* in, int* hist, int n, int bins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&hist[in[i]], 1);
    }
}

int main() {
    // Capability gate: the image must be compiled for an A-enabled device.
#if !defined(VX_CFG_EXT_A_ENABLED) || !VX_CFG_EXT_A_ENABLED
    printf("atomics: built without -DVX_CFG_EXT_A_ENABLE, skipping\n");
    printf("FAILED\n");
    return 1;
#else
    const int N = 512;
    const int BINS = 8;
    int* in = (int*)hipMalloc(N * sizeof(int));
    int* hist = (int*)hipMalloc(BINS * sizeof(int));
    if (!in || !hist) {
        printf("hipMalloc failed\n");
        return 1;
    }
    int ref_hist[BINS] = {0};
    for (int i = 0; i < N; ++i) {
        in[i] = (i * i) % BINS;
        ++ref_hist[in[i]];
    }
    for (int b = 0; b < BINS; ++b) hist[b] = 0;

    int threads = 16;
    int blocks = (N + threads - 1) / threads;
    hipLaunchKernelGGL(histogram_kernel, dim3(blocks), dim3(threads), 0, 0,
                       in, hist, N, BINS);
    hipDeviceSynchronize();

    int errors = 0;
    for (int b = 0; b < BINS; ++b) {
        if (hist[b] != ref_hist[b]) {
            printf("hist[%d]=%d ref=%d\n", b, hist[b], ref_hist[b]);
            ++errors;
        }
    }

    printf("atomics: hist_errors=%d\n", errors);
    if (errors == 0) {
        printf("PASSED\n");
        return 0;
    }
    printf("FAILED\n");
    return 1;
#endif  // A gate
}
