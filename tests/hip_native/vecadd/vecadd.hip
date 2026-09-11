// Native HIPVortex vecadd: first HIP source compiled end-to-end by
// hipcc-vortex into a .vxbin that runs on simx/rtlsim.
//
// This is the repository's hostless execution model: main() runs on the
// device itself, so the HIP host APIs (hipMalloc/hipMemcpy/hipLaunch) are
// the device-local shims from third_party/hip-vortex/include/hip.

#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void vecadd_kernel(const int* a, const int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    printf("hip_native vecadd: warpSize=%d\n", warpSize);

    const int N = 64;
    int* a = (int*)hipMalloc(N * sizeof(int));
    int* b = (int*)hipMalloc(N * sizeof(int));
    int* c = (int*)hipMalloc(N * sizeof(int));
    if (!a || !b || !c) {
        printf("hipMalloc failed\n");
        return 1;
    }

    for (int i = 0; i < N; ++i) {
        a[i] = i * 2;
        b[i] = i * 3;
        c[i] = 0;
    }

    int threads = warpSize;
    int blocks = (N + threads - 1) / threads;
    hipLaunchKernelGGL(vecadd_kernel, dim3(blocks), dim3(threads), 0, 0,
                       a, b, c, N);
    hipDeviceSynchronize();

    if (hipGetLastError() != hipSuccess) {
        printf("kernel launch failed\n");
        return 1;
    }

    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (c[i] != a[i] + b[i]) {
            if (errors < 4) {
                printf("mismatch at %d: %d != %d\n", i, c[i], a[i] + b[i]);
            }
            ++errors;
        }
    }

    if (errors == 0) {
        printf("PASSED\n");
        return 0;
    }
    printf("FAILED (%d errors)\n", errors);
    return 1;
}
