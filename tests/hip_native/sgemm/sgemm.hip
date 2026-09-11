// Native HIPVortex FP32 SGEMM (plan P2-03): the generic FPU kernel that
// P3-01's TCU path will be measured against. 2D grid/block mapping and a
// row-major reference computed on device.

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <math.h>

// C = A x B ; row-major; A is MxK, B is KxN, C is MxN.
__global__ void sgemm_kernel(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    uint32_t i = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (uint32_t)M && j < (uint32_t)N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = acc;
    }
}

int main() {
    const int M = 32, N = 32, K = 32;
    float* A = (float*)hipMalloc(M * K * sizeof(float));
    float* B = (float*)hipMalloc(K * N * sizeof(float));
    float* C = (float*)hipMalloc(M * N * sizeof(float));
    if (!A || !B || !C) {
        printf("hipMalloc failed\n");
        return 1;
    }

    unsigned int seed = 42;
    auto frand = [&seed]() {
        seed = seed * 1103515245u + 12345u;
        return (float)(seed >> 8 & 0xffffu) / 65536.0f - 0.5f;
    };

    for (int i = 0; i < M * K; ++i) A[i] = frand();
    for (int i = 0; i < K * N; ++i) B[i] = frand();

    dim3 block(4, 4);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    hipLaunchKernelGGL(sgemm_kernel, grid, block, 0, 0, A, B, C, M, N, K);
    hipDeviceSynchronize();

    int errors = 0;
    float max_rel = 0.0f;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float ref = 0.0f;
            for (int k = 0; k < K; ++k) {
                ref += A[i * K + k] * B[k * N + j];
            }
            float got = C[i * N + j];
            float rel = fabsf(got - ref) / (fabsf(ref) + 1e-6f);
            if (rel > max_rel) max_rel = rel;
            if (rel > 1e-4f) {
                if (errors < 3) {
                    printf("mismatch C[%d][%d]=%.6f ref=%.6f\n", i, j,
                           (double)got, (double)ref);
                }
                ++errors;
            }
        }
    }

    printf("sgemm %dx%dx%d: max_rel=%.2e errors=%d\n", M, N, K,
           (double)max_rel, errors);
    if (errors == 0) {
        printf("PASSED\n");
        return 0;
    }
    printf("FAILED\n");
    return 1;
}
