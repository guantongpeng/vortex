// Stream/event overlap and cross-stream ordering acceptance (P2-03).
//
// Two streams each run: async H2D -> kernel -> event record. Stream 2 waits
// on stream 1's mid-point event before its own kernel, forming a DAG the
// runtime must honor even though the default CP has one hardware queue
// (correctness first, concurrency is a later capability node).

#include <hip/hip_runtime_api.h>

#include <cstdio>
#include <cstdint>
#include <vector>

#define HIP_CHECK(expr) do { \
    hipError_t _e = (expr); \
    if (_e != hipSuccess) { \
        fprintf(stderr, "FAILED at %s:%d: '%s' returned %s\n", \
                __FILE__, __LINE__, #expr, hipGetErrorString(_e)); \
        return 1; \
    } \
} while (0)

#if HIP_TEST_DEV_PTR_WIDTH == 32
typedef uint32_t dev_ptr_t;
#else
typedef uint64_t dev_ptr_t;
#endif
struct scale_args_t {
    dev_ptr_t in;
    dev_ptr_t out;
    uint32_t n;
    uint32_t scale;
    uint32_t pad;
};

int main(int argc, char** argv) {
    const char* vxbin = (argc > 1) ? argv[1] : "kernel.vxbin";

    HIP_CHECK(hipInit(0));
    hipDeviceProp_t prop = {};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    const uint32_t N = 128;
    std::vector<int> h_a(N), h_b(N), h_out1(N), h_out2(N);
    for (uint32_t i = 0; i < N; ++i) {
        h_a[i] = (int)i;
        h_b[i] = (int)(N - i);
    }

    void *d_a = nullptr, *d_b = nullptr, *d_o1 = nullptr, *d_o2 = nullptr;
    HIP_CHECK(hipMalloc(&d_a, N * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_b, N * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_o1, N * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_o2, N * sizeof(int)));

    hipModule_t module = nullptr;
    hipFunction_t func = nullptr;
    HIP_CHECK(hipModuleLoad(&module, vxbin));
    HIP_CHECK(hipModuleGetFunction(&func, module, "scale_kernel"));

    hipStream_t s1 = nullptr, s2 = nullptr;
    HIP_CHECK(hipStreamCreate(&s1));
    HIP_CHECK(hipStreamCreate(&s2));
    hipEvent_t e1_mid = nullptr, e1_done = nullptr, e2_done = nullptr;
    HIP_CHECK(hipEventCreate(&e1_mid));
    HIP_CHECK(hipEventCreate(&e1_done));
    HIP_CHECK(hipEventCreate(&e2_done));

    // Stream 1: upload a, scale by 3, mark mid + done.
    HIP_CHECK(hipMemcpyAsync(d_a, h_a.data(), N * sizeof(int),
                             hipMemcpyHostToDevice, s1));
    scale_args_t args1 = {};
    args1.in = (dev_ptr_t)(uintptr_t)d_a;
    args1.out = (dev_ptr_t)(uintptr_t)d_o1;
    args1.n = N;
    args1.scale = 3;
    void* params1[1] = { &args1 };
    uint32_t threads = (uint32_t)prop.warpSize;
    uint32_t blocks = (N + threads - 1) / threads;
    HIP_CHECK(hipModuleLaunchKernel(func, blocks, 1, 1, threads, 1, 1,
                                    0, s1, params1, nullptr));
    HIP_CHECK(hipEventRecord(e1_mid, s1));
    HIP_CHECK(hipMemcpyAsync(d_o1, d_o1, 0, hipMemcpyDeviceToDevice, s1)); // ordering marker (0 bytes)
    HIP_CHECK(hipEventRecord(e1_done, s1));

    // Stream 2: waits for stream 1's mid event, then uploads b and scales by 5.
    HIP_CHECK(hipStreamWaitEvent(s2, e1_mid));
    HIP_CHECK(hipMemcpyAsync(d_b, h_b.data(), N * sizeof(int),
                             hipMemcpyHostToDevice, s2));
    scale_args_t args2 = {};
    args2.in = (dev_ptr_t)(uintptr_t)d_b;
    args2.out = (dev_ptr_t)(uintptr_t)d_o2;
    args2.n = N;
    args2.scale = 5;
    void* params2[1] = { &args2 };
    HIP_CHECK(hipModuleLaunchKernel(func, blocks, 1, 1, threads, 1, 1,
                                    0, s2, params2, nullptr));
    HIP_CHECK(hipEventRecord(e2_done, s2));

    // Sync both streams, then fetch results with the (blocking) default path.
    HIP_CHECK(hipEventSynchronize(e1_done));
    HIP_CHECK(hipEventSynchronize(e2_done));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_out1.data(), d_o1, N * sizeof(int),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_out2.data(), d_o2, N * sizeof(int),
                        hipMemcpyDeviceToHost));

    uint32_t errors = 0;
    for (uint32_t i = 0; i < N; ++i) {
        if (h_out1[i] != h_a[i] * 3) ++errors;
        if (h_out2[i] != h_b[i] * 5) ++errors;
    }

    HIP_CHECK(hipEventDestroy(e1_mid));
    HIP_CHECK(hipEventDestroy(e1_done));
    HIP_CHECK(hipEventDestroy(e2_done));
    HIP_CHECK(hipStreamDestroy(s1));
    HIP_CHECK(hipStreamDestroy(s2));
    HIP_CHECK(hipModuleUnload(module));
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_o1));
    HIP_CHECK(hipFree(d_o2));
    HIP_CHECK(hipDeviceReset());

    printf("stream_overlap: errors=%u\n", errors);
    if (errors == 0) {
        printf("PASSED\n");
        return 0;
    }
    fprintf(stderr, "FAILED\n");
    return 1;
}
