// libhip_vortex module API acceptance (P2-02).
//
// Exercises the host HIP subset end-to-end against a KMU image built by
// hipcc-vortex: device/context query, hipMalloc/hipMemcpy/hipMemset,
// hipModuleLoad/GetFunction/LaunchKernel with the single-pointer arg-block
// convention (args_size from the image's VXKMDATA metadata), stream/event
// creation + record + synchronize + elapsed, and error surfaces.

#include <hip/hip_runtime_api.h>
#include <vortex2.h>

#include <cstdio>
#include <cstdlib>
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

// Must mirror kernel.hip's vecadd_args_t using DEVICE-width integers:
// the host is 64-bit but an rv32 device reads 32-bit pointers from the
// argument block, so the host packs with the device's pointer width.
// HIP_TEST_DEV_PTR_WIDTH comes from the test Makefile (32 or 64).
#if HIP_TEST_DEV_PTR_WIDTH == 32
typedef uint32_t dev_ptr_t;
#else
typedef uint64_t dev_ptr_t;
#endif
struct vecadd_args_t {
    dev_ptr_t src0;
    dev_ptr_t src1;
    dev_ptr_t dst;
    uint32_t num_elements;
    uint32_t pad;
};

int main(int argc, char** argv) {
    const char* vxbin = (argc > 1) ? argv[1] : "kernel.vxbin";

    HIP_CHECK(hipInit(0));
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    if (count < 1) {
        fprintf(stderr, "FAILED: no devices\n");
        return 1;
    }
    hipDeviceProp_t prop = {};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("device=%s sm=%d warpSize=%d clock=%dkHz mem=%zuMB lmem=%zu\n",
           prop.name, prop.multiProcessorCount, prop.warpSize,
           prop.clockRate, prop.totalGlobalMem >> 20,
           prop.sharedMemPerBlock);
    if (prop.warpSize != 4) {
        // Default profile is 4 lanes/warp — anything else means the CSR
        // query path is broken and warpSize was guessed.
        fprintf(stderr, "FAILED: unexpected warpSize %d\n", prop.warpSize);
        return 1;
    }

    const uint32_t N = 256;
    std::vector<int> h_src0(N), h_src1(N), h_dst(N, 0);
    for (uint32_t i = 0; i < N; ++i) {
        h_src0[i] = (int)(i * 2);
        h_src1[i] = (int)(i * 3);
    }

    void *d_src0 = nullptr, *d_src1 = nullptr, *d_dst = nullptr;
    HIP_CHECK(hipMalloc(&d_src0, N * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_src1, N * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_dst, N * sizeof(int)));
    HIP_CHECK(hipMemset(d_dst, 0, N * sizeof(int)));
    HIP_CHECK(hipMemcpy(d_src0, h_src0.data(), N * sizeof(int),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_src1, h_src1.data(), N * sizeof(int),
                        hipMemcpyHostToDevice));

    hipModule_t module = nullptr;
    hipFunction_t func = nullptr;
    HIP_CHECK(hipModuleLoad(&module, vxbin));
    HIP_CHECK(hipModuleGetFunction(&func, module, "vecadd_kernel"));

    int max_threads = 0, shared_bytes = -1, dynamic_bytes = 0;
    HIP_CHECK(hipFuncGetAttribute(&max_threads,
                                  hipFuncAttributeMaxThreadsPerBlock, func));
    HIP_CHECK(hipFuncGetAttribute(&shared_bytes,
                                  hipFuncAttributeSharedSizeBytes, func));
    HIP_CHECK(hipFuncGetAttribute(&dynamic_bytes,
                                  hipFuncAttributeMaxDynamicSharedSizeBytes, func));
    if (max_threads <= 0 || shared_bytes != 0 || dynamic_bytes <= 0) {
        fprintf(stderr, "FAILED: function attributes max=%d shared=%d dynamic=%d\n",
                max_threads, shared_bytes, dynamic_bytes);
        return 1;
    }
    if (hipFuncGetAttribute(&shared_bytes, hipFuncAttributeLocalSizeBytes, func) !=
        hipErrorNotSupported) {
        fprintf(stderr, "FAILED: unsupported local-size attribute must be diagnosed\n");
        return 1;
    }

    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    hipEvent_t ev_start = nullptr, ev_stop = nullptr;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));
    HIP_CHECK(hipEventRecord(ev_start, stream));

    vecadd_args_t args = {};
    args.src0 = (dev_ptr_t)(uintptr_t)d_src0;
    args.src1 = (dev_ptr_t)(uintptr_t)d_src1;
    args.dst = (dev_ptr_t)(uintptr_t)d_dst;
    args.num_elements = N;
    void* kernel_params[1] = { &args };

    uint32_t threads = (uint32_t)prop.warpSize;
    uint32_t blocks = (N + threads - 1) / threads;
    HIP_CHECK(hipModuleLaunchKernel(func, blocks, 1, 1, threads, 1, 1,
                                    0, stream, kernel_params, nullptr));

    HIP_CHECK(hipEventRecord(ev_stop, stream));
    HIP_CHECK(hipEventSynchronize(ev_stop));
    float kernel_ms = -1.0f;
    hipError_t elapsed_rc = hipEventElapsedTime(&kernel_ms, ev_start, ev_stop);
    printf("kernel elapsed: rc=%s ms=%.3f\n", hipGetErrorString(elapsed_rc),
           kernel_ms);

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpy(h_dst.data(), d_dst, N * sizeof(int),
                        hipMemcpyDeviceToHost));

    uint32_t errors = 0;
    for (uint32_t i = 0; i < N; ++i) {
        if (h_dst[i] != h_src0[i] + h_src1[i]) {
            if (errors < 4) {
                fprintf(stderr, "mismatch at %u: %d != %d\n", i, h_dst[i],
                        h_src0[i] + h_src1[i]);
            }
            ++errors;
        }
    }

    void* vx_dev_raw = nullptr;
    HIP_CHECK(hipGetVxDevice(&vx_dev_raw));
    if (vx_device_dump_perf((vx_device_h)vx_dev_raw, stdout) != VX_SUCCESS) {
        fprintf(stderr, "FAILED: vx_device_dump_perf\n");
        return 1;
    }

    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipModuleUnload(module));
    HIP_CHECK(hipFree(d_src0));
    HIP_CHECK(hipFree(d_src1));
    HIP_CHECK(hipFree(d_dst));
    HIP_CHECK(hipDeviceReset());

    if (errors == 0) {
        printf("PASSED\n");
        return 0;
    }
    fprintf(stderr, "FAILED (%u errors)\n", errors);
    return 1;
}
