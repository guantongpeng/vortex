#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#define HIP_CHECK(e) do { hipError_t _e=(e); if(_e!=hipSuccess){ \
    fprintf(stderr,"FAIL %s=%d @%d\n",#e,(int)_e,__LINE__); return 1;} } while(0)

#if HIP_TEST_DEV_PTR_WIDTH == 32
using dev_ptr_t = uint32_t;
#else
using dev_ptr_t = uint64_t;
#endif

struct BigArgs {
    dev_ptr_t out;
    uint32_t n;
    uint32_t start;
    float data[1200];
};

int main(int argc, char** argv) {
    const char* vxbin = argc > 1 ? argv[1] : "kernel.vxbin";
    HIP_CHECK(hipInit(0));

    const uint32_t n = 256, start = 900;  // tail slice: 900..1155
    std::vector<float> payload(1200);
    for (uint32_t i = 0; i < payload.size(); ++i) payload[i] = (float)(i * 7) * 0.25f;

    void* dout = nullptr;
    HIP_CHECK(hipMalloc(&dout, n * 4));
    HIP_CHECK(hipMemset(dout, 0, n * sizeof(float)));

    hipModule_t m; hipFunction_t f;
    HIP_CHECK(hipModuleLoad(&m, vxbin));
    HIP_CHECK(hipModuleGetFunction(&f, m, "bigargs_kernel"));

    BigArgs args = {};
    args.out = (dev_ptr_t)(uintptr_t)dout;
    args.n = n;
    args.start = start;
    for (uint32_t i = 0; i < 1200; ++i) args.data[i] = payload[i];
    static_assert(sizeof(BigArgs) == 4808 + sizeof(dev_ptr_t), "unexpected ABI growth");

    void* params[1] = { &args };
    hipError_t launch = hipModuleLaunchKernel(f, (n + 3) / 4, 1, 1, 4, 1, 1, 0,
                                              nullptr, params, nullptr);
    if (launch != hipErrorInvalidValue) {
        fprintf(stderr, "FAIL oversized launch returned %d, expected %d\n",
                (int)launch, (int)hipErrorInvalidValue);
        hipModuleUnload(m);
        hipFree(dout);
        hipDeviceReset();
        return 1;
    }
    // Exercise the explicit HIP buffer/size ABI as well. This second launch
    // isolates the public 4 KiB runtime limit from image metadata validation.
    void* extra[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        reinterpret_cast<void*>(static_cast<uintptr_t>(4097)),
        nullptr,
    };
    launch = hipModuleLaunchKernel(f, (n + 3) / 4, 1, 1, 4, 1, 1, 0,
                                   nullptr, nullptr, extra);
    if (launch != hipErrorInvalidValue) {
        fprintf(stderr, "FAIL explicit 4097B launch returned %d, expected %d\n",
                (int)launch, (int)hipErrorInvalidValue);
        hipModuleUnload(m);
        hipFree(dout);
        hipDeviceReset();
        return 1;
    }
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> got(n, -1.0f);
    HIP_CHECK(hipMemcpy(got.data(), dout, n * sizeof(float), hipMemcpyDeviceToHost));
    for (uint32_t i = 0; i < n; ++i) {
        if (got[i] != 0.0f) {
            fprintf(stderr, "FAIL rejected kernel changed output[%u] to %.2f\n",
                    i, (double)got[i]);
            return 1;
        }
    }
    printf("bigargs(%zuB blob): rejected, output unchanged\n", sizeof(args));
    HIP_CHECK(hipFree(dout));
    HIP_CHECK(hipModuleUnload(m));
    HIP_CHECK(hipDeviceReset());
    printf("PASSED\n");
    return 0;
}
