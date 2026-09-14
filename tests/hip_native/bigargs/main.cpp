// Verifies a >4 KiB argument block survives launch staging intact:
// 1200 inline floats (4800 B) + pointer + scalars = 4816-byte blob.

#include <hip/hip_runtime_api.h>

#include <cstdio>
#include <vector>

#define HIP_CHECK(e) do { hipError_t _e=(e); if(_e!=hipSuccess){ \
    fprintf(stderr,"FAIL %s=%d @%d\n",#e,(int)_e,__LINE__); return 1;} } while(0)

struct BigArgs {
    uint64_t out;
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

    hipModule_t m; hipFunction_t f;
    HIP_CHECK(hipModuleLoad(&m, vxbin));
    HIP_CHECK(hipModuleGetFunction(&f, m, "bigargs_kernel"));

    BigArgs args = {};
    args.out = (uint64_t)(uintptr_t)dout;
    args.n = n;
    args.start = start;
    for (uint32_t i = 0; i < 1200; ++i) args.data[i] = payload[i];
    static_assert(sizeof(BigArgs) == 4816, "unexpected ABI growth");

    void* params[1] = { &args };
    HIP_CHECK(hipModuleLaunchKernel(f, (n + 3) / 4, 1, 1, 4, 1, 1, 0,
                                    nullptr, params, nullptr));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> got(n, -1.0f);
    HIP_CHECK(hipMemcpy(got.data(), dout, n * 4, hipMemcpyDeviceToHost));
    uint32_t bad = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (got[i] != payload[start + i]) ++bad;
    }
    printf("bigargs(4816B blob, tail@%u): bad=%u last=%.2f\n", start, bad,
           (double)got[n - 1]);
    HIP_CHECK(hipFree(dout));
    HIP_CHECK(hipModuleUnload(m));
    HIP_CHECK(hipDeviceReset());
    if (bad == 0) { printf("PASSED\n"); return 0; }
    fprintf(stderr, "FAILED\n"); return 1;
}
