// KMU-mode HIP kernel image for the libhip_vortex module API test.
//
// Compiled by hipcc-vortex with --kernel-lib=vortex2 (-DHIP_VORTEX_KMU):
// there is no device main; a host runtime (libhip_vortex) loads the .vxbin,
// resolves the kernel by name and launches it through vx_enqueue_launch.
// CTA threads read their identity from the KMU CSRs, and the kernel takes
// the single-pointer argument-block ABI (pointers are device addresses
// written into the block by the host).

#include <hip/hip_runtime.h>
#include <stdint.h>

struct vecadd_args_t {
    uintptr_t src0;
    uintptr_t src1;
    uintptr_t dst;
    uint32_t num_elements;
    uint32_t pad;
};

__global__ void vecadd_kernel(vecadd_args_t* arg) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < arg->num_elements) {
        const int* src0 = (const int*)(uintptr_t)arg->src0;
        const int* src1 = (const int*)(uintptr_t)arg->src1;
        int* dst = (int*)(uintptr_t)arg->dst;
        dst[i] = src0[i] + src1[i];
    }
}
