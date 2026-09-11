// KMU-mode kernel image for the stream overlap test (plan P2-03).
// One scaling kernel reused by both streams on different buffers.

#include <hip/hip_runtime.h>
#include <stdint.h>

struct scale_args_t {
    uintptr_t in;
    uintptr_t out;
    uint32_t n;
    uint32_t scale;
    uint32_t pad;
};

__global__ void scale_kernel(scale_args_t* arg) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < arg->n) {
        const int* in = (const int*)(uintptr_t)arg->in;
        int* out = (int*)(uintptr_t)arg->out;
        out[i] = in[i] * (int)arg->scale;
    }
}
