// Triton-shaped kernel for the triton-vortex driver launch test:
// the exact add_kernel signature the interpreter version uses
// (pointers + n; BLOCK folded into the grid computation on the host).
#include <hip/hip_runtime.h>
#include <stdint.h>

struct args_t {
    uintptr_t x;
    uintptr_t y;
    uintptr_t out;
    uint32_t n;
    uint32_t pad;
};

__global__ void add_kernel(args_t* arg) {
    // One CTA per BLOCK=16 chunk, 16 threads: thread i handles element
    // base+i*1 (a Triton BLOCK is one program; we map it to one CTA here).
    uint32_t pid = blockIdx.x;
    uint32_t block = 16;
    for (uint32_t i = threadIdx.x; i < block; i += blockDim.x) {
        uint32_t off = pid * block + i;
        if (off < arg->n) {
            float* out = (float*)(uintptr_t)arg->out;
            out[off] = ((const float*)(uintptr_t)arg->x)[off] +
                       ((const float*)(uintptr_t)arg->y)[off];
        }
    }
}
