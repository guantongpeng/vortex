#include <hip/hip_runtime.h>
#include <stdint.h>

struct big_args_t {
    uintptr_t out;
    uint32_t n;
    uint32_t start;
    float data[1200];
};

__global__ void bigargs_kernel(big_args_t* arg) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < arg->n;
         i += blockDim.x * gridDim.x) {
        float* out = (float*)(uintptr_t)arg->out;
        out[i] = arg->data[arg->start + i];
    }
}
