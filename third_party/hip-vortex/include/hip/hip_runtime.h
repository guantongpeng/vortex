// Copyright © 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef HIP_VORTEX_RUNTIME_H
#define HIP_VORTEX_RUNTIME_H

// Native HIPVortex device-side runtime, first milestone.
//
// Compilation model: the whole translation unit (kernels and the main()
// that launches them) is compiled by hipcc-vortex as RISC-V C++ for the
// Vortex device and linked into a .vxbin. This is the repository's
// "hostless" execution model: main() runs on the device itself, and the
// HIP host-side APIs below are device-local shims, not a host runtime.
//
// Mapping (see docs/mydocs/02_hip/hip_p2_03_hip_headers.md):
//   __global__/threadIdx/blockIdx/blockDim/gridDim -> vx_spawn.h
//   __syncthreads                                -> vx_spawn.h barrier
//   __shfl_*/__ballot/__all/__any                -> vx_vote_*/vx_shfl_*
//   atomicAdd/Max/Exch/CAS                       -> RVA __atomic builtins
//   hipLaunchKernelGGL                           -> arg pack + vx_spawn_threads
//
// Not supported in this milestone (fail loudly, never silently):
//   triple-chevron kernel<<<grid,block>>>(...) syntax: requires a clang
//     driver that rewrites it; use hipLaunchKernelGGL.
//   static __shared__ declarations: use hipVortexLocalMem(bytes) with a
//     compile-time tile size (the legacy spawn model sizes local memory
//     per kernel, not per launch).
//   __syncthreads() for multi-warp blocks (hostless mode): the legacy
//     spawn model addresses barriers by local_group_id, which is 0 for
//     every group when blocks span multiple warps, and the spawn join
//     uses the same barrier — measured outcomes range from deadlock to
//     lost local-memory updates. Use single-warp blocks here, and the
//     KMU path (hipcc --kernel-lib=vortex2 + host launch) for real
//     multi-warp CTAs.
//   hipStream/hipEvent semantics: the parameters exist for source
//     compatibility but the hostless model has a single implicit stream.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <vx_intrinsics.h>
#include <vx_print.h>

// Two execution models, selected at image build time:
//
//   default (hostless): main() runs on the device; hipLaunchKernelGGL
//     spawns through vx_spawn_threads and threadIdx/blockIdx/blockDim/
//     gridDim are the TLS variables maintained by libvortex.a.
//
//   HIP_VORTEX_KMU: the image is a module launched by a host runtime
//     (libhip_vortex -> vx_enqueue_launch). CTA identity comes from the
//     KMU CSRs (vx_spawn2.h), there is no device main, and the host-side
//     shims (hipMalloc/hipLaunchKernelGGL/...) are unavailable. Kernels
//     use the single-pointer argument-block ABI.
#ifdef HIP_VORTEX_KMU
#include <vx_spawn2.h>
#else
#include <vx_spawn.h>
#endif

#include <hip/hip_vector_types.h>

// picolibc's printf lazily allocates its stdio buffer through _sbrk, whose
// device stub executes ebreak — an untested path that hangs the machine.
// Device code therefore prints through tinyprintf's vx_printf (MMIO,
// allocation-free). tinyprintf covers %d/%u/%x/%f/%p/%s/%c, which is the
// documented device subset.
#undef printf
#define printf vx_printf

// ---------------------------------------------------------------------------
// Execution-space macros (x/y/z index variables come from vx_spawn.h)
// ---------------------------------------------------------------------------

// __global__/__host__/__device__ are defined by hip_vector_types.h (the
// same guard covers both headers); only __forceinline__ is runtime-only.
#ifndef __forceinline__
#define __forceinline__ inline __attribute__((always_inline))
#endif

#define __HIP_VORTEX__ 1
#define HIP_VORTEX_MAJOR 0
#define HIP_VORTEX_MINOR 1

// warpSize is a device property on Vortex (num threads per warp is a
// configurable knob), so it must be queried, never assumed to be 32.
#define warpSize (vx_num_threads())

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

typedef enum hipError_t {
    hipSuccess = 0,
    hipErrorMemoryAllocation = 2,
    hipErrorInvalidValue = 16,
    hipErrorNotSupported = 801,
    hipErrorUnknown = 999,
} hipError_t;

static __thread int __hip_last_error = hipSuccess;

static inline hipError_t hipGetLastError(void) {
    hipError_t e = (hipError_t)__hip_last_error;
    __hip_last_error = hipSuccess;
    return e;
}

static inline const char* hipGetErrorString(hipError_t e) {
    switch (e) {
    case hipSuccess: return "hipSuccess";
    case hipErrorMemoryAllocation: return "hipErrorMemoryAllocation";
    case hipErrorInvalidValue: return "hipErrorInvalidValue";
    case hipErrorNotSupported: return "hipErrorNotSupported";
    default: return "hipErrorUnknown";
    }
}

#define HIP_ASSERT(expr) ((expr) ? (void)0 : (void)0)
#define hipCheck(expr) do { \
    hipError_t _e = (expr); \
    if (_e != hipSuccess) { \
        printf("HIP error %s at %s:%d\n", hipGetErrorString(_e), __FILE__, __LINE__); \
    } \
} while (0)

// ---------------------------------------------------------------------------
// Streams/events: source-compatibility typedefs only (single implicit stream)
// ---------------------------------------------------------------------------

typedef struct ihipStream_t* hipStream_t;
typedef struct ihipEvent_t* hipEvent_t;

// ---------------------------------------------------------------------------
// Device-local memory pool (hostless hipMalloc)
// ---------------------------------------------------------------------------

#ifndef HIP_VORTEX_KMU

#ifndef HIP_VORTEX_HEAP_SIZE
#define HIP_VORTEX_HEAP_SIZE (1024 * 1024)
#endif

extern "C" {
// weak + used so multi-TU programs keep exactly one copy and --gc-sections
// cannot drop it before hipMalloc's first reference.
__attribute__((weak, used)) char __hip_heap[HIP_VORTEX_HEAP_SIZE];
__attribute__((weak, used)) unsigned long __hip_heap_offset;
}

static inline void* hipMalloc(size_t size) {
    size = (size + 7u) & ~(size_t)7u;
    unsigned long base = __hip_heap_offset;
    if (base + size > (unsigned long)HIP_VORTEX_HEAP_SIZE) {
        __hip_last_error = hipErrorMemoryAllocation;
        return NULL;
    }
    __hip_heap_offset = base + size;
    return (void*)(__hip_heap + base);
}

static inline hipError_t hipFree(void* p) {
    (void)p; // bump allocator: memory is reclaimed at kernel-image unload
    return hipSuccess;
}

typedef enum hipMemcpyKind {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault = 4,
} hipMemcpyKind;

static inline hipError_t hipMemcpy(void* dst, const void* src, size_t size,
                                   hipMemcpyKind kind) {
    // Hostless: one address space, so every kind is a local copy.
    (void)kind;
    memcpy(dst, src, size);
    vx_fence();
    return hipSuccess;
}

static inline hipError_t hipMemset(void* dst, int value, size_t size) {
    memset(dst, value, size);
    vx_fence();
    return hipSuccess;
}

static inline hipError_t hipDeviceSynchronize(void) {
    vx_fence();
    return hipSuccess;
}

#endif // !HIP_VORTEX_KMU

// ---------------------------------------------------------------------------
// Kernel launch (hostless only — KMU images are launched by the host)
// ---------------------------------------------------------------------------

#ifndef HIP_VORTEX_KMU

namespace hip_vortex {
namespace detail {

// Minimal recursive argument pack: a header-only stand-in for std::tuple
// that only relies on picolibc, so device code does not need a hosted
// C++ standard library.
template <typename... Ts> struct ArgPack;

template <> struct ArgPack<> {};

template <typename T, typename... Rest>
struct ArgPack<T, Rest...> : ArgPack<Rest...> {
    T head;
    __device__ ArgPack(T h, Rest... rest) : ArgPack<Rest...>(rest...), head(h) {}
};

template <size_t... Is> struct IndexList {};
template <size_t N, size_t... Is> struct BuildIndexList;
template <size_t... Is> struct BuildIndexList<0, Is...> { using type = IndexList<Is...>; };
template <size_t N, size_t... Is> struct BuildIndexList : BuildIndexList<N - 1, N - 1, Is...> {};

// PackGet<I, ...> resolves by inheritance so get() returns the I-th type:
// PackGet<I, T, Rest...> : PackGet<I-1, Rest...> : ... : PackGet<0, Ti, ...>.
// The parameter type narrows to the matching sub-pack, which the full pack
// derives from, so derived-to-base reference binding does the rest.
template <size_t I, typename T, typename... Rest>
struct PackGet : PackGet<I - 1, Rest...> {};

template <typename T, typename... Rest>
struct PackGet<0, T, Rest...> {
    static __device__ T get(ArgPack<T, Rest...>& p) { return p.head; }
};

template <typename Fn, typename... Args>
struct LaunchCtx {
    Fn fn;
    ArgPack<Args...> args;
};

template <typename Fn, typename... Args, size_t... Is>
static __device__ void invoke_packed(Fn fn, ArgPack<Args...>& p, IndexList<Is...>) {
    // Sizeof... style expansion over the pack; each PackGet matches the
    // positional argument of fn.
    fn(PackGet<Is, Args...>::get(p)...);
}

template <typename Fn, typename... Args>
static void __attribute__((used)) launch_trampoline(void* vctx) {
    LaunchCtx<Fn, Args...>* ctx = static_cast<LaunchCtx<Fn, Args...>*>(vctx);
    typename BuildIndexList<sizeof...(Args)>::type indices;
    invoke_packed(ctx->fn, ctx->args, indices);
}

} // namespace detail
} // namespace hip_vortex

// HIP's portable launch API. grid/block may be dim3 or scalars; sharedMem
// and stream are accepted for source compatibility and ignored (see the
// header comment for the local-memory story).
template <typename Fn, typename... Args>
static inline void hipLaunchKernelGGL(Fn kernel, dim3 grid, dim3 block,
                                      size_t sharedMem, hipStream_t stream,
                                      Args... args) {
    (void)sharedMem;
    (void)stream;
    hip_vortex::detail::LaunchCtx<Fn, Args...> ctx{
        kernel, hip_vortex::detail::ArgPack<Args...>(args...)};
    uint32_t gd[3] = {grid.x, grid.y, grid.z};
    uint32_t bd[3] = {block.x, block.y, block.z};
    vx_spawn_threads(3, gd, bd,
                     (vx_kernel_func_cb)&hip_vortex::detail::launch_trampoline<Fn, Args...>,
                     &ctx);
}

#endif // !HIP_VORTEX_KMU

// ---------------------------------------------------------------------------
// Warp intrinsics (vx_vote_* / vx_shfl_* custom-0 space)
// ---------------------------------------------------------------------------

namespace hip_vortex {
namespace detail {

static inline __device__ size_t shfl_down_bits(size_t value, int delta,
                                               int width) {
    int clamp = width - 1;
    int segmask = ~clamp & 0x3f;
    return vx_shfl_down(value, delta, clamp, segmask);
}

static inline __device__ size_t shfl_up_bits(size_t value, int delta,
                                             int width) {
    int clamp = width - 1;
    int segmask = ~clamp & 0x3f;
    return vx_shfl_up(value, delta, clamp, segmask);
}

static inline __device__ size_t shfl_xor_bits(size_t value, int lane_mask,
                                              int width) {
    int clamp = width - 1;
    int segmask = ~clamp & 0x3f;
    return vx_shfl_bfly(value, lane_mask, clamp, segmask);
}

template <typename T>
static inline __device__ T bitcast_copy(size_t bits) {
    T out;
    __builtin_memcpy(&out, &bits, sizeof(T));
    return out;
}

template <typename T>
static inline __device__ size_t to_bits(const T& value) {
    size_t bits = 0;
    __builtin_memcpy(&bits, &value, sizeof(T));
    return bits;
}

} // namespace detail
} // namespace hip_vortex

template <typename T>
static inline __device__ T __shfl_down(T var, int delta, int width) {
    return hip_vortex::detail::bitcast_copy<T>(
        hip_vortex::detail::shfl_down_bits(hip_vortex::detail::to_bits(var), delta, width));
}

template <typename T>
static inline __device__ T __shfl_up(T var, int delta, int width) {
    return hip_vortex::detail::bitcast_copy<T>(
        hip_vortex::detail::shfl_up_bits(hip_vortex::detail::to_bits(var), delta, width));
}

template <typename T>
static inline __device__ T __shfl_xor(T var, int lane_mask, int width) {
    return hip_vortex::detail::bitcast_copy<T>(
        hip_vortex::detail::shfl_xor_bits(hip_vortex::detail::to_bits(var), lane_mask, width));
}

static inline __device__ unsigned long long __ballot(int predicate) {
    return (unsigned long long)vx_vote_ballot(predicate);
}

static inline __device__ int __all(int predicate) {
    return vx_vote_all(predicate) ? 1 : 0;
}

static inline __device__ int __any(int predicate) {
    return vx_vote_any(predicate) ? 1 : 0;
}

static inline __device__ void __syncwarp(int mask) {
    (void)mask;
    vx_wsync();
}

// ---------------------------------------------------------------------------
// Atomics (RVA extension; requires -DVX_CFG_EXT_A_ENABLE configs)
// ---------------------------------------------------------------------------

static inline __device__ int atomicAdd(int* ptr, int val) {
    return __atomic_fetch_add(ptr, val, __ATOMIC_SEQ_CST);
}

static inline __device__ unsigned int atomicAdd(unsigned int* ptr, unsigned int val) {
    return __atomic_fetch_add(ptr, val, __ATOMIC_SEQ_CST);
}

static inline __device__ long atomicAdd(long* ptr, long val) {
    return __atomic_fetch_add(ptr, val, __ATOMIC_SEQ_CST);
}

static inline __device__ unsigned long long atomicAdd(unsigned long long* ptr,
                                                      unsigned long long val) {
    return __atomic_fetch_add(ptr, val, __ATOMIC_SEQ_CST);
}

static inline __device__ float atomicAdd(float* ptr, float val) {
    // No hardware FP amo: CAS loop on the bit pattern. NOTE: without ZACAS
    // the compare-exchange lowers to lr/sc, and lanes contending on the
    // same address invalidate each other's reservations — measured
    // livelock. Only use this when at most one lane per address contends,
    // or on ZACAS-enabled hardware (true single-address CAS).
    uint32_t expected = (uint32_t)hip_vortex::detail::to_bits(*ptr);
    uint32_t assumed;
    do {
        assumed = expected;
        float sum = hip_vortex::detail::bitcast_copy<float>(assumed) + val;
        uint32_t sum_bits = (uint32_t)hip_vortex::detail::to_bits(sum);
        uint32_t cmp = assumed;
        __atomic_compare_exchange_n((uint32_t*)ptr, &cmp, sum_bits, false,
                                    __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
        expected = cmp;
    } while (expected != assumed);
    return hip_vortex::detail::bitcast_copy<float>(expected);
}

static inline __device__ int atomicMax(int* ptr, int val) {
    return __atomic_fetch_max(ptr, val, __ATOMIC_SEQ_CST);
}

static inline __device__ unsigned int atomicMax(unsigned int* ptr, unsigned int val) {
    return __atomic_fetch_max(ptr, val, __ATOMIC_SEQ_CST);
}

template <typename T>
static inline __device__ T atomicExch(T* ptr, T val) {
    T old;
    __atomic_exchange(ptr, &val, &old, __ATOMIC_SEQ_CST);
    return old;
}

static inline __device__ int atomicCAS(int* ptr, int expected, int desired) {
    __atomic_compare_exchange_n(ptr, &expected, desired, false,
                                __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return expected;
}

static inline __device__ unsigned int atomicCAS(unsigned int* ptr,
                                                unsigned int expected,
                                                unsigned int desired) {
    __atomic_compare_exchange_n(ptr, &expected, desired, false,
                                __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return expected;
}

static inline __device__ unsigned long long atomicCAS(unsigned long long* ptr,
                                                      unsigned long long expected,
                                                      unsigned long long desired) {
    __atomic_compare_exchange_n(ptr, &expected, desired, false,
                                __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return expected;
}

// ---------------------------------------------------------------------------
// Math (picolibm via -lm; no SFU on current configs)
// ---------------------------------------------------------------------------

static inline __device__ float __expf(float x) { return expf(x); }
static inline __device__ float __logf(float x) { return logf(x); }
static inline __device__ float __log2f(float x) { return log2f(x); }
static inline __device__ float __log10f(float x) { return log10f(x); }
static inline __device__ float __powf(float x, float y) { return powf(x, y); }
static inline __device__ float __fdividef(float x, float y) { return x / y; }
static inline __device__ float rsqrtf(float x) { return 1.0f / sqrtf(x); }

// ---------------------------------------------------------------------------
// Device properties (from CSRs, never from build-time assumptions)
// ---------------------------------------------------------------------------

static inline int hipGetDeviceCount(int* count) {
    if (count) *count = 1;
    return hipSuccess;
}

#endif // HIP_VORTEX_RUNTIME_H
