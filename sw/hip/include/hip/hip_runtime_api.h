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

#ifndef HIP_VORTEX_RUNTIME_API_H
#define HIP_VORTEX_RUNTIME_API_H

// libhip_vortex: the host-side HIP API subset for Vortex (plan P2-02).
//
// Every entry maps onto the canonical vortex2.h ABI — there is no private
// backend path in this library. Error codes convert through a single table
// (vx_result_t -> hipError_t); the last error is kept per host thread and
// asynchronous failures surface at the next synchronization point, never
// silently.
//
// Scope of this first cut (see docs/mydocs/p2_02_libhip_vortex.md):
//   supported     device/context, memory, stream, event, memcpy/memset,
//                 module + hipModuleLaunchKernel
//   not supported peer access, graphs, texture/surface, hiprtc,
//                 hipLaunchKernelGGL from the host (needs device-side
//                 trampolines; the hostless headers cover it today)
//
// Kernel argument convention for hipModuleLaunchKernel: Vortex kernel
// entries take a single pointer to an argument block. kernelParams must
// contain exactly one pointer to that block, or use the classic
// HIP_LAUNCH_PARAM_BUFFER_POINTER/SIZE form via `extra`.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum hipError_t {
    hipSuccess = 0,
    hipErrorInvalidValue = 1,
    hipErrorOutOfMemory = 2,
    hipErrorInvalidResourceHandle = 201,
    hipErrorNotInitialized = 401,
    hipErrorNotSupported = 801,
    hipErrorTimeout = 802,
    hipErrorUnknown = 999,
} hipError_t;

typedef enum hipMemcpyKind {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault = 4,
} hipMemcpyKind;

typedef struct _hipDevice_t* hipDevice_t;
typedef struct _hipStream_t* hipStream_t;
typedef struct _hipEvent_t* hipEvent_t;
typedef struct _hipModule_t* hipModule_t;
typedef struct _hipFunction_t* hipFunction_t;

typedef struct hipDeviceProp_t {
    char name[64];
    int multiProcessorCount;   // VX_CAPS_NUM_CORES
    int warpSize;              // VX_CAPS_NUM_THREADS — queried, not assumed
    int clockRate;             // kHz, VX_CAPS_CLOCK_RATE
    size_t totalGlobalMem;     // VX_CAPS_GLOBAL_MEM_SIZE
    size_t sharedMemPerBlock;  // VX_CAPS_LOCAL_MEM_SIZE
    int maxThreadsPerBlock;    // NUM_WARPS * NUM_THREADS
} hipDeviceProp_t;

#define HIP_LAUNCH_PARAM_BUFFER_POINTER ((void*)0x01)
#define HIP_LAUNCH_PARAM_BUFFER_SIZE    ((void*)0x02)

// ---- device / context ----------------------------------------------------

hipError_t hipInit(unsigned int flags);
hipError_t hipGetDeviceCount(int* count);
hipError_t hipSetDevice(int device);
hipError_t hipGetDevice(int* device);
hipError_t hipDeviceGet(hipDevice_t* device, int ordinal);
hipError_t hipDeviceSynchronize(void);
hipError_t hipDeviceReset(void);
hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int device);

// ---- memory --------------------------------------------------------------

hipError_t hipMalloc(void** ptr, size_t size);
hipError_t hipFree(void* ptr);
// Stream-ordered allocator (v1 semantics): the allocation itself is
// conservative — the address is valid immediately after the call — while
// hipFreeAsync defers reuse until the stream's prior work (and the wait
// list) completes, via the runtime's vx_enqueue_free. This is the
// correctness-critical half of the HIP contract (free never overtakes
// in-flight accesses); allocation pooling is a later optimization.
hipError_t hipMallocAsync(void** ptr, size_t size, hipStream_t stream);
hipError_t hipFreeAsync(void* ptr, hipStream_t stream);
hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags);
hipError_t hipHostFree(void* ptr);
hipError_t hipMemcpy(void* dst, const void* src, size_t size, hipMemcpyKind kind);
hipError_t hipMemset(void* dst, int value, size_t size);
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t size,
                          hipMemcpyKind kind, hipStream_t stream);
hipError_t hipMemsetAsync(void* dst, int value, size_t size,
                          hipStream_t stream);

// ---- stream / event --------------------------------------------------------

hipError_t hipStreamCreate(hipStream_t* stream);
hipError_t hipStreamDestroy(hipStream_t stream);
hipError_t hipStreamSynchronize(hipStream_t stream);
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event);

hipError_t hipEventCreate(hipEvent_t* event);
hipError_t hipEventDestroy(hipEvent_t event);
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream);
hipError_t hipEventSynchronize(hipEvent_t event);
hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop);

// ---- vortex2 handles (bridge to the DL libraries) --------------------------
//
// The device DL libraries (sw/dl) speak vortex2.h: their entry points take a
// vx_device_h and a vx_queue_h. Handing them handles that belong to a *second*
// device or queue would break ordering with everything launched through HIP, so
// they are exposed here instead of letting a caller open its own.
//
// The queue is the one the given stream would launch on, so a DL launch lands
// behind the caller's own work on that stream. A null stream means the default
// queue. Unlike hipDeviceGet, which round-trips the device handle through an
// integer type, this says what it is for.

// Both are `void*` in the vortex2 ABI (see vx_device_h in vortex2.h). They are
// spelled as void** here so this header keeps no dependency on vortex2.h.
hipError_t hipGetVxDevice(void** out);
hipError_t hipStreamGetQueue(hipStream_t stream, void** out);

// ---- module / kernel -------------------------------------------------------

hipError_t hipModuleLoad(hipModule_t* module, const char* fname);
hipError_t hipModuleUnload(hipModule_t module);
hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module,
                                const char* kname);
hipError_t hipModuleLaunchKernel(hipFunction_t f,
                                 uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
                                 uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
                                 uint32_t sharedMemBytes, hipStream_t stream,
                                 void** kernelParams, void** extra);

// ---- hiprtc: explicitly NOT implemented (plan P2.3 item 6) ----------------
// Runtime compilation would need a working -x hip mode in VOLT clang
// (currently blocked by AMDGPU-only driver options); every entry returns
// hipErrorNotSupported instead of pretending.

typedef struct _hiprtcProgram* hiprtcProgram;

hipError_t hiprtcCreateProgram(hiprtcProgram* prog, const char* src,
                               const char* name, int numHeaders,
                               const char** headers, const char** includeNames);
hipError_t hiprtcCompileProgram(hiprtcProgram prog, int numOptions,
                                const char** options);
hipError_t hiprtcGetCodeSize(hiprtcProgram prog, size_t* codeSizeRet);
hipError_t hiprtcGetCode(hiprtcProgram prog, char* code);
hipError_t hiprtcDestroyProgram(hiprtcProgram* prog);

// ---- errors ----------------------------------------------------------------

hipError_t hipGetLastError(void);
const char* hipGetErrorString(hipError_t error);
const char* hipGetErrorName(hipError_t error);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // HIP_VORTEX_RUNTIME_API_H
