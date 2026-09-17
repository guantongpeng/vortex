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

// libhip_vortex implementation: HIP device/memory/stream/event/module subset
// over the canonical vortex2.h ABI. No private backend path.

#include "hip/hip_runtime_api.h"

#include <vortex2.h>

#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Error conversion and per-thread last error
// ---------------------------------------------------------------------------

static thread_local hipError_t g_last_error = hipSuccess;

static hipError_t hip_error_from_vx(vx_result_t r) {
    switch (r) {
    case VX_SUCCESS:            return hipSuccess;
    case VX_ERR_INVALID_HANDLE: return hipErrorInvalidResourceHandle;
    case VX_ERR_INVALID_INFO:   return hipErrorInvalidValue;
    case VX_ERR_INVALID_VALUE:  return hipErrorInvalidValue;
    case VX_ERR_OUT_OF_HOST_MEMORY:
    case VX_ERR_OUT_OF_DEVICE_MEMORY:
                                return hipErrorOutOfMemory;
    case VX_ERR_TIMEOUT:        return hipErrorTimeout;
    case VX_ERR_EVENT_FAILED:   return hipErrorUnknown;
    case VX_ERR_NOT_SUPPORTED:  return hipErrorNotSupported;
    case VX_ERR_DEVICE_LOST:
    case VX_ERR_INTERNAL:
    default:                    return hipErrorUnknown;
    }
}

// CHECK: convert, remember, and pass through. Every public entry wraps its
// vortex2.h calls with this so hipGetLastError is always meaningful.
static hipError_t RET(hipError_t e) {
    if (e != hipSuccess) g_last_error = e;
    return e;
}

static hipError_t RET_VX(vx_result_t r) {
    return RET(hip_error_from_vx(r));
}

// ---------------------------------------------------------------------------
// Primary context: one device per process (v1)
// ---------------------------------------------------------------------------

namespace {

struct DeviceState {
    vx_device_h dev = nullptr;
    bool initialized = false;
    int current_device = 0;
};

struct StreamState {
    vx_queue_h q;
};

struct EventState {
    vx_event_h ev;              // timeline counter event
    vx_event_h done = nullptr;  // completion event of the last record signal
    uint64_t last_value = 0;    // value of the most recent record
    uint64_t last_ns = 0;       // completion time of the most recent record
};

struct ModuleState {
    vx_module_h mod;
};

struct FuncState {
    vx_kernel_h k;
};

struct BufferRecord {
    vx_buffer_h buf;
    uint64_t addr;
    uint64_t size;
    void* mapped_host = nullptr;  // non-null for host allocations
};

DeviceState g_device;
std::mutex g_mutex;
std::unordered_map<uint64_t, BufferRecord> g_by_addr;       // device addr -> record
std::unordered_map<void*, uint64_t> g_host_to_addr;         // mapped host ptr -> device addr
vx_queue_h g_default_queue = nullptr;

// Profiling is enabled on every HIP queue so hipEventElapsedTime has
// timestamps; the cost is per-command bookkeeping only.
vx_result_t queue_create_profiled(vx_device_h dev, vx_queue_h* out) {
    vx_queue_info_t qi = {};
    qi.struct_size = sizeof(qi);
    qi.flags = VX_QUEUE_PROFILING_ENABLE;
    return vx_queue_create(dev, &qi, out);
}

// Every queue this process has handed out, so hipDeviceSynchronize can be a
// device-wide barrier rather than a default-queue one. Guarded by g_mutex.
std::vector<vx_queue_h> g_live_queues;

vx_queue_h default_queue() {
    // Function-local static: the previous lazy init was an unsynchronized
    // check-then-set, and the c10 allocator's deleter can run this on any
    // thread. C++11 guarantees the initialisation is done exactly once.
    static vx_queue_h q = [] {
        vx_queue_h created = nullptr;
        queue_create_profiled(g_device.dev, &created);
        return created;
    }();
    g_default_queue = q;
    return q;
}

vx_queue_h stream_queue(hipStream_t s) {
    return s ? ((StreamState*)s)->q : default_queue();
}

void register_queue(vx_queue_h q) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_live_queues.push_back(q);
}

void unregister_queue(vx_queue_h q) {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (size_t i = 0; i < g_live_queues.size(); ++i) {
        if (g_live_queues[i] == q) {
            g_live_queues.erase(g_live_queues.begin() + i);
            return;
        }
    }
}

// Device-wide barrier: finish the default queue and every queue still alive.
vx_result_t finish_all_queues() {
    std::vector<vx_queue_h> queues;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        queues = g_live_queues;
        queues.push_back(default_queue());
    }
    for (vx_queue_h q : queues) {
        vx_result_t r = vx_queue_finish(q, VX_TIMEOUT_INFINITE);
        if (r != VX_SUCCESS) return r;
    }
    return VX_SUCCESS;
}

const BufferRecord* find_buffer(uint64_t addr) {
    auto it = g_by_addr.find(addr);
    return it == g_by_addr.end() ? nullptr : &it->second;
}

BufferRecord* find_buffer_mut(uint64_t addr) {
    auto it = g_by_addr.find(addr);
    return it == g_by_addr.end() ? nullptr : &it->second;
}

// Wait for a freshly enqueued command so the *_sync APIs are synchronous.
hipError_t wait_last(vx_result_t r, vx_event_h ev) {
    if (r != VX_SUCCESS) return RET_VX(r);
    if (ev) {
        hipError_t e = RET_VX(vx_event_wait_value(ev, 1, UINT64_MAX));
        vx_event_release(ev);
        return e;
    }
    return hipSuccess;
}

uint64_t caps(uint32_t id) {
    uint64_t v = 0;
    vx_device_query(g_device.dev, id, &v);
    return v;
}

} // namespace

// ---------------------------------------------------------------------------
// device / context
// ---------------------------------------------------------------------------

hipError_t hipInit(unsigned int flags) {
    (void)flags;
    uint32_t count = 0;
    vx_result_t r = vx_device_count(&count);
    if (r != VX_SUCCESS) return RET_VX(r);
    if (count == 0) return RET(hipErrorNotInitialized);
    return hipSetDevice(0);
}

hipError_t hipGetDeviceCount(int* count) {
    if (!count) return RET(hipErrorInvalidValue);
    uint32_t n = 0;
    vx_result_t r = vx_device_count(&n);
    *count = (int)n;
    return RET_VX(r);
}

hipError_t hipSetDevice(int device) {
    if (device != 0) return RET(hipErrorInvalidValue);  // single device (v1)
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_device.initialized) {
        vx_result_t r = vx_device_open((uint32_t)device, &g_device.dev);
        if (r != VX_SUCCESS) return RET_VX(r);
        g_device.initialized = true;
    }
    g_device.current_device = device;
    return hipSuccess;
}

hipError_t hipGetDevice(int* device) {
    if (!device) return RET(hipErrorInvalidValue);
    *device = g_device.current_device;
    return hipSuccess;
}

hipError_t hipDeviceGet(hipDevice_t* device, int ordinal) {
    if (!device || ordinal != 0) return RET(hipErrorInvalidValue);
    hipError_t e = hipSetDevice(ordinal);
    if (e != hipSuccess) return e;
    *device = (hipDevice_t)g_device.dev;
    return hipSuccess;
}

hipError_t hipDeviceSynchronize(void) {
    if (!g_device.initialized) return RET(hipErrorNotInitialized);
    // vx_queue_finish, not vx_queue_flush: flush only wakes the worker so it
    // notices newly queued commands, it does not wait for anything. Mapping
    // synchronize onto it made hipDeviceSynchronize return success having
    // waited for nothing, which every caller above reads as "results are
    // ready". The barrier is an event enqueued behind the pending work.
    //
    // VX_TIMEOUT_INFINITE, not 0: a zero timeout is a poll (wait_for(0ns)),
    // not "wait forever" -- Event::wait_value returns VX_ERR_TIMEOUT at once.
    //
    // Device-wide, not just the default queue: work launched on an explicit
    // stream has to be covered too, which is what the name promises.
    return RET_VX(finish_all_queues());
}

hipError_t hipDeviceReset(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_device.initialized) return hipSuccess;
    for (auto& kv : g_by_addr) {
        if (kv.second.mapped_host) {
            vx_buffer_unmap(kv.second.buf, kv.second.mapped_host);
        }
        vx_buffer_release(kv.second.buf);
    }
    g_by_addr.clear();
    g_host_to_addr.clear();
    if (g_default_queue) {
        vx_queue_release(g_default_queue);
        g_default_queue = nullptr;
    }
    vx_device_release(g_device.dev);
    g_device = DeviceState{};
    return hipSuccess;
}

hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int device) {
    if (!prop || device != 0) return RET(hipErrorInvalidValue);
    hipError_t e = hipSetDevice(device);
    if (e != hipSuccess) return e;
    memset(prop, 0, sizeof(*prop));
    strncpy(prop->name, "vortex", sizeof(prop->name) - 1);
    prop->multiProcessorCount = (int)caps(VX_CAPS_NUM_CORES);
    prop->warpSize = (int)caps(VX_CAPS_NUM_THREADS);
    prop->clockRate = (int)(caps(VX_CAPS_CLOCK_RATE) * 1000);  // MHz -> kHz
    prop->totalGlobalMem = caps(VX_CAPS_GLOBAL_MEM_SIZE);
    prop->sharedMemPerBlock = caps(VX_CAPS_LOCAL_MEM_SIZE);
    prop->maxThreadsPerBlock =
        (int)(caps(VX_CAPS_NUM_WARPS) * caps(VX_CAPS_NUM_THREADS));
    return hipSuccess;
}

// ---------------------------------------------------------------------------
// memory
// ---------------------------------------------------------------------------

hipError_t hipMalloc(void** ptr, size_t size) {
    if (!ptr || size == 0) return RET(hipErrorInvalidValue);
    hipError_t e = hipSetDevice(0);
    if (e != hipSuccess) return e;
    vx_buffer_h buf = nullptr;
    vx_result_t r = vx_buffer_create(g_device.dev, (uint64_t)size, 0, &buf);
    if (r != VX_SUCCESS) return RET_VX(r);
    uint64_t addr = 0;
    r = vx_buffer_address(buf, &addr);
    if (r != VX_SUCCESS) {
        vx_buffer_release(buf);
        return RET_VX(r);
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    g_by_addr[addr] = BufferRecord{buf, addr, (uint64_t)size, nullptr};
    *ptr = (void*)(uintptr_t)addr;
    return hipSuccess;
}

hipError_t hipFree(void* ptr) {
    if (!ptr) return hipSuccess;
    uint64_t addr = (uint64_t)(uintptr_t)ptr;
    std::lock_guard<std::mutex> lock(g_mutex);
    BufferRecord* rec = find_buffer_mut(addr);
    if (!rec) return RET(hipErrorInvalidResourceHandle);
    if (rec->mapped_host) vx_buffer_unmap(rec->buf, rec->mapped_host);
    vx_result_t r = vx_buffer_release(rec->buf);
    g_by_addr.erase(addr);
    return RET_VX(r);
}

hipError_t hipMallocAsync(void** ptr, size_t size, hipStream_t stream) {
    // v1: eager allocation (address valid immediately — conservative and
    // correct); pooling is deferred until a real stream-ordered malloc
    // primitive exists in the runtime. See hip_runtime_api.h.
    (void)stream;
    return hipMalloc(ptr, size);
}

hipError_t hipFreeAsync(void* ptr, hipStream_t stream) {
    if (!ptr) return hipSuccess;
    uint64_t addr = (uint64_t)(uintptr_t)ptr;
    vx_buffer_h buf = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        BufferRecord* rec = find_buffer_mut(addr);
        if (!rec) return RET(hipErrorInvalidResourceHandle);
        if (rec->mapped_host) {
            vx_buffer_unmap(rec->buf, rec->mapped_host);
            rec->mapped_host = nullptr;
        }
        buf = rec->buf;
        rec->buf = nullptr;  // hand the handle to the queued free
        g_by_addr.erase(addr);
    }
    // Reuse is gated on the stream's prior commands: vx_enqueue_free
    // retains the buffer until then (P1-01 primitive).
    vx_event_h ev = nullptr;
    vx_result_t r = vx_enqueue_free(stream_queue(stream), buf, 0, nullptr, &ev);
    if (r != VX_SUCCESS) {
        vx_buffer_release(buf);
        return RET_VX(r);
    }
    if (ev) vx_event_release(ev);
    // Drop the caller's reference now that the queue holds its own
    // (Queue::enqueue_free retains). Without this the buffer is never
    // destroyed: the queued release takes the refcount from 2 back to 1 and
    // nothing ever takes it to 0, so every hipFreeAsync leaked.
    vx_buffer_release(buf);
    return RET_VX(r);
}

hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags) {
    (void)flags;
    if (!ptr || size == 0) return RET(hipErrorInvalidValue);
    hipError_t e = hipSetDevice(0);
    if (e != hipSuccess) return e;
    vx_buffer_h buf = nullptr;
    vx_result_t r = vx_buffer_create(g_device.dev, (uint64_t)size, 0, &buf);
    if (r != VX_SUCCESS) return RET_VX(r);
    uint64_t addr = 0;
    r = vx_buffer_address(buf, &addr);
    if (r != VX_SUCCESS) {
        vx_buffer_release(buf);
        return RET_VX(r);
    }
    void* host = nullptr;
    r = vx_buffer_map(buf, 0, (uint64_t)size, 0, &host);
    if (r != VX_SUCCESS) {
        vx_buffer_release(buf);
        return RET_VX(r);
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    g_by_addr[addr] = BufferRecord{buf, addr, (uint64_t)size, host};
    g_host_to_addr[host] = addr;
    *ptr = host;
    return hipSuccess;
}

hipError_t hipHostFree(void* ptr) {
    if (!ptr) return hipSuccess;
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_host_to_addr.find(ptr);
    if (it == g_host_to_addr.end()) return RET(hipErrorInvalidResourceHandle);
    uint64_t addr = it->second;
    g_host_to_addr.erase(it);
    BufferRecord* rec = find_buffer_mut(addr);
    if (!rec) return RET(hipErrorInvalidResourceHandle);
    vx_buffer_unmap(rec->buf, rec->mapped_host);
    vx_result_t r = vx_buffer_release(rec->buf);
    g_by_addr.erase(addr);
    return RET_VX(r);
}

// Enqueue (but do not wait for) a copy. Shared by hipMemcpy/hipMemcpyAsync.
static hipError_t enqueue_memcpy(vx_queue_h q, void* dst, const void* src,
                                 size_t size, hipMemcpyKind kind,
                                 vx_event_h* out_ev) {
    if (!dst || !src) return RET(hipErrorInvalidValue);

    uint64_t dst_addr = (uint64_t)(uintptr_t)dst;
    uint64_t src_addr = (uint64_t)(uintptr_t)src;

    // Resolve mapped host pointers to their device addresses so device
    // targets work uniformly.
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_host_to_addr.find(dst);
        if (it != g_host_to_addr.end()) dst_addr = it->second;
        auto it2 = g_host_to_addr.find(const_cast<void*>(src));
        if (it2 != g_host_to_addr.end()) src_addr = it2->second;
    }

    if (kind == hipMemcpyDefault) {
        bool dst_dev = g_by_addr.count(dst_addr) != 0;
        bool src_dev = g_by_addr.count(src_addr) != 0;
        kind = dst_dev && src_dev ? hipMemcpyDeviceToDevice
              : dst_dev           ? hipMemcpyHostToDevice
                                  : hipMemcpyDeviceToHost;
    }

    if (kind == hipMemcpyDeviceToDevice) {
        const BufferRecord* d = find_buffer(dst_addr);
        const BufferRecord* s = find_buffer(src_addr);
        if (!d || !s) return RET(hipErrorInvalidResourceHandle);
        return RET_VX(vx_enqueue_copy(q, d->buf, 0, s->buf, 0,
                                       (uint64_t)size, 0, nullptr, out_ev));
    }
    if (kind == hipMemcpyHostToDevice) {
        const BufferRecord* d = find_buffer(dst_addr);
        if (!d) return RET(hipErrorInvalidResourceHandle);
        return RET_VX(vx_enqueue_write(q, d->buf, 0, src,
                                        (uint64_t)size, 0, nullptr, out_ev));
    }
    if (kind == hipMemcpyDeviceToHost) {
        const BufferRecord* s = find_buffer(src_addr);
        if (!s) return RET(hipErrorInvalidResourceHandle);
        return RET_VX(vx_enqueue_read(q, dst, s->buf, 0,
                                       (uint64_t)size, 0, nullptr, out_ev));
    }
    memcpy(dst, src, size);
    return hipSuccess;
}

hipError_t hipMemcpy(void* dst, const void* src, size_t size, hipMemcpyKind kind) {
    if (size == 0) return hipSuccess;
    vx_event_h ev = nullptr;
    hipError_t e = enqueue_memcpy(default_queue(), dst, src, size, kind, &ev);
    if (e != hipSuccess) return e;
    return wait_last(VX_SUCCESS, ev);
}

hipError_t hipMemcpyAsync(void* dst, const void* src, size_t size,
                          hipMemcpyKind kind, hipStream_t stream) {
    if (size == 0) return hipSuccess;
    vx_event_h ev = nullptr;
    hipError_t e = enqueue_memcpy(stream_queue(stream), dst, src, size,
                                  kind, &ev);
    if (ev) vx_event_release(ev);  // stream-ordered: caller syncs the stream
    return e;
}

// Enqueue (but do not wait for) a fill. Shared by hipMemset/hipMemsetAsync.
static hipError_t enqueue_memset(vx_queue_h q, void* dst, int value,
                                 size_t size, vx_event_h* out_ev) {
    if (!dst) return RET(hipErrorInvalidValue);
    uint64_t addr = (uint64_t)(uintptr_t)dst;
    const BufferRecord* rec;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_host_to_addr.find(dst);
        if (it != g_host_to_addr.end()) addr = it->second;
        rec = find_buffer(addr);
    }
    if (!rec) return RET(hipErrorInvalidResourceHandle);
    unsigned char pattern = (unsigned char)value;
    return RET_VX(vx_enqueue_fill_buffer(q, rec->buf, 0, (uint64_t)size,
                                          &pattern, 1, 0, nullptr, out_ev));
}

hipError_t hipMemset(void* dst, int value, size_t size) {
    vx_event_h ev = nullptr;
    hipError_t e = enqueue_memset(default_queue(), dst, value, size, &ev);
    if (e != hipSuccess) return e;
    return wait_last(VX_SUCCESS, ev);
}

hipError_t hipMemsetAsync(void* dst, int value, size_t size,
                          hipStream_t stream) {
    vx_event_h ev = nullptr;
    hipError_t e = enqueue_memset(stream_queue(stream), dst, value, size, &ev);
    if (ev) vx_event_release(ev);
    return e;
}

// ---------------------------------------------------------------------------
// stream / event
// ---------------------------------------------------------------------------

hipError_t hipGetVxDevice(void** out) {
    if (!out) return RET(hipErrorInvalidValue);
    if (!g_device.initialized) return RET(hipErrorNotInitialized);
    *out = g_device.dev;
    return hipSuccess;
}

hipError_t hipStreamGetQueue(hipStream_t stream, void** out) {
    if (!out) return RET(hipErrorInvalidValue);
    if (!g_device.initialized) return RET(hipErrorNotInitialized);
    // stream_queue(nullptr) is the default queue; passing the caller's stream
    // through unchanged is what keeps DL launches ordered behind its work.
    *out = stream_queue(stream);
    return hipSuccess;
}

hipError_t hipStreamCreate(hipStream_t* stream) {
    if (!stream) return RET(hipErrorInvalidValue);
    hipError_t e = hipSetDevice(0);
    if (e != hipSuccess) return e;
    auto* s = new StreamState();
    vx_result_t r = queue_create_profiled(g_device.dev, &s->q);
    if (r != VX_SUCCESS) {
        delete s;
        return RET_VX(r);
    }
    register_queue(s->q);   // so hipDeviceSynchronize covers this stream
    *stream = (hipStream_t)s;
    return hipSuccess;
}

hipError_t hipStreamDestroy(hipStream_t stream) {
    if (!stream) return RET(hipErrorInvalidResourceHandle);
    auto* s = (StreamState*)stream;
    // Finish before releasing: the worker thread runs the queued work against
    // this queue, so releasing it under an in-flight command is a
    // use-after-free rather than a lost stream.
    vx_result_t r = vx_queue_finish(s->q, VX_TIMEOUT_INFINITE);
    if (r != VX_SUCCESS) return RET_VX(r);
    unregister_queue(s->q);
    r = vx_queue_release(s->q);
    delete s;
    return RET_VX(r);
}

hipError_t hipStreamSynchronize(hipStream_t stream) {
    if (!g_device.initialized) return RET(hipErrorNotInitialized);
    // See hipDeviceSynchronize: flush is not a barrier, finish is.
    return RET_VX(vx_queue_finish(stream_queue(stream), VX_TIMEOUT_INFINITE));
}

hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event) {
    if (!event) return RET(hipErrorInvalidResourceHandle);
    auto* ev = (EventState*)event;
    if (ev->last_value == 0) return hipSuccess;  // never recorded: no-op
    vx_event_h done = nullptr;
    vx_result_t r = vx_enqueue_wait_value(stream_queue(stream), ev->ev,
                                           ev->last_value, 0, nullptr, &done);
    if (r == VX_SUCCESS && done) vx_event_release(done);
    return RET_VX(r);
}

hipError_t hipEventCreate(hipEvent_t* event) {
    if (!event) return RET(hipErrorInvalidValue);
    hipError_t e = hipSetDevice(0);
    if (e != hipSuccess) return e;
    auto* ev = new EventState();
    vx_result_t r = vx_event_create(g_device.dev, &ev->ev);
    if (r != VX_SUCCESS) {
        delete ev;
        return RET_VX(r);
    }
    *event = (hipEvent_t)ev;
    return hipSuccess;
}

hipError_t hipEventDestroy(hipEvent_t event) {
    if (!event) return RET(hipErrorInvalidResourceHandle);
    auto* ev = (EventState*)event;
    if (ev->done) vx_event_release(ev->done);
    vx_result_t r = vx_event_release(ev->ev);
    delete ev;
    return RET_VX(r);
}

hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
    if (!event) return RET(hipErrorInvalidResourceHandle);
    auto* ev = (EventState*)event;
    ++ev->last_value;
    vx_event_h done = nullptr;
    // The signal command itself completes after the stream's prior work,
    // which is exactly HIP's record-then-observe semantics. Record stays
    // asynchronous; profiling is picked up at synchronize/elapsed time.
    vx_result_t r = vx_enqueue_signal(stream_queue(stream), ev->ev,
                                       ev->last_value, 0, nullptr, &done);
    if (r != VX_SUCCESS) return RET_VX(r);
    if (ev->done) vx_event_release(ev->done);
    ev->done = done;
    return hipSuccess;
}

hipError_t hipEventSynchronize(hipEvent_t event) {
    if (!event) return RET(hipErrorInvalidResourceHandle);
    auto* ev = (EventState*)event;
    if (ev->last_value == 0) return hipSuccess;
    hipError_t e = RET_VX(vx_event_wait_value(ev->ev, ev->last_value, UINT64_MAX));
    if (e != hipSuccess) return e;
    if (ev->done) {
        vx_profile_info_t prof;
        if (vx_event_get_profiling(ev->done, &prof) == VX_SUCCESS) {
            ev->last_ns = prof.end_ns;
        }
    }
    return hipSuccess;
}

hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop) {
    if (!ms || !start || !stop) return RET(hipErrorInvalidValue);
    auto* s = (EventState*)start;
    auto* e = (EventState*)stop;
    // Ensure both records have completed so their timestamps are settled.
    hipError_t err = hipEventSynchronize(start);
    if (err != hipSuccess) return err;
    err = hipEventSynchronize(stop);
    if (err != hipSuccess) return err;
    if (s->last_ns == 0 || e->last_ns == 0 || e->last_ns < s->last_ns) {
        return RET(hipErrorInvalidValue);
    }
    *ms = (float)((double)(e->last_ns - s->last_ns) / 1e6);
    return hipSuccess;
}

// ---------------------------------------------------------------------------
// module / kernel
// ---------------------------------------------------------------------------

hipError_t hipModuleLoad(hipModule_t* module, const char* fname) {
    if (!module || !fname) return RET(hipErrorInvalidValue);
    hipError_t e = hipSetDevice(0);
    if (e != hipSuccess) return e;
    auto* m = new ModuleState();
    vx_result_t r = vx_module_load_file(g_device.dev, fname, &m->mod);
    if (r != VX_SUCCESS) {
        delete m;
        return RET_VX(r);
    }
    *module = (hipModule_t)m;
    return hipSuccess;
}

hipError_t hipModuleUnload(hipModule_t module) {
    if (!module) return RET(hipErrorInvalidResourceHandle);
    auto* m = (ModuleState*)module;
    vx_result_t r = vx_module_release(m->mod);
    delete m;
    return RET_VX(r);
}

hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module,
                                const char* kname) {
    if (!function || !module || !kname) return RET(hipErrorInvalidValue);
    auto* m = (ModuleState*)module;
    auto* f = new FuncState();
    vx_result_t r = vx_module_get_kernel(m->mod, kname, &f->k);
    if (r != VX_SUCCESS) {
        delete f;
        return RET_VX(r);
    }
    *function = (hipFunction_t)f;
    return hipSuccess;
}

hipError_t hipModuleLaunchKernel(hipFunction_t f,
                                 uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
                                 uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
                                 uint32_t sharedMemBytes, hipStream_t stream,
                                 void** kernelParams, void** extra) {
    if (!f) return RET(hipErrorInvalidResourceHandle);
    auto* fn = (FuncState*)f;

    const void* args_blob = nullptr;
    size_t args_size = 0;

    if (extra) {
        // classic pointer/size pairs terminated by NULL
        for (void** e = extra; e[0] != nullptr; e += 2) {
            if (e[0] == HIP_LAUNCH_PARAM_BUFFER_POINTER) {
                args_blob = e[1];
            } else if (e[0] == HIP_LAUNCH_PARAM_BUFFER_SIZE) {
                args_size = (size_t)(uintptr_t)e[1];
            } else {
                return RET(hipErrorNotSupported);
            }
        }
    } else if (kernelParams) {
        // Single-pointer kernel ABI: kernelParams[0] points at the arg block.
        // Its size comes from the image metadata (P1.3) when present.
        args_blob = kernelParams[0];
        vx_kernel_info_t info = {};
        info.struct_size = sizeof(info);
        vx_result_t r = vx_kernel_get_info(fn->k, &info);
        if (r != VX_SUCCESS) return RET_VX(r);
        args_size = info.args_size;
        if (args_size == 0) {
            // No metadata: the caller must describe the blob via `extra`.
            return RET(hipErrorInvalidValue);
        }
    }
    if (args_size && !args_blob) return RET(hipErrorInvalidValue);

    vx_launch_info_t launch = {};
    launch.struct_size = sizeof(launch);
    launch.kernel = fn->k;
    launch.args_host = args_blob;
    launch.args_size = args_size;
    launch.ndim = 3;
    launch.grid_dim[0] = gridDimX;
    launch.grid_dim[1] = gridDimY;
    launch.grid_dim[2] = gridDimZ;
    launch.block_dim[0] = blockDimX;
    launch.block_dim[1] = blockDimY;
    launch.block_dim[2] = blockDimZ;
    launch.lmem_size = sharedMemBytes;

    vx_event_h ev = nullptr;
    vx_result_t r = vx_enqueue_launch(stream_queue(stream), &launch,
                                       0, nullptr, &ev);
    // Launch is async per HIP semantics; hold the completion event without
    // waiting. Errors on the device surface at the next sync point.
    if (r == VX_SUCCESS && ev) {
        vx_event_release(ev);
        return hipSuccess;
    }
    return RET_VX(r);
}

// ---------------------------------------------------------------------------
// hiprtc stubs: not implemented, loudly (see hip_runtime_api.h)
// ---------------------------------------------------------------------------

hipError_t hiprtcCreateProgram(hiprtcProgram*, const char*, const char*, int,
                               const char**, const char**) {
    return RET(hipErrorNotSupported);
}
hipError_t hiprtcCompileProgram(hiprtcProgram, int, const char**) {
    return RET(hipErrorNotSupported);
}
hipError_t hiprtcGetCodeSize(hiprtcProgram, size_t*) {
    return RET(hipErrorNotSupported);
}
hipError_t hiprtcGetCode(hiprtcProgram, char*) {
    return RET(hipErrorNotSupported);
}
hipError_t hiprtcDestroyProgram(hiprtcProgram*) {
    return RET(hipErrorNotSupported);
}

// ---------------------------------------------------------------------------
// errors
// ---------------------------------------------------------------------------

hipError_t hipGetLastError(void) {
    hipError_t e = g_last_error;
    g_last_error = hipSuccess;
    return e;
}

const char* hipGetErrorString(hipError_t error) {
    switch (error) {
    case hipSuccess: return "hipSuccess";
    case hipErrorInvalidValue: return "hipErrorInvalidValue";
    case hipErrorOutOfMemory: return "hipErrorOutOfMemory";
    case hipErrorInvalidResourceHandle: return "hipErrorInvalidResourceHandle";
    case hipErrorNotInitialized: return "hipErrorNotInitialized";
    case hipErrorNotSupported: return "hipErrorNotSupported";
    case hipErrorTimeout: return "hipErrorTimeout";
    default: return "hipErrorUnknown";
    }
}

const char* hipGetErrorName(hipError_t error) {
    return hipGetErrorString(error);
}
