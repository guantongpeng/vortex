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

// torch-vortex C++ extension (plan P5-01): registers `vortex` as a
// PrivateUse1 backend — allocator (hipMalloc-backed), device guard,
// copy_ for every direction, and eager add/mul/fill_ that launch the
// kernels/torch_ops.vxbin image through libhip_vortex.
//
// v1 constraints (enforced loudly, never silently fallen back):
//   - FP32 tensors only in device ops (TORCH_CHECK otherwise)
//   - contiguous tensors (TORCH_CHECK otherwise)
//   - one device (hipInit/hipSetDevice(0) at import)

#include <torch/extension.h>
#include <torch/version.h>

#include <atomic>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <c10/core/Allocator.h>
#if __has_include(<c10/core/CachingDeviceAllocator.h>)
#include <c10/core/CachingDeviceAllocator.h>
#define VX_HAS_DEVICE_ALLOCATOR 1
#else
// PyTorch 2.4 has the PrivateUse1 allocator ABI but predates the generic
// DeviceAllocator/CachingDeviceAllocator interface. Keep the backend usable
// on that baseline; newer releases still get the richer memory API.
#define VX_HAS_DEVICE_ALLOCATOR 0
#endif
#include <ATen/EmptyTensor.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/view_native.h>
#include <c10/core/Device.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/string_view.h>

extern "C" {
#include <hip/hip_runtime_api.h>
}

// The single definition of every kernel argument block, shared with the device
// compiler. Never declare an argument struct in this file.
#include "torch_kernel_args.h"

// The generic accelerator capability and device-wide synchronization hooks
// landed after the 2.4 PrivateUse1 ABI. Keep the older baseline buildable;
// newer versions use the richer overrides below.
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 14)
#define VX_HAS_ACCELERATOR_GUARD_API 1
#else
#define VX_HAS_ACCELERATOR_GUARD_API 0
#endif

#if VX_HAS_ACCELERATOR_GUARD_API
using vx_schema_string_view = std::string_view;
#else
using vx_schema_string_view = c10::string_view;
#endif

// The device DL library (sw/dl). Ops that have an equivalent there call into
// it rather than keeping a second copy of the same algorithm -- see W3.1 of
// docs/mydocs/pytorch_plan.md. Its entry points speak vortex2.h and take this
// process's device and queue, which is why sw/hip exposes the two accessors.
#include <vortex/blas.h>
#include <vortex/dnn.h>
#include <vortex/prim.h>

#define VX_CHECK(expr)                                                        \
    do {                                                                      \
        hipError_t _e = (expr);                                               \
        TORCH_CHECK(_e == hipSuccess, "torch_vortex: " #expr " failed: ",     \
                    hipGetErrorString(_e));                                   \
    } while (0)

// The DL library has its own status enums (all with 0 = OK) rather than
// hipError_t, so it gets its own check. Initialisation is not device work and
// uses this one.
#define DL_CHECK(expr)                                                        \
    do {                                                                      \
        const int _s = (int)(expr);                                           \
        TORCH_CHECK(_s == 0, "torch_vortex: " #expr " failed with status ",   \
                    _s);                                                      \
    } while (0)

// A DL call that queues a kernel. It has to say so: the DL library launches on
// the queue it was handed and never passes through launch(), which is the only
// place the allocator's epoch is kept. A DL launch that skipped this left
// work_since_last_barrier() false, so freeing a tensor the kernel was still
// reading took the immediate hipFree path -- the race the deferred free exists
// to remove, reintroduced by the migration that moved these ops into the
// library.
#define DL_LAUNCH(expr)                                                       \
    do {                                                                      \
        DL_CHECK(expr);                                                       \
        note_device_work();                                                   \
    } while (0)



// ---------------------------------------------------------------------------
// Counters
// ---------------------------------------------------------------------------
//
// The backend's central claim is that a vortex tensor's work happens on the
// device. That is only checkable if the host side reports what it did, so
// every launch, transfer, allocation and blocking sync is counted here and
// exposed as torch_vortex.stats(). Process-global and deliberately not
// thread-safe: these are diagnostics for single-threaded tests, not a
// profiler (W7.1 of docs/mydocs/pytorch_plan.md owns that).

namespace {

struct VortexStats {
    std::atomic<uint64_t> launches{0};
    std::atomic<uint64_t> skipped_launches{0};
    std::atomic<uint64_t> h2d_bytes{0};
    std::atomic<uint64_t> d2h_bytes{0};
    std::atomic<uint64_t> d2d_bytes{0};
    std::atomic<uint64_t> allocations{0};
    std::atomic<uint64_t> allocated_bytes{0};
    std::atomic<uint64_t> freed_bytes{0};
    std::atomic<uint64_t> frees{0};
    std::atomic<uint64_t> immediate_frees{0};
    std::atomic<uint64_t> blocking_syncs{0};
    std::atomic<uint64_t> host_numeric_ops{0};
};

VortexStats g_stats;

// False until load_ops has initialised the device, and false again once it is
// torn down. The allocator's deleter consults it: the stream-ordered free
// needs a live queue, and during teardown there may not be one.
std::atomic<bool> g_device_ready{false};

// Live device bytes. c10's deleter is a bare void(*)(void*) with no size in
// it, so the size is remembered here for the duration of the block.
std::atomic<int64_t> g_live_bytes{0};
std::atomic<int64_t> g_peak_bytes{0};
std::mutex g_block_mu;
std::unordered_map<void*, size_t> g_block_sizes;

// Epoch counters for the allocator's fast path.
//
// The queue is FIFO and every host-side barrier (hipMemcpy waits on its own
// completion event, hipDeviceSynchronize finishes the queue) retires
// everything enqueued before it. So "no launch since the last barrier" means
// there is no in-flight kernel that could still be reading a buffer we are
// about to free, and the free can be immediate.
//
// Without this every free was deferred, which is correct but not free: the
// test suite went from 25s to 48s because no address was ever reused.
std::atomic<uint64_t> g_launch_epoch{0};
std::atomic<uint64_t> g_barrier_epoch{0};
// Set when anything was launched on an explicit stream. hipMemcpy drains only
// the default queue, so an epoch match cannot cover those launches; only a
// device-wide barrier clears this.
std::atomic<bool> g_nondefault_work{false};

// After something that drains the default queue host-side (hipMemcpy).
void note_default_queue_barrier() {
    g_barrier_epoch.store(g_launch_epoch.load());
}

// After a device-wide barrier (hipDeviceSynchronize).
void note_device_barrier() {
    g_barrier_epoch.store(g_launch_epoch.load());
    g_nondefault_work = false;
}

bool work_since_last_barrier() {
    return g_nondefault_work.load() ||
           g_launch_epoch.load() != g_barrier_epoch.load();
}

std::map<std::string, int64_t> stats_snapshot() {
    return {
        {"launches", (int64_t)g_stats.launches.load()},
        {"skipped_launches", (int64_t)g_stats.skipped_launches.load()},
        {"h2d_bytes", (int64_t)g_stats.h2d_bytes.load()},
        {"d2h_bytes", (int64_t)g_stats.d2h_bytes.load()},
        {"d2d_bytes", (int64_t)g_stats.d2d_bytes.load()},
        {"allocations", (int64_t)g_stats.allocations.load()},
        {"allocated_bytes", (int64_t)g_stats.allocated_bytes.load()},
        {"freed_bytes", (int64_t)g_stats.freed_bytes.load()},
        {"live_bytes", (int64_t)g_live_bytes.load()},
        {"frees", (int64_t)g_stats.frees.load()},
        {"immediate_frees", (int64_t)g_stats.immediate_frees.load()},
        {"blocking_syncs", (int64_t)g_stats.blocking_syncs.load()},
        {"host_numeric_ops", (int64_t)g_stats.host_numeric_ops.load()},
    };
}

void stats_reset() {
    g_stats.launches = 0;
    g_stats.skipped_launches = 0;
    g_stats.h2d_bytes = 0;
    g_stats.d2h_bytes = 0;
    g_stats.d2d_bytes = 0;
    g_stats.allocations = 0;
    g_stats.allocated_bytes = 0;
    g_stats.freed_bytes = 0;
    g_stats.frees = 0;
    g_stats.immediate_frees = 0;
    g_stats.blocking_syncs = 0;
    g_stats.host_numeric_ops = 0;
}

// Dimensions and element counts travel through uint32_t fields in the argument
// blocks; refuse anything that would be silently truncated.
static uint32_t u32_dim(int64_t v, const char* what) {
    TORCH_CHECK(v >= 0 && v <= (int64_t)UINT32_MAX,
                "torch_vortex: ", what, " = ", v, " does not fit in uint32");
    return (uint32_t)v;
}

static uint32_t u32_numel(const torch::Tensor& t, const char* what) {
    TORCH_CHECK(t.numel() <= (int64_t)UINT32_MAX, "torch_vortex: ", what,
                " has ", t.numel(), " elements, which does not fit in uint32");
    return (uint32_t)t.numel();
}

// ---------------------------------------------------------------------------
// Streams
// ---------------------------------------------------------------------------
//
// c10 identifies a stream by (device, id). The id -> hipStream_t mapping is
// process-wide by nature; the *current* stream is thread-local, which is what
// the guard interface contract requires (DeviceGuardImplInterface.h documents
// thread-local semantics, and the previous process-global pair meant two host
// threads shared one "current stream" slot).
//
// id 0 is the device's default queue, i.e. a null hipStream_t.

namespace {

std::mutex g_stream_mu;
std::unordered_map<c10::StreamId, hipStream_t> g_streams;
std::atomic<c10::StreamId> g_next_stream_id{1};

c10::Stream default_stream() {
    return c10::Stream(c10::Stream::DEFAULT,
                       c10::Device(c10::DeviceType::PrivateUse1, 0));
}

thread_local c10::Stream t_current_stream = default_stream();
thread_local c10::DeviceIndex t_current_device = 0;

hipStream_t hip_stream_of(c10::Stream s) {
    if (s.device_type() != c10::DeviceType::PrivateUse1) {
        TORCH_CHECK(false, "torch_vortex: stream ", s, " is not a vortex stream");
    }
    if (s.id() == default_stream().id()) {
        return nullptr;   // the device's default queue
    }
    std::lock_guard<std::mutex> g(g_stream_mu);
    auto it = g_streams.find(s.id());
    TORCH_CHECK(it != g_streams.end(), "torch_vortex: unknown stream id ", s.id(),
                " -- it was not created by this process, or it was destroyed");
    return it->second;
}

c10::Stream create_stream() {
    hipStream_t hs = nullptr;
    VX_CHECK(hipStreamCreate(&hs));
    const c10::StreamId id = g_next_stream_id++;
    {
        std::lock_guard<std::mutex> g(g_stream_mu);
        g_streams.emplace(id, hs);
    }
    return c10::Stream(c10::Stream::UNSAFE,
                       c10::Device(c10::DeviceType::PrivateUse1, 0), id);
}

// The stream ops launch on. Before this existed every launch passed nullptr
// and the guard's stream was decorative.
hipStream_t current_hip_stream() {
    return hip_stream_of(t_current_stream);
}

// The DL library's queue for whatever stream the caller is on, so a DL launch
// lands behind the caller's own work rather than on a parallel queue.
vx_queue_h current_queue() {
    void* q = nullptr;
    VX_CHECK(hipStreamGetQueue(current_hip_stream(), &q));
    return (vx_queue_h)q;
}

// Bookkeeping every submitted kernel owes, whichever path submitted it: the
// diagnostic counter and the epochs the allocator's deferred free reads.
// Defined once because there are now two launch paths into the device -- this
// extension's own kernels and the DL library's -- and only one of them used to
// pay it.
void note_device_work() {
    ++g_stats.launches;
    if (current_hip_stream() == nullptr) {
        ++g_launch_epoch;
    } else {
        g_nondefault_work = true;
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Launch
// ---------------------------------------------------------------------------

// The one place a kernel is launched. It takes the argument block size from
// the host compiler's sizeof via HIP_LAUNCH_PARAM_BUFFER_SIZE rather than
// letting the runtime read args_size out of the image metadata, so a stale or
// hand-edited metadata file can no longer make the runtime copy the wrong
// number of bytes off this stack frame (which is what a 104-byte conv record
// for an 88-byte struct used to do).
//
// A zero grid dimension means "nothing to do": no launch is issued and the
// skip is counted, rather than enqueueing a zero-CTA kernel whose output is
// left uninitialised.
template <typename Args>
void launch(hipFunction_t f, const Args& args, uint32_t gx, uint32_t gy = 1,
            uint32_t gz = 1, uint32_t bx = 4, uint32_t lmem = 0) {
    static_assert(std::is_standard_layout<Args>::value,
                  "argument blocks must be standard layout");
    if (gx == 0 || gy == 0 || gz == 0) {
        ++g_stats.skipped_launches;
        return;
    }
    void* extra[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, const_cast<Args*>(&args),
        HIP_LAUNCH_PARAM_BUFFER_SIZE, (void*)(uintptr_t)sizeof(Args),
        (void*)0,
    };
    VX_CHECK(hipModuleLaunchKernel(f, gx, gy, gz, bx, 1, 1, lmem,
                                   current_hip_stream(), nullptr, extra));
    note_device_work();
}

// The previous calling convention (a single pointer in kernelParams, with the
// size read from image metadata) is deliberately not used anywhere any more.

} // namespace

// Device-wide barrier. Defined with the other pybind entry points, below;
// the guard's synchronizeDevice needs it, so it is declared at file scope.
void device_synchronize_impl();
void vortex_at_exit();

// ---------------------------------------------------------------------------
// Allocator: device memory via hipMalloc (vx_buffer_create under it)
// ---------------------------------------------------------------------------

namespace {

void vortex_free(void* ctx);

#if VX_HAS_DEVICE_ALLOCATOR
struct VortexAllocator final : public c10::DeviceAllocator {
#else
struct VortexAllocator final : public c10::Allocator {
#endif
    c10::DataPtr allocate(size_t n) override {
        void* p = nullptr;
        if (n != 0) {
            VX_CHECK(hipMalloc(&p, n));
            ++g_stats.allocations;
            g_stats.allocated_bytes += n;
            {
                std::lock_guard<std::mutex> g(g_block_mu);
                g_block_sizes[p] = n;
            }
            g_live_bytes += (int64_t)n;
            if (g_live_bytes.load() > g_peak_bytes.load()) {
                g_peak_bytes = g_live_bytes.load();
            }
        }
        return c10::DataPtr(p, p, &vortex_free,
                            c10::Device(c10::DeviceType::PrivateUse1, 0));
    }
    void copy_data(void* dest, const void* src, std::size_t count) const override {
        // Device-to-device byte copy through the host-visible buffer
        // mapping (same-address-space model on current backends).
        VX_CHECK(hipMemcpy(dest, src, count, hipMemcpyDeviceToDevice));
        g_stats.d2d_bytes += count;
    }

#if VX_HAS_DEVICE_ALLOCATOR
    // ---- c10::DeviceAllocator ----------------------------------------------
    //
    // torch calls these through getDeviceAllocator(), which dynamic_casts the
    // registered allocator to DeviceAllocator and asserts. Before this the
    // cast failed, so torch.accelerator.memory_allocated() died with
    // "Allocator for vortex is not a DeviceAllocator".

    bool initialized() override { return g_device_ready.load(); }

    // There is no cache to empty: this allocator hands every block straight to
    // hipMalloc and releases it through the queue (see vortex_free). It
    // deliberately does not force the deferred frees to complete.
    void emptyCache(c10::MempoolId_t = {0, 0}) override {}

    // A caching allocator would tag the block with the stream so it is not
    // recycled while that stream still uses it. There are no blocks to tag
    // yet -- W2.1 defers the release through the queue instead -- so this is
    // where W3.1's caching allocator will hang its bookkeeping.
    void recordStream(const c10::DataPtr&, c10::Stream) override {}

    c10::CachingDeviceAllocator::DeviceStats
    getDeviceStats(c10::DeviceIndex) override {
        // The header puts DeviceStats in one namespace and the Stat types
        // in another; that is what it does, not a typo here.
        using c10::CachingDeviceAllocator::DeviceStats;
        using c10::CachingAllocator::Stat;
        using c10::CachingAllocator::StatType;
        DeviceStats s;
        const size_t agg = static_cast<size_t>(StatType::AGGREGATE);
        const int64_t live = g_live_bytes.load();

        auto fill = [&](c10::CachingAllocator::StatArray& a, int64_t cur, int64_t peak,
                        int64_t total, int64_t freed) {
            Stat& st = a[agg];
            st.current = cur;
            st.peak = peak;
            st.allocated = total;
            st.freed = freed;
        };
        // Every block is live from hipMalloc until the queued release runs, and
        // this allocator never pools or splits, so reserved == allocated and
        // the header fields mirror the aggregate ones.
        fill(s.allocation, g_stats.allocations.load(), g_stats.allocations.load(),
             g_stats.allocations.load(), g_stats.frees.load());
        fill(s.segment, g_stats.allocations.load(), g_stats.allocations.load(),
             g_stats.allocations.load(), g_stats.frees.load());
        fill(s.active, g_stats.allocations.load() - g_stats.frees.load(),
             g_stats.allocations.load() - g_stats.frees.load(),
             g_stats.allocations.load(), g_stats.frees.load());
        fill(s.inactive_split, 0, 0, 0, 0);
        fill(s.allocated_bytes, live, g_peak_bytes.load(),
             g_stats.allocated_bytes.load(), g_stats.freed_bytes.load());
        fill(s.reserved_bytes, live, g_peak_bytes.load(),
             g_stats.allocated_bytes.load(), g_stats.freed_bytes.load());
        fill(s.active_bytes, live, g_peak_bytes.load(),
             g_stats.allocated_bytes.load(), g_stats.freed_bytes.load());
        fill(s.inactive_split_bytes, 0, 0, 0, 0);
        fill(s.requested_bytes, live, g_peak_bytes.load(),
             g_stats.allocated_bytes.load(), g_stats.freed_bytes.load());
        s.num_device_alloc = (int64_t)g_stats.allocations.load();
        s.num_device_free = (int64_t)g_stats.frees.load();
        s.num_alloc_retries = 0;
        s.num_ooms = 0;
        return s;
    }

    void resetAccumulatedStats(c10::DeviceIndex) override {
        g_stats.allocated_bytes = 0;
        g_stats.freed_bytes = 0;
    }
    void resetPeakStats(c10::DeviceIndex) override {
        g_peak_bytes = g_live_bytes.load();
    }

    std::pair<size_t, size_t> getMemoryInfo(c10::DeviceIndex) override {
        hipDeviceProp_t prop;
        VX_CHECK(hipGetDeviceProperties(&prop, 0));
        const size_t total = (size_t)prop.totalGlobalMem;
        const size_t live = (size_t)g_live_bytes.load();
        return {total > live ? total - live : 0, total};
    }
#endif
};

// The c10 deleter. It runs on whichever thread drops the last reference, and
// c10::Allocator gives it nowhere to record which stream the tensor was used
// on (allocate/deleter take no stream and DataPtr carries only a raw pointer),
// so this releases through the default queue -- which is where every op
// currently runs.
//
// hipFree, not hipFree, is NOT safe here: it returns the address to the device
// allocator's free list immediately, on this thread, with no regard for a
// launch that is still reading it. Measured before this change: a 32768-element
// relu recycled its input's address 44 us after the launch returned, with 11.6 s
// of kernel still to run. hipFreeAsync defers the release behind the queue's
// pending work, so the address cannot come back until the kernel has retired.
//
// What it costs: memory is held until the queue drains, so a program that frees
// a lot while a long kernel runs will use more. Lifetime correctness first --
// the plan is explicit that a caching allocator comes after, not before.
void vortex_free(void* ctx) {
    if (ctx == nullptr) {
        return;
    }
    ++g_stats.frees;
    {
        std::lock_guard<std::mutex> g(g_block_mu);
        auto it = g_block_sizes.find(ctx);
        if (it != g_block_sizes.end()) {
            g_stats.freed_bytes += it->second;
            g_live_bytes -= (int64_t)it->second;
            g_block_sizes.erase(it);
        }
    }
    // Fast path: nothing has been enqueued since the last host-side barrier,
    // so no kernel can still be reading this buffer.
    if (g_device_ready && work_since_last_barrier()) {
        if (hipFreeAsync(ctx, nullptr) == hipSuccess) {
            return;
        }
        // Counted rather than swallowed: a silent fallback to the immediate
        // free is exactly the race this function exists to remove.
        ++g_stats.immediate_frees;
    } else {
        ++g_stats.immediate_frees;
    }
    hipFree(ctx);
}

// Deliberately leaked, for the same reason as the registrations below.
//
// c10::SetAllocator is non-owning: torch holds this pointer for the life of
// the process and can still call it while this .so's statics are being
// destroyed. A static object here is destroyed first, which is a "pure virtual
// method called" abort as soon as anything is freed during teardown.
VortexAllocator& vortex_allocator() {
    static VortexAllocator* a = new VortexAllocator();
    return *a;
}

// ---------------------------------------------------------------------------
// Device guard: one device, a real per-thread current stream
// ---------------------------------------------------------------------------

// A c10::Event is an opaque void* owned by the backend; this is what we put
// behind it. It wraps the same hipEvent_t the HIP layer already implements, so
// event semantics live in one place.
namespace {

struct VortexEvent {
    hipEvent_t ev = nullptr;
};

void check_device_arg(c10::Device d) {
    TORCH_CHECK(d.type() == c10::DeviceType::PrivateUse1,
                "torch_vortex: expected a vortex device, got ", d);
    // index() is -1 when the caller wrote device="vortex" with no index
    TORCH_CHECK(d.index() == -1 || d.index() == 0,
                "torch_vortex: only device 0 exists, got \"", d.str(), "\"");
}

} // namespace

struct VortexGuardImpl final : public c10::impl::DeviceGuardImplInterface {
    c10::DeviceType type() const override {
        return c10::DeviceType::PrivateUse1;
    }
    c10::Device exchangeDevice(c10::Device d) const override {
        const c10::Device old(c10::DeviceType::PrivateUse1, t_current_device);
        setDevice(d);
        return old;
    }
    void setDevice(c10::Device d) const override {
        check_device_arg(d);
        t_current_device = 0;
    }
    c10::Device getDevice() const override {
        return c10::Device(c10::DeviceType::PrivateUse1, t_current_device);
    }
    c10::DeviceIndex deviceCount() const noexcept override { return 1; }
    void uncheckedSetDevice(c10::Device d) const noexcept override {
        t_current_device = (d.index() < 0) ? 0 : d.index();
    }

#if VX_HAS_ACCELERATOR_GUARD_API
    // The default is "every scalar type is supported", which for this backend
    // would claim double, half, int and the quantized types. Only float32 is
    // implemented for compute in v1 (W3.2/W4.1), so say exactly that.
    c10::DeviceCapability getDeviceCapability(c10::Device) const override {
        c10::DeviceCapability cap;
        cap.capability_data.capability_bits = 1ULL << c10::kIndex_Float;
        return cap;
    }
#endif

    c10::Stream getStream(c10::Device) const noexcept override {
        return t_current_stream;
    }
    c10::Stream getDefaultStream(c10::Device) const override {
        return default_stream();
    }
    c10::Stream exchangeStream(c10::Stream s) const noexcept override {
        auto old = t_current_stream;
        t_current_stream = s;
        return old;
    }
    // Creates a real HIP queue. There is no destroyStream hook in this torch
    // version, so a stream lives until the process exits -- the same shape the
    // CUDA guard has for non-pooled streams.
    c10::Stream getNewStream(c10::Device, int = 0) const override {
        return create_stream();
    }

    // ---- synchronisation ---------------------------------------------------
    void synchronizeStream(const c10::Stream& s) const override {
        VX_CHECK(hipStreamSynchronize(hip_stream_of(s)));
        // A stream barrier drains that queue, so nothing enqueued on *it* is in
        // flight any more. Work on other streams is covered separately by the
        // non-default flag.
        note_default_queue_barrier();
    }
#if VX_HAS_ACCELERATOR_GUARD_API
    void synchronizeDevice(const c10::DeviceIndex) const override {
        device_synchronize_impl();
    }
#endif
    // queryStream/queryEvent are deliberately not overridden: the runtime has
    // no non-blocking queue/event query (vx_queue_finish always enqueues a
    // barrier, so polling with it would grow the queue), and this increment
    // may not change sw/runtime. The base class raises
    // "Backend doesn't support querying streams." rather than answering
    // wrongly. W2.2 of docs/mydocs/pytorch_plan.md records the gap.

    // ---- events ------------------------------------------------------------
    void record(void** event, const c10::Stream& s, const c10::DeviceIndex,
                const c10::EventFlag) const override {
        TORCH_CHECK(event != nullptr, "torch_vortex: record needs an event slot");
        if (*event == nullptr) {
            auto* e = new VortexEvent();
            VX_CHECK(hipEventCreate(&e->ev));
            *event = e;
        }
        VX_CHECK(hipEventRecord(static_cast<VortexEvent*>(*event)->ev,
                                hip_stream_of(s)));
    }
    void block(void* event, const c10::Stream& s) const override {
        TORCH_CHECK(event != nullptr, "torch_vortex: block needs an event");
        VX_CHECK(hipStreamWaitEvent(hip_stream_of(s),
                                    static_cast<VortexEvent*>(event)->ev));
    }
    void synchronizeEvent(void* event) const override {
        TORCH_CHECK(event != nullptr, "torch_vortex: synchronizeEvent needs an event");
        VX_CHECK(hipEventSynchronize(static_cast<VortexEvent*>(event)->ev));
    }
    void destroyEvent(void* event, const c10::DeviceIndex) const noexcept override {
        if (event == nullptr) {
            return;   // never recorded, so nothing was allocated
        }
        auto* e = static_cast<VortexEvent*>(event);
        hipEventDestroy(e->ev);
        delete e;
    }
    double elapsedTime(void* start, void* end,
                       const c10::DeviceIndex) const override {
        TORCH_CHECK(start != nullptr && end != nullptr,
                    "torch_vortex: elapsedTime needs two recorded events");
        float ms = 0.0f;
        VX_CHECK(hipEventElapsedTime(&ms, static_cast<VortexEvent*>(start)->ev,
                                     static_cast<VortexEvent*>(end)->ev));
        return (double)ms;
    }
};

} // namespace

C10_REGISTER_GUARD_IMPL(PrivateUse1, VortexGuardImpl);

// ---------------------------------------------------------------------------
// Op kernel image
// ---------------------------------------------------------------------------

namespace {

// The remaining torch-image argument blocks are defined once in
// kernels/torch_kernel_args.h and shared verbatim with the device compiler.
// Elementwise binary blocks live in sw/dl/src/prim_args.h and are submitted
// through the prim API above.

hipModule_t g_ops_module = nullptr;

// One handle per row of TORCH_KERNEL_TABLE, in table order. Adding a kernel
// means adding a row in kernels/torch_kernel_args.h: load_ops resolves it and
// the arg-size cross-check covers it automatically.
#define TORCH_KERNEL_DECLARE(name, type, mbx, lmem) hipFunction_t h_##name = nullptr;
TORCH_KERNEL_TABLE(TORCH_KERNEL_DECLARE)
#undef TORCH_KERNEL_DECLARE

// Filled in by load_ops. The conv kernel stages one filter's weights in LMEM,
// so this is the ceiling on ci*kh*kw*4.
int64_t g_shared_mem_per_block = 0;

// The DL library's device handle: the same one the HIP layer owns, handed to
// it rather than opened separately.
vx_device_h g_dl_device = nullptr;

void launch_binary_op(uint64_t dst, uint64_t a, uint64_t b, uint32_t n,
                      uint32_t op) {
    DL_LAUNCH(vx_prim_binary(current_queue(), (vx_prim_binary_op)op,
                             dst, a, b, n));
}

// The unary operations are sw/dl's prim kernel rather than one of ours (W3.1).
// `op` is a vx_prim_op, in-place forms included: source and destination are the
// same buffer there, which is safe because each thread reads and writes the
// same index.
void launch_unary_op(uint64_t dst, uint64_t a, uint32_t n, uint32_t op) {
    // Nothing to do is not a launch, and the DL entry point refuses n == 0
    // rather than accepting an empty job. This used to fall out of the grid
    // being zero; it has to be said out loud now.
    if (n == 0) {
        ++g_stats.skipped_launches;
        return;
    }
    DL_LAUNCH(vx_prim_unary(current_queue(), (vx_prim_op)op, a, dst, n));
}

void launch_scalar_op(uint64_t dst, uint64_t a, float value, uint32_t n,
                      uint32_t op, uint32_t reverse) {
    DL_LAUNCH(vx_prim_scalar(current_queue(), (vx_prim_binary_op)op,
                             dst, a, value, n, reverse));
}

void check_vortex_f32(const torch::Tensor& t, const char* what) {
    // Naming the offending device matters here: `x + 1.0` reaches this with a
    // 0-dim CPU tensor (PyTorch wraps the scalar), and "must be a vortex
    // tensor" reads like a bug in the caller rather than a v1 boundary.
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1,
                "torch_vortex: ", what, " is on ", t.device(), " rather than ",
                "the vortex device; scalar operands (which PyTorch wraps as ",
                "0-dim CPU tensors) and cross-device operands are unsupported ",
                "in v1 -- see W3.2 in docs/mydocs/pytorch_plan.md");
    TORCH_CHECK(t.scalar_type() == at::kFloat,
                "torch_vortex: ", what, " must be float32 in v1, got ",
                t.scalar_type());
    TORCH_CHECK(t.is_contiguous(),
                "torch_vortex: ", what, " must be contiguous in v1");
}

} // namespace

// ---------------------------------------------------------------------------
// aten registrations
// ---------------------------------------------------------------------------

static const auto kVortexDispatchKeys =
    c10::DispatchKeySet(c10::DispatchKey::PrivateUse1) |
    c10::DispatchKeySet(c10::DispatchKey::AutogradPrivateUse1);

static std::vector<int64_t> sym_to_vec(c10::SymIntArrayRef syms) {
    std::vector<int64_t> out;
    out.reserve(syms.size());
    for (const auto& s : syms) {
        out.push_back(s.expect_int());
    }
    return out;
}

// A device argument naming vortex:1 or higher must be refused. It used to be
// silently accepted and answered with a vortex:0 tensor.
static void check_vortex_device_arg(const std::optional<c10::Device>& d,
                                    const char* what) {
    if (!d.has_value()) {
        return;
    }
    TORCH_CHECK(d->type() == c10::DeviceType::PrivateUse1,
                "torch_vortex: wrong device in ", what, ": ", *d);
    // index() is -1 when the caller wrote device="vortex" with no index
    TORCH_CHECK(d->index() == -1 || d->index() == 0,
                "torch_vortex: ", what, ": only device 0 exists, got \"",
                d->str(), "\"");
}

static torch::Tensor empty_impl(c10::SymIntArrayRef sym_size,
                                std::optional<c10::ScalarType> dtype_opt,
                                std::optional<c10::Layout> layout_opt,
                                std::optional<c10::Device> device_opt,
                                std::optional<bool> pin_memory_opt,
                                std::optional<c10::MemoryFormat> memory_format_opt) {
    TORCH_CHECK(!layout_opt.has_value() || *layout_opt == c10::Layout::Strided,
                "torch_vortex: strided layout only");
    check_vortex_device_arg(device_opt, "empty");
    auto dtype = dtype_opt.value_or(at::kFloat);
    auto sizes = sym_to_vec(sym_size);
    auto base = at::detail::empty_generic(
        c10::IntArrayRef(sizes), &vortex_allocator(), kVortexDispatchKeys,
        dtype, memory_format_opt);
    return torch::Tensor(std::move(base));
}

static torch::Tensor empty_strided_impl(
    c10::SymIntArrayRef sym_size, c10::SymIntArrayRef sym_stride,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
    TORCH_CHECK(!layout_opt.has_value() || *layout_opt == c10::Layout::Strided,
                "torch_vortex: strided layout only");
    check_vortex_device_arg(device_opt, "empty_strided");
    auto dtype = dtype_opt.value_or(at::kFloat);
    auto sizes = sym_to_vec(sym_size);
    auto strides = sym_to_vec(sym_stride);
    auto base = at::detail::empty_strided_generic(
        c10::IntArrayRef(sizes), c10::IntArrayRef(strides),
        &vortex_allocator(), kVortexDispatchKeys, dtype);
    return torch::Tensor(std::move(base));
}

static torch::Tensor arange_start_step_impl(
    const c10::Scalar& start, const c10::Scalar& end, const c10::Scalar& step,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
    TORCH_CHECK(!layout_opt.has_value() || *layout_opt == c10::Layout::Strided,
                "torch_vortex: arange supports strided layout only");
    check_vortex_device_arg(device_opt, "arange");
    TORCH_CHECK(!pin_memory_opt.value_or(false),
                "torch_vortex: arange pin_memory is unsupported");

    const bool integral = start.isIntegral(false) && end.isIntegral(false) &&
                          step.isIntegral(false);
    const auto dtype = dtype_opt.value_or(integral ? at::kLong : at::kFloat);
    TORCH_CHECK(dtype == at::kFloat || dtype == at::kInt || dtype == at::kLong,
                "torch_vortex: arange supports only float32, int32 and int64, got ",
                dtype);
    TORCH_CHECK(step.isIntegral(false) || step.isFloatingPoint(),
                "torch_vortex: arange step must be an integer or float");

    uint64_t n = 0;
    int64_t start_i = 0;
    int64_t step_i = 0;
    float start_f = 0.0f;
    float step_f = 0.0f;
    if (integral) {
        start_i = start.toLong();
        const int64_t end_i = end.toLong();
        step_i = step.toLong();
        TORCH_CHECK(step_i != 0, "torch_vortex: arange step must be nonzero");
        if ((step_i > 0 && end_i > start_i) ||
            (step_i < 0 && end_i < start_i)) {
            const __int128 distance = step_i > 0
                                          ? (__int128)end_i - start_i
                                          : (__int128)start_i - end_i;
            const __int128 stride = step_i > 0 ? step_i : -(__int128)step_i;
            const __int128 count = (distance + stride - 1) / stride;
            TORCH_CHECK(count <= UINT32_MAX,
                        "torch_vortex: arange result has too many elements");
            n = (uint64_t)count;
        }
    } else {
        const double start_d = start.toDouble();
        const double end_d = end.toDouble();
        const double step_d = step.toDouble();
        TORCH_CHECK(std::isfinite(start_d) && std::isfinite(end_d) &&
                        std::isfinite(step_d),
                    "torch_vortex: arange requires finite start, end and step");
        TORCH_CHECK(step_d != 0.0, "torch_vortex: arange step must be nonzero");
        const long double span = ((long double)end_d - start_d) / step_d;
        if (span > 0.0L) {
            const long double count = std::ceil(span);
            TORCH_CHECK(count <= (long double)UINT32_MAX,
                        "torch_vortex: arange result has too many elements");
            n = (uint64_t)count;
        }
        start_f = (float)start_d;
        step_f = (float)step_d;
        TORCH_CHECK(dtype == at::kFloat,
                    "torch_vortex: integer arange dtype requires integral bounds and step");
    }

    auto options = torch::TensorOptions()
                       .device(c10::Device(c10::DeviceType::PrivateUse1, 0))
                       .dtype(dtype);
    auto out = torch::empty({(int64_t)n}, options);
    arange_args_t args = {};
    args.dst = (uint64_t)(uintptr_t)out.data_ptr();
    args.n = (uint32_t)n;
    args.dtype = dtype == at::kFloat ? 0 : (dtype == at::kInt ? 1 : 2);
    args.start = start_f;
    args.step = step_f;
    args.start_i = start_i;
    args.step_i = step_i;
    launch(h_arange_kernel, args, (args.n + 3) / 4);
    return out;
}

static torch::Tensor arange_impl(
    const c10::Scalar& end, std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
    return arange_start_step_impl(c10::Scalar(0), end, c10::Scalar(1),
                                  dtype_opt, layout_opt, device_opt,
                                  pin_memory_opt);
}

static torch::Tensor arange_start_impl(
    const c10::Scalar& start, const c10::Scalar& end,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
    return arange_start_step_impl(start, end, c10::Scalar(1), dtype_opt,
                                  layout_opt, device_opt, pin_memory_opt);
}

static uint32_t copy_dtype(c10::ScalarType dtype) {
    switch (dtype) {
    case at::kFloat: return TORCH_COPY_F32;
    case at::kDouble: return TORCH_COPY_F64;
    case at::kHalf: return TORCH_COPY_F16;
    case at::kBFloat16: return TORCH_COPY_BF16;
    case at::kInt: return TORCH_COPY_I32;
    case at::kLong: return TORCH_COPY_I64;
    case at::kBool: return TORCH_COPY_BOOL;
    default:
        TORCH_CHECK(false, "torch_vortex: copy/to does not support dtype ", dtype);
    }
}

// copy_: same and converted dtypes, broadcast-compatible shapes, and arbitrary
// strides are handled by the device copy kernel. A linear hipMemcpy remains
// the fast path for equal dtype and equal contiguous shapes.
static void launch_copy_strided(torch::Tensor& dst, const torch::Tensor& src);

static torch::Tensor& copy_impl(torch::Tensor& self, const torch::Tensor& src,
                                bool non_blocking) {
    const bool self_dev = self.is_privateuseone();
    const bool src_dev = src.is_privateuseone();
    TORCH_CHECK(self_dev || src_dev,
                "torch_vortex: copy_ reached the vortex backend with neither "
                "side on the vortex device");

    if (self.is_same(src)) {
        return self;  // self-copy is a no-op, and hipMemcpy would be UB
    }

    // The direction is a property of the *pair*, not of self. The old ternary
    // labelled (CPU, vortex) as DeviceToHost; it happened to be right only
    // because that was the case it was ever reached in.
    hipMemcpyKind kind;
    if (self_dev && src_dev) {
        kind = hipMemcpyDeviceToDevice;
    } else if (self_dev) {
        kind = hipMemcpyHostToDevice;
    } else {
        kind = hipMemcpyDeviceToHost;
    }

    copy_dtype(self.scalar_type());
    copy_dtype(src.scalar_type());
    TORCH_CHECK(src.dim() <= self.dim(),
                "torch_vortex: copy_ source shape ", src.sizes(),
                " cannot broadcast to destination ", self.sizes());
    const int64_t rank_delta = self.dim() - src.dim();
    for (int64_t d = 0; d < src.dim(); ++d) {
        const int64_t source_extent = src.size(d);
        const int64_t destination_extent = self.size(d + rank_delta);
        TORCH_CHECK(source_extent == 1 || source_extent == destination_extent,
                    "torch_vortex: copy_ shape mismatch: source ", src.sizes(),
                    " cannot broadcast to destination ", self.sizes());
    }

    if (self.numel() == 0) {
        return self;
    }
    if (non_blocking) {
        // Honest rather than silent: hipMemcpy enqueues and then waits on its
        // own completion event, so this call blocks whatever the flag says.
        ++g_stats.blocking_syncs;
    }

    // A non-contiguous CPU source is materialised on the host first: it is a
    // pure ATen op, and it is what makes `t.t().to("vortex")` work rather than
    // silently copying storage order.
    torch::Tensor src_t = src;
    if (!src_dev && !src.is_contiguous()) {
        src_t = src.contiguous();
        ++g_stats.host_numeric_ops;
    }

    const bool same_dtype = self.scalar_type() == src.scalar_type();
    const bool same_shape = self.sizes() == src_t.sizes();
    if (same_dtype && same_shape && self.is_contiguous() &&
        self.storage_offset() == 0 && src_t.is_contiguous() &&
        src_t.storage_offset() == 0) {
        const int64_t bytes = self.numel() * self.element_size();
        VX_CHECK(hipMemcpy(self.data_ptr(), src_t.data_ptr(), (size_t)bytes, kind));
        // hipMemcpy enqueues and then waits on its own completion event; the
        // queue is FIFO, so everything enqueued before it has retired by the
        // time it returns. The allocator's fast path relies on knowing that.
        // It drains the *default* queue only, which is why the fast path also
        // tracks whether anything was launched on an explicit stream.
        note_default_queue_barrier();
        switch (kind) {
            case hipMemcpyHostToDevice: g_stats.h2d_bytes += (uint64_t)bytes; break;
            case hipMemcpyDeviceToHost: g_stats.d2h_bytes += (uint64_t)bytes; break;
            default: g_stats.d2d_bytes += (uint64_t)bytes; break;
        }
        return self;
    }

    // Conversion and strided addressing happen on device memory. Host inputs
    // are staged with their original dtype, never with the destination dtype.
    torch::Tensor device_src = src_t;
    if (!src_dev) {
        device_src = torch::empty(src_t.sizes(),
                                  self.options().dtype(src.scalar_type()));
        const int64_t src_bytes = src_t.numel() * src_t.element_size();
        VX_CHECK(hipMemcpy(device_src.data_ptr(), src_t.data_ptr(),
                           (size_t)src_bytes,
                           hipMemcpyHostToDevice));
        g_stats.h2d_bytes += (uint64_t)src_bytes;
        note_default_queue_barrier();
    }

    torch::Tensor device_dst = self;
    if (!self_dev) {
        device_dst = torch::empty(self.sizes(),
                                  self.options().device(c10::Device(
                                      c10::DeviceType::PrivateUse1, 0)));
    }
    launch_copy_strided(device_dst, device_src);

    if (!self_dev) {
        const int64_t dst_bytes = device_dst.numel() * device_dst.element_size();
        auto staged = torch::empty(self.sizes(),
                                   torch::TensorOptions().dtype(self.scalar_type()));
        VX_CHECK(hipMemcpy(staged.data_ptr(), device_dst.data_ptr(),
                           (size_t)dst_bytes, hipMemcpyDeviceToHost));
        g_stats.d2h_bytes += (uint64_t)dst_bytes;
        note_default_queue_barrier();
        self.copy_(staged);
        return self;
    }

    g_stats.d2d_bytes += (uint64_t)(self.numel() * self.element_size());
    return self;
}

// Tensor.to()/cpu() route through aten::_copy_from (returns dst). The
// dispatcher canonical signature takes dst by const& (matches Tracer).
static torch::Tensor copy_from_impl(const torch::Tensor& self,
                                    const torch::Tensor& dst,
                                    bool non_blocking) {
    copy_impl(const_cast<torch::Tensor&>(dst), self, non_blocking);
    return dst;
}

static torch::Tensor& fill__impl(torch::Tensor& self, const c10::Scalar& value) {
    check_vortex_f32(self, "fill_");
    fill_args_t args = {(uint64_t)(uintptr_t)self.data_ptr(),
                        (uint32_t)self.numel(), value.to<float>(), 0};
    launch(h_fill_kernel, args, (uint32_t)((self.numel() + 3) / 4));
    return self;
}

static torch::Tensor& zero__impl(torch::Tensor& self) {
    return fill__impl(self, c10::Scalar(0.0));
}

// view must share storage with its input and reject whatever ATen rejects.
//
// Delegating to ATen's own implementation gets -1 resolution, the element
// count check, computeStride for non-contiguous inputs, the storage offset and
// the shared version counter. The previous hand-built TensorImpl got none of
// them: view(-1) produced a tensor with a negative size, view(1000) on a
// 4-element tensor was accepted, and a view of an offset tensor silently read
// from the storage base. `reshape` is deliberately not registered — ATen's
// CompositeImplicitAutograd kernel already implements view-or-copy, which is
// exactly the distinction that matters.
// as_strided is a metadata operation: it reinterprets the same storage. Like
// view, it delegates to ATen so the bounds and alias rules are ATen's rather
// than ours. It is what transpose/permute/t and any indexing go through, so
// without it those were all unreachable.
static torch::Tensor as_strided_impl(const torch::Tensor& self,
                                     c10::SymIntArrayRef sym_size,
                                     c10::SymIntArrayRef sym_stride,
                                     std::optional<c10::SymInt> storage_offset) {
    TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
                "torch_vortex: as_strided on non-vortex tensor");
    // as_strided_tensorimpl is the int64 view implementation; the symint entry
    // points ATen exports are the in-place and copy variants, neither of which
    // is a view. Concretising is safe here in a way it is not during shape
    // inference: this kernel is only reached for a real vortex tensor, and
    // FakeTensor/Meta dispatch to ATen's own meta kernel instead.
    std::vector<int64_t> sizes, strides;
    sizes.reserve(sym_size.size());
    for (const auto& s : sym_size) sizes.push_back(s.expect_int());
    strides.reserve(sym_stride.size());
    for (const auto& s : sym_stride) strides.push_back(s.expect_int());
    std::optional<int64_t> offset;
    if (storage_offset.has_value()) offset = storage_offset->expect_int();
    auto out = at::native::as_strided_tensorimpl(self, c10::IntArrayRef(sizes),
                                                 c10::IntArrayRef(strides), offset);
    TORCH_CHECK(out.device().type() == c10::DeviceType::PrivateUse1,
                "torch_vortex: as_strided produced a tensor on ", out.device());
    return out;
}

static torch::Tensor view_impl(const torch::Tensor& self,
                               c10::SymIntArrayRef sym_sizes) {
    TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
                "torch_vortex: view on non-vortex tensor");
#if VX_HAS_ACCELERATOR_GUARD_API
    auto out = at::native::view_symint(self, sym_sizes);
#else
    std::vector<int64_t> sizes;
    sizes.reserve(sym_sizes.size());
    for (const auto& s : sym_sizes) sizes.push_back(s.expect_int());
    auto out = at::native::view(self, c10::IntArrayRef(sizes));
#endif
    TORCH_CHECK(out.device().type() == c10::DeviceType::PrivateUse1,
                "torch_vortex: view produced a tensor on ", out.device());
    return out;
}

// relu, non-inplace. PyTorch's relu is max(x, 0) with two properties a naive
// `x > 0 ? x : 0` gets wrong: NaN propagates, and -0.0 stays -0.0. It also
// must not touch its input — this used to run the in-place kernel on self and
// return self, so a residual branch sharing the input was silently modified.
// ---- strided copy ---------------------------------------------------------
//
// hipMemcpy moves a linear range. A transposed view, a storage offset or a
// broadcast is not one, so those go through a kernel. This is what makes
// as_strided usable and what `t.t().to("vortex")` needs: _to_copy asks for a
// strided *destination*, which v1 used to refuse.

static void launch_copy_strided(torch::Tensor& dst, const torch::Tensor& src) {
    TORCH_CHECK(dst.dim() <= 4, "torch_vortex: copy supports at most 4 dims, got ",
                dst.dim());
    TORCH_CHECK(src.dim() <= dst.dim(), "torch_vortex: copy source rank ", src.dim(),
                " exceeds destination rank ", dst.dim());
    copy_strided_args_t args = {};
    args.dst = (uint64_t)(uintptr_t)dst.data_ptr();
    args.src = (uint64_t)(uintptr_t)src.data_ptr();
    args.ndim = (uint32_t)dst.dim();
    args.total = u32_numel(dst, "copy");
    args.dst_type = copy_dtype(dst.scalar_type());
    args.src_type = copy_dtype(src.scalar_type());
    const int64_t rank_delta = dst.dim() - src.dim();
    for (int64_t i = 0; i < dst.dim(); ++i) {
        args.sizes[i] = u32_dim(dst.size(i), "copy size");
        args.dst_strides[i] = u32_dim(dst.stride(i), "copy stride");
        if (i < rank_delta) {
            args.src_strides[i] = 0;
        } else {
            const int64_t src_dim = i - rank_delta;
            TORCH_CHECK(src.size(src_dim) == 1 || src.size(src_dim) == dst.size(i),
                        "torch_vortex: copy source shape ", src.sizes(),
                        " cannot broadcast to destination ", dst.sizes());
            args.src_strides[i] = src.size(src_dim) == 1
                                      ? 0
                                      : u32_dim(src.stride(src_dim), "copy stride");
        }
    }
    launch(h_copy_strided_kernel, args, (args.total + 3) / 4);
}

static uint32_t checked_product(c10::IntArrayRef sizes, int64_t begin,
                                int64_t end, const char* what) {
    uint64_t product = 1;
    for (int64_t i = begin; i < end; ++i) {
        TORCH_CHECK(sizes[i] >= 0, "torch_vortex: ", what,
                    " has a negative extent");
        const uint64_t extent = (uint64_t)sizes[i];
        TORCH_CHECK(extent == 0 || product <= UINT32_MAX / extent,
                    "torch_vortex: ", what,
                    " does not fit in uint32");
        product *= extent;
    }
    return (uint32_t)product;
}

static void launch_cat_piece(const torch::Tensor& src, torch::Tensor& dst,
                             uint32_t outer, uint32_t src_dim,
                             uint32_t out_dim, uint32_t inner,
                             uint32_t dst_offset) {
    const uint32_t total = u32_numel(src, "cat input");
    if (total == 0) {
        ++g_stats.skipped_launches;
        return;
    }
    cat_args_t args = {(uint64_t)(uintptr_t)dst.data_ptr(),
                       (uint64_t)(uintptr_t)src.data_ptr(), outer, src_dim,
                       out_dim, inner, dst_offset, total};
    launch(h_cat_kernel, args, (total + 3) / 4);
}

static int64_t normalize_cat_dim(int64_t dim, int64_t ndim, const char* name,
                                 bool allow_insert) {
    const int64_t limit = allow_insert ? ndim + 1 : ndim;
    TORCH_CHECK(ndim > 0, "torch_vortex: ", name,
                " expects tensors with at least one dimension");
    TORCH_CHECK(dim >= -limit && dim < limit, "torch_vortex: ", name,
                " dim ", dim, " is out of range for ", ndim, "-D input");
    return dim < 0 ? dim + limit : dim;
}

static torch::Tensor cat_impl(const at::ITensorListRef& tensors,
                              int64_t dim) {
    TORCH_CHECK(tensors.size() > 0, "torch_vortex: cat expects a non-empty list");
    const auto materialized = tensors.materialize();
    const torch::Tensor& first = materialized[0].get();
    check_vortex_f32(first, "cat input");
    const int64_t ndim = first.dim();
    const int64_t d = normalize_cat_dim(dim, ndim, "cat", false);
    std::vector<int64_t> out_sizes(first.sizes().begin(), first.sizes().end());
    int64_t cat_size = first.size(d);
    for (size_t i = 1; i < materialized.size(); ++i) {
        const auto& t = materialized[i].get();
        check_vortex_f32(t, "cat input");
        TORCH_CHECK(t.dim() == ndim, "torch_vortex: cat inputs must have the same rank");
        for (int64_t j = 0; j < ndim; ++j) {
            TORCH_CHECK(j == d || t.size(j) == first.size(j),
                        "torch_vortex: cat input shape mismatch at dimension ", j);
        }
        TORCH_CHECK(cat_size <= INT64_MAX - t.size(d),
                    "torch_vortex: cat output dimension overflows int64");
        cat_size += t.size(d);
    }
    out_sizes[d] = cat_size;
    auto out = torch::empty(out_sizes, first.options());
    u32_numel(out, "cat output");
    const uint32_t outer = checked_product(first.sizes(), 0, d, "cat outer");
    const uint32_t inner = checked_product(first.sizes(), d + 1, ndim, "cat inner");
    const uint32_t out_dim = u32_dim(cat_size, "cat output dimension");
    uint32_t offset = 0;
    for (size_t i = 0; i < materialized.size(); ++i) {
        const auto& t = materialized[i].get();
        const uint32_t src_dim = u32_dim(t.size(d), "cat input dimension");
        launch_cat_piece(t, out, outer, src_dim, out_dim, inner, offset);
        TORCH_CHECK(offset <= UINT32_MAX - src_dim,
                    "torch_vortex: cat offset does not fit in uint32");
        offset += src_dim;
    }
    return out;
}

static torch::Tensor stack_impl(at::TensorList tensors, int64_t dim) {
    TORCH_CHECK(tensors.size() > 0, "torch_vortex: stack expects a non-empty list");
    const torch::Tensor& first = tensors[0];
    check_vortex_f32(first, "stack input");
    const int64_t ndim = first.dim();
    const int64_t d = normalize_cat_dim(dim, ndim, "stack", true);
    for (size_t i = 1; i < tensors.size(); ++i) {
        check_vortex_f32(tensors[i], "stack input");
        TORCH_CHECK(tensors[i].sizes() == first.sizes(),
                    "torch_vortex: stack inputs must have the same shape");
    }
    std::vector<int64_t> out_sizes(first.sizes().begin(), first.sizes().end());
    out_sizes.insert(out_sizes.begin() + d, (int64_t)tensors.size());
    auto out = torch::empty(out_sizes, first.options());
    u32_numel(out, "stack output");
    const uint32_t outer = checked_product(first.sizes(), 0, d, "stack outer");
    const uint32_t inner = checked_product(first.sizes(), d, ndim, "stack inner");
    const uint32_t out_dim = u32_dim(tensors.size(), "stack count");
    for (size_t i = 0; i < tensors.size(); ++i) {
        launch_cat_piece(tensors[i], out, outer, 1, out_dim, inner,
                         u32_dim(i, "stack index"));
    }
    return out;
}

static torch::Tensor& cat_out_impl(const at::ITensorListRef& tensors,
                                   int64_t dim, torch::Tensor& out) {
    check_vortex_f32(out, "cat out");
    auto tmp = cat_impl(tensors, dim);
    TORCH_CHECK(out.sizes() == tmp.sizes(), "torch_vortex: cat out shape ",
                out.sizes(), " does not match result ", tmp.sizes());
    out.copy_(tmp);
    return out;
}

static torch::Tensor& stack_out_impl(at::TensorList tensors, int64_t dim,
                                     torch::Tensor& out) {
    check_vortex_f32(out, "stack out");
    auto tmp = stack_impl(tensors, dim);
    TORCH_CHECK(out.sizes() == tmp.sizes(), "torch_vortex: stack out shape ",
                out.sizes(), " does not match result ", tmp.sizes());
    out.copy_(tmp);
    return out;
}

static uint32_t check_index_tensor(const torch::Tensor& index,
                                   const char* name) {
    TORCH_CHECK(index.device().type() == c10::DeviceType::PrivateUse1,
                "torch_vortex: ", name, " must be a vortex tensor");
    TORCH_CHECK(index.scalar_type() == at::kInt || index.scalar_type() == at::kLong,
                "torch_vortex: ", name, " must have int32 or int64 dtype, got ",
                index.scalar_type());
    TORCH_CHECK(index.is_contiguous(), "torch_vortex: ", name,
                " must be contiguous in v1");
    TORCH_CHECK(index.dim() <= 4, "torch_vortex: ", name,
                " supports at most 4 dimensions, got ", index.dim());
    return u32_numel(index, name);
}

static int64_t normalize_index_dim(int64_t dim, int64_t ndim,
                                   const char* name) {
    return normalize_cat_dim(dim, ndim, name, false);
}

static void check_index_shape(const torch::Tensor& self,
                              const torch::Tensor& index, int64_t dim,
                              const char* name) {
    TORCH_CHECK(self.dim() == index.dim(), "torch_vortex: ", name,
                " index must have the same number of dimensions as self");
    for (int64_t d = 0; d < self.dim(); ++d) {
        TORCH_CHECK(d == dim || index.size(d) <= self.size(d),
                    "torch_vortex: ", name, " index shape exceeds self at dimension ", d);
    }
}

static index_args_t make_index_args(const torch::Tensor& dst,
                                    const torch::Tensor& src,
                                    const torch::Tensor& index,
                                    const torch::Tensor& invalid,
                                    const torch::Tensor& layout,
                                    int64_t dim, uint32_t total,
                                    const char* name) {
    index_args_t args = {};
    args.dst = (uint64_t)(uintptr_t)dst.data_ptr();
    args.src = (uint64_t)(uintptr_t)src.data_ptr();
    args.index = (uint64_t)(uintptr_t)index.data_ptr();
    args.invalid = (uint64_t)(uintptr_t)invalid.data_ptr();
    args.ndim = u32_dim(index.dim(), name);
    args.dim = u32_dim(dim, name);
    args.dim_size = u32_dim(layout.size(dim), name);
    args.index_type = index.scalar_type() == at::kInt ? 0 : 1;
    args.total = total;
    for (int64_t d = 0; d < index.dim(); ++d) {
        args.sizes[d] = u32_dim(index.size(d), name);
        args.strides[d] = u32_dim(layout.stride(d), name);
    }
    return args;
}

static torch::Tensor make_index_error_flag(const torch::Tensor& ref) {
    auto flag = torch::empty({1}, ref.options().dtype(at::kInt));
    VX_CHECK(hipMemsetAsync(flag.data_ptr(), 0, sizeof(int32_t),
                            current_hip_stream()));
    return flag;
}

static void check_index_error_flag(const torch::Tensor& flag, const char* name) {
    int32_t invalid = 0;
    VX_CHECK(hipMemcpy(&invalid, flag.data_ptr(), sizeof(invalid),
                       hipMemcpyDeviceToHost));
    g_stats.d2h_bytes += sizeof(invalid);
    if (current_hip_stream() == nullptr) {
        note_default_queue_barrier();
    } else {
        VX_CHECK(hipStreamSynchronize(current_hip_stream()));
    }
    TORCH_CHECK(invalid == 0, "torch_vortex: ", name,
                " index contains an out-of-range value");
}

static torch::Tensor gather_impl(const torch::Tensor& self, int64_t dim,
                                 const torch::Tensor& index, bool sparse_grad) {
    check_vortex_f32(self, "gather input");
    u32_numel(self, "gather input");
    TORCH_CHECK(!sparse_grad, "torch_vortex: gather sparse_grad is unsupported");
    TORCH_CHECK(self.dim() > 0,
                "torch_vortex: gather expects a tensor with at least one dimension");
    const int64_t d = normalize_index_dim(dim, self.dim(), "gather");
    const uint32_t total = check_index_tensor(index, "gather index");
    check_index_shape(self, index, d, "gather");
    TORCH_CHECK(total == 0 || self.size(d) > 0,
                "torch_vortex: gather index has no valid source dimension");
    auto out = torch::empty(index.sizes(), self.options());
    if (total == 0) {
        ++g_stats.skipped_launches;
        return out;
    }
    auto invalid = make_index_error_flag(self);
    auto args = make_index_args(out, self, index, invalid, self, d, total,
                                "gather");
    launch(h_gather_kernel, args, (total + 3) / 4);
    check_index_error_flag(invalid, "gather");
    return out;
}

static torch::Tensor& gather_out_impl(const torch::Tensor& self, int64_t dim,
                                      const torch::Tensor& index,
                                      bool sparse_grad, torch::Tensor& out) {
    check_vortex_f32(out, "gather out");
    auto tmp = gather_impl(self, dim, index, sparse_grad);
    TORCH_CHECK(out.sizes() == tmp.sizes(), "torch_vortex: gather out shape ",
                out.sizes(), " does not match result ", tmp.sizes());
    out.copy_(tmp);
    return out;
}

static torch::Tensor scatter_impl(const torch::Tensor& self, int64_t dim,
                                  const torch::Tensor& index,
                                  const torch::Tensor& src) {
    check_vortex_f32(self, "scatter input");
    check_vortex_f32(src, "scatter source");
    TORCH_CHECK(self.dim() > 0,
                "torch_vortex: scatter expects a tensor with at least one dimension");
    const int64_t d = normalize_index_dim(dim, self.dim(), "scatter");
    const uint32_t total = check_index_tensor(index, "scatter index");
    check_index_shape(self, index, d, "scatter");
    TORCH_CHECK(src.sizes() == index.sizes(),
                "torch_vortex: scatter source and index must have the same shape");
    u32_numel(self, "scatter input");
    auto out = torch::empty(self.sizes(), self.options());
    if (self.numel() != 0) {
        const size_t bytes = (size_t)self.numel() * sizeof(float);
        VX_CHECK(hipMemcpy(out.data_ptr(), self.data_ptr(), bytes,
                           hipMemcpyDeviceToDevice));
        g_stats.d2d_bytes += bytes;
        note_default_queue_barrier();
    }
    if (total == 0) {
        ++g_stats.skipped_launches;
        return out;
    }
    auto invalid = make_index_error_flag(self);
    auto args = make_index_args(out, src, index, invalid, self, d, total,
                                "scatter");
    launch(h_scatter_kernel, args, (total + 3) / 4);
    check_index_error_flag(invalid, "scatter");
    return out;
}

static torch::Tensor& scatter_src_out_impl(const torch::Tensor& self,
                                           int64_t dim,
                                           const torch::Tensor& index,
                                           const torch::Tensor& src,
                                           torch::Tensor& out) {
    check_vortex_f32(out, "scatter out");
    auto tmp = scatter_impl(self, dim, index, src);
    TORCH_CHECK(out.sizes() == tmp.sizes(), "torch_vortex: scatter out shape ",
                out.sizes(), " does not match result ", tmp.sizes());
    out.copy_(tmp);
    return out;
}

static torch::Tensor& scatter__impl(torch::Tensor& self, int64_t dim,
                                    const torch::Tensor& index,
                                    const torch::Tensor& src) {
    auto tmp = scatter_impl(self, dim, index, src);
    self.copy_(tmp);
    return self;
}

static torch::Tensor index_add_impl(const torch::Tensor& self, int64_t dim,
                                    const torch::Tensor& index,
                                    const torch::Tensor& source,
                                    const c10::Scalar& alpha) {
    check_vortex_f32(self, "index_add input");
    check_vortex_f32(source, "index_add source");
    TORCH_CHECK(self.dim() > 0,
                "torch_vortex: index_add expects a tensor with at least one dimension");
    const int64_t d = normalize_index_dim(dim, self.dim(), "index_add");
    const uint32_t index_count = check_index_tensor(index, "index_add index");
    TORCH_CHECK(index.dim() == 1,
                "torch_vortex: index_add index must be one-dimensional");
    TORCH_CHECK(source.dim() == self.dim(),
                "torch_vortex: index_add source must have the same rank as self");
    for (int64_t j = 0; j < self.dim(); ++j) {
        TORCH_CHECK(j == d || source.size(j) == self.size(j),
                    "torch_vortex: index_add source shape mismatch at dimension ", j);
    }
    TORCH_CHECK(source.size(d) == (int64_t)index_count,
                "torch_vortex: index_add source dimension must equal index length");
    const uint32_t total = u32_numel(self, "index_add input");
    auto out = torch::empty(self.sizes(), self.options());
    if (index_count == 0) {
        if (total != 0) {
            const size_t bytes = (size_t)self.numel() * sizeof(float);
            VX_CHECK(hipMemcpy(out.data_ptr(), self.data_ptr(), bytes,
                               hipMemcpyDeviceToDevice));
            g_stats.d2d_bytes += bytes;
            note_default_queue_barrier();
        }
        ++g_stats.skipped_launches;
        return out;
    }

    index_add_args_t args = {};
    args.dst = (uint64_t)(uintptr_t)out.data_ptr();
    args.self = (uint64_t)(uintptr_t)self.data_ptr();
    args.src = (uint64_t)(uintptr_t)source.data_ptr();
    args.index = (uint64_t)(uintptr_t)index.data_ptr();
    auto invalid = make_index_error_flag(self);
    args.invalid = (uint64_t)(uintptr_t)invalid.data_ptr();
    args.ndim = u32_dim(self.dim(), "index_add rank");
    args.dim = u32_dim(d, "index_add dim");
    args.dim_size = u32_dim(self.size(d), "index_add dimension");
    args.index_type = index.scalar_type() == at::kInt ? 0 : 1;
    args.index_count = index_count;
    args.total = total;
    args.alpha = alpha.to<float>();
    for (int64_t j = 0; j < self.dim(); ++j) {
        args.sizes[j] = u32_dim(self.size(j), "index_add size");
        args.self_strides[j] = u32_dim(self.stride(j), "index_add stride");
        args.src_strides[j] = u32_dim(source.stride(j), "index_add source stride");
    }
    if (total != 0) {
        launch(h_index_add_kernel, args, (total + 3) / 4);
        check_index_error_flag(invalid, "index_add");
    }
    return out;
}

static torch::Tensor& index_add_out_impl(const torch::Tensor& self, int64_t dim,
                                         const torch::Tensor& index,
                                         const torch::Tensor& source,
                                         const c10::Scalar& alpha,
                                         torch::Tensor& out) {
    check_vortex_f32(out, "index_add out");
    auto tmp = index_add_impl(self, dim, index, source, alpha);
    TORCH_CHECK(out.sizes() == tmp.sizes(), "torch_vortex: index_add out shape ",
                out.sizes(), " does not match result ", tmp.sizes());
    out.copy_(tmp);
    return out;
}

static torch::Tensor& index_add__impl(torch::Tensor& self, int64_t dim,
                                      const torch::Tensor& index,
                                      const torch::Tensor& source,
                                      const c10::Scalar& alpha) {
    auto tmp = index_add_impl(self, dim, index, source, alpha);
    self.copy_(tmp);
    return self;
}

static std::tuple<torch::Tensor, torch::Tensor> nll_loss_forward_impl(
    const torch::Tensor& input, const torch::Tensor& target,
    const std::optional<torch::Tensor>& weight, int64_t reduction,
    c10::SymInt ignore_index) {
    check_vortex_f32(input, "nll_loss input");
    TORCH_CHECK(input.dim() == 1 || input.dim() == 2,
                "torch_vortex: nll_loss input must be 1-D or 2-D");
    TORCH_CHECK(target.device().type() == c10::DeviceType::PrivateUse1 &&
                    target.scalar_type() == at::kLong && target.is_contiguous(),
                "torch_vortex: nll_loss target must be contiguous Vortex int64");
    const uint32_t rows = input.dim() == 1 ? 1 : u32_dim(input.size(0),
                                                          "nll_loss rows");
    const uint32_t classes = u32_dim(input.dim() == 1 ? input.size(0)
                                                       : input.size(1),
                                      "nll_loss classes");
    TORCH_CHECK(classes > 0, "torch_vortex: nll_loss classes must be nonzero");
    TORCH_CHECK((input.dim() == 1 && target.dim() == 0) ||
                    (input.dim() == 2 && target.dim() == 1 &&
                     target.size(0) == input.size(0)),
                "torch_vortex: nll_loss target shape does not match input");
    TORCH_CHECK(reduction >= 0 && reduction <= 2,
                "torch_vortex: nll_loss reduction must be none, mean or sum");
    uint64_t weight_addr = 0;
    if (weight.has_value() && weight->defined()) {
        check_vortex_f32(*weight, "nll_loss weight");
        TORCH_CHECK(weight->dim() == 1 && weight->size(0) == classes,
                    "torch_vortex: nll_loss weight must have one value per class");
        weight_addr = (uint64_t)(uintptr_t)weight->data_ptr();
    }
    const auto out_shape = (reduction == 0 && input.dim() == 2)
                               ? std::vector<int64_t>{(int64_t)rows}
                               : std::vector<int64_t>{};
    auto out = torch::empty(out_shape, input.options());
    auto total_weight = torch::empty({}, input.options());
    auto invalid = make_index_error_flag(input);
    nll_loss_args_t args = {};
    args.dst = (uint64_t)(uintptr_t)out.data_ptr();
    args.total_weight = (uint64_t)(uintptr_t)total_weight.data_ptr();
    args.input = (uint64_t)(uintptr_t)input.data_ptr();
    args.target = (uint64_t)(uintptr_t)target.data_ptr();
    args.weight = weight_addr;
    args.invalid = (uint64_t)(uintptr_t)invalid.data_ptr();
    args.rows = rows;
    args.classes = classes;
    args.reduction = (uint32_t)reduction;
    args.ignore_index = ignore_index.expect_int();
    launch(h_nll_loss_kernel, args, 1);
    check_index_error_flag(invalid, "nll_loss");
    return std::make_tuple(out, total_weight);
}

// ---- reductions -----------------------------------------------------------
//
// The kernel reduces the trailing dimension of a contiguous (rows, cols)
// view. Everything else -- a full reduction, a middle dimension -- is
// normalised into that shape on the host with movedim+contiguous before it
// gets there, so the kernel never has to know about strides and stays in the
// simple shape VOLT compiles correctly.
//
// The normalising copy costs a pass. A stride-aware reduction kernel is the
// obvious next step (W7.1 measures whether it matters); correctness first.

// The reduce ops are the DL prim library's (vx_prim_op's reduction half), so
// there is no local enum here: a second numbering is how the host and the
// kernel start disagreeing about what "1" means.

// The shape the result has, which ATen would otherwise compute for us.
static std::vector<int64_t> reduced_shape(const torch::Tensor& self,
                                          c10::OptionalArrayRef<int64_t> dim,
                                          bool keepdim) {
    const int64_t nd = self.dim();
    std::vector<int64_t> s(self.sizes().begin(), self.sizes().end());
    const bool all = !dim.has_value() || dim->size() == 0;
    if (keepdim) {
        for (int64_t i = 0; i < nd; ++i) {
            if (all) {
                s[i] = 1;
            } else {
                for (auto d : *dim) {
                    if ((d < 0 ? d + nd : d) == i) s[i] = 1;
                }
            }
        }
        return s;
    }
    if (all) {
        return {};
    }
    std::vector<int64_t> out;
    for (int64_t i = 0; i < nd; ++i) {
        bool reduced = false;
        for (auto d : *dim) {
            if ((d < 0 ? d + nd : d) == i) reduced = true;
        }
        if (!reduced) out.push_back(s[i]);
    }
    return out;
}

// Validate the dims before anything is allocated, so a bad call fails without
// having sized a result first. The layout helper repeats these checks; this
// only exists to run them earlier.
static void check_reduce_dims(const torch::Tensor& self,
                              c10::OptionalArrayRef<int64_t> dim,
                              const char* name) {
    const int64_t nd = self.dim();
    TORCH_CHECK(nd <= 4, "torch_vortex: ", name,
                " supports at most 4 dimensions in v1, got ", nd);
    if (!dim.has_value() || dim->size() == 0) {
        return;
    }
    TORCH_CHECK(dim->size() == 1, "torch_vortex: ", name, " over several "
                "dimensions at once is unsupported in v1; reduce one at a "
                "time (W3.2 in docs/mydocs/pytorch_plan.md)");
    if (nd == 0) {
        TORCH_CHECK((*dim)[0] == 0 || (*dim)[0] == -1,
                    "torch_vortex: ", name, " dim ", (*dim)[0],
                    " is out of range for a 0-D tensor");
        return;
    }
    int64_t d = (*dim)[0];
    if (d < 0) d += nd;
    TORCH_CHECK(d >= 0 && d < nd, "torch_vortex: ", name, " dim ", (*dim)[0],
                " is out of range for a ", nd, "-D tensor");
}

// A contiguous (rows, cols) view whose trailing dimension is the one being
// reduced. A full reduction is one row of everything.
static torch::Tensor reduce_layout(const torch::Tensor& self,
                                   c10::OptionalArrayRef<int64_t> dim,
                                   const char* name) {
    const int64_t nd = self.dim();
    if (nd > 4) {
        TORCH_CHECK(false, "torch_vortex: ", name,
                    " supports at most 4 dimensions in v1, got ", nd);
    }
    if (!dim.has_value() || dim->size() == 0) {
        return self.reshape({1, -1}).contiguous();
    }
    TORCH_CHECK(dim->size() == 1, "torch_vortex: ", name, " over several "
                "dimensions at once is unsupported in v1; reduce one at a "
                "time (W3.2 in docs/mydocs/pytorch_plan.md)");
    if (nd == 0) {
        const int64_t d = (*dim)[0];
        TORCH_CHECK(d == 0 || d == -1, "torch_vortex: ", name, " dim ", d,
                    " is out of range for a 0-D tensor");
        return self.reshape({1, 1}).contiguous();
    }
    int64_t d = (*dim)[0];
    if (d < 0) d += nd;
    TORCH_CHECK(d >= 0 && d < nd, "torch_vortex: ", name, " dim ", (*dim)[0],
                " is out of range for a ", nd, "-D tensor");
    auto moved = (d == nd - 1) ? self : self.movedim(d, nd - 1);
    auto contig = moved.contiguous();
    const int64_t cols = contig.size(nd - 1);
    TORCH_CHECK(cols > 0, "torch_vortex: ", name, " on an empty dimension");
    return contig.reshape({contig.numel() / cols, cols});
}

static torch::Tensor& reduce_into(const torch::Tensor& self, torch::Tensor& out,
                                  c10::OptionalArrayRef<int64_t> dim,
                                  uint32_t op, const char* name) {
    check_vortex_f32(self, name);
    TORCH_CHECK(out.device().type() == c10::DeviceType::PrivateUse1,
                "torch_vortex: ", name, " out must be a vortex tensor");
    TORCH_CHECK(out.scalar_type() == at::kFloat && out.is_contiguous(),
                "torch_vortex: ", name, " out must be contiguous float32");
    // holding the normalised view alive across the launch: the allocator
    // defers its release behind the queue now, but the tensor still has to
    // exist when the kernel reads it
    auto src = reduce_layout(self, dim, name);
    TORCH_CHECK(out.numel() == src.size(0), "torch_vortex: ", name, " out has ",
                out.numel(), " elements but the reduction produces ",
                src.size(0));
    // Nothing to reduce is not a launch, and the DL entry point refuses both
    // zero shapes rather than accepting an empty job. The local kernel this
    // replaced expressed neither: a zero row count came out as a zero grid and
    // was skipped, while a zero row length launched a CTA whose loop body
    // never ran, leaving the accumulator at its seed. Both answers are now
    // stated rather than inherited from that.
    if (src.size(0) == 0) {
        // Zero rows: the result is empty, so there is nothing to write.
        ++g_stats.skipped_launches;
        return out;
    }
    if (src.size(1) == 0) {
        // One row of zero elements. sum and mean still have answers, and no
        // kernel can produce them -- there is nothing to accumulate -- so the
        // identity is written instead. max has no identity, and torch refuses
        // the same call with "amax(): Expected reduction dim to be specified
        // for input.numel() == 0"; agreeing with that beats inventing a value.
        // (The seed the old kernel left behind was -INFINITY, which torch never
        // returns.)
        TORCH_CHECK(op != VX_PRIM_OP_MAX && op != VX_PRIM_OP_MIN,
                    "torch_vortex: ", name,
                    " of a tensor with no elements has no result; torch amax "
                    "requires a reduction dim for an empty input");
        fill_args_t args = {(uint64_t)(uintptr_t)out.data_ptr(),
                            u32_numel(out, name),
                            op == VX_PRIM_OP_MEAN ? (float)NAN
                                                  : (op == VX_PRIM_OP_MIN
                                                         ? (float)INFINITY : 0.0f),
                            0};
        launch(h_fill_kernel, args, (args.n + 3) / 4);
        return out;
    }
    DL_LAUNCH(vx_prim_reduce(current_queue(), (vx_prim_op)op,
                             (uint64_t)(uintptr_t)src.data_ptr(),
                             (uint64_t)(uintptr_t)out.data_ptr(),
                             u32_dim(src.size(0), "reduce rows"),
                             u32_dim(src.size(1), "reduce columns")));
    return out;
}

static void check_reduce_dtype(std::optional<c10::ScalarType> dtype,
                               const char* name) {
    TORCH_CHECK(!dtype.has_value() || *dtype == at::kFloat,
                "torch_vortex: ", name, " dtype ", dtype.value(),
                " is unsupported; only float32 is implemented (W3.2)");
}

// ---- index reductions: argmax/argmin and max/min(dim=) ---------------------
//
// One DL pass produces both the index and the extreme it points at, so
// max(dim=)'s pair costs no more than argmax's index alone. The index comes
// back as uint32 -- that is vx_prim_reduce's contract -- and ATen wants int64,
// which the widen kernel converts.

static void launch_widen_u32_i64(uint64_t dst, uint64_t src, uint32_t n) {
    // Nothing to widen is not a launch, and launch() would skip a zero grid
    // anyway; this says so before allocating the argument block.
    if (n == 0) {
        ++g_stats.skipped_launches;
        return;
    }
    widen_args_t args = {dst, src, n, 0};
    launch(h_widen_u32_i64_kernel, args, (n + 3) / 4);
}

// `indices` is filled in the caller's shape; `values`, when given, receives the
// extreme at each index and must already be shaped and float32.
static void index_reduce(const torch::Tensor& self,
                         c10::OptionalArrayRef<int64_t> dim,
                         torch::Tensor& indices, torch::Tensor* values,
                         uint32_t op, const char* name) {
    check_vortex_f32(self, name);
    TORCH_CHECK(indices.scalar_type() == at::kLong && indices.is_contiguous(),
                "torch_vortex: ", name, " indices must be contiguous int64");
    // torch refuses both empty shapes rather than answering one, and with
    // different messages: no dim at all wants a dim named, and an empty
    // reduction dim wants a different input. The layout helper catches the
    // second; the first is only reachable through the full-reduction form.
    if (!dim.has_value() || dim->size() == 0) {
        TORCH_CHECK(self.numel() > 0, "torch_vortex: ", name, " of a tensor with "
                    "no elements has no index; torch requires a reduction dim "
                    "for an empty input");
    }
    auto src = reduce_layout(self, dim, name);
    const int64_t rows = src.size(0), cols = src.size(1);
    TORCH_CHECK(indices.numel() == rows, "torch_vortex: ", name, " indices has ",
                indices.numel(), " elements but the reduction produces ", rows);
    if (values) {
        TORCH_CHECK(values->numel() == rows && values->is_contiguous() &&
                    values->scalar_type() == at::kFloat,
                    "torch_vortex: ", name, " values must be contiguous float32 "
                    "with one element per row");
    }
    if (rows == 0) {
        ++g_stats.skipped_launches;
        return;
    }
    auto stage = torch::empty({rows}, self.options().dtype(at::kInt));
    DL_LAUNCH(vx_prim_index_reduce(
        current_queue(), (vx_prim_op)op, (uint64_t)(uintptr_t)src.data_ptr(),
        (uint64_t)(uintptr_t)stage.data_ptr(),
        values ? (uint64_t)(uintptr_t)values->data_ptr() : 0,
        u32_dim(rows, "argmax rows"), u32_dim(cols, "argmax cols")));
    launch_widen_u32_i64((uint64_t)(uintptr_t)indices.data_ptr(),
                         (uint64_t)(uintptr_t)stage.data_ptr(),
                         u32_dim(rows, "argmax rows"));
}

static torch::Tensor argmax_impl(const torch::Tensor& self,
                                 std::optional<int64_t> dim, bool keepdim) {
    check_vortex_f32(self, "argmax");
    c10::OptionalArrayRef<int64_t> d;
    std::vector<int64_t> one;
    if (dim.has_value()) {
        one = {*dim};
        d = c10::OptionalArrayRef<int64_t>(one);
    }
    check_reduce_dims(self, d, "argmax");
    // No dim flattens the tensor and answers with a 0-d index, which is what
    // reduced_shape returns for an empty dim list.
    auto out = torch::empty(reduced_shape(self, d, keepdim),
                            self.options().dtype(at::kLong));
    index_reduce(self, d, out, nullptr, VX_PRIM_OP_ARGMAX, "argmax");
    return out;
}

static torch::Tensor argmin_impl(const torch::Tensor& self,
                                 std::optional<int64_t> dim, bool keepdim) {
    check_vortex_f32(self, "argmin");
    c10::OptionalArrayRef<int64_t> d;
    std::vector<int64_t> one;
    if (dim.has_value()) {
        one = {*dim};
        d = c10::OptionalArrayRef<int64_t>(one);
    }
    check_reduce_dims(self, d, "argmin");
    auto out = torch::empty(reduced_shape(self, d, keepdim),
                            self.options().dtype(at::kLong));
    index_reduce(self, d, out, nullptr, VX_PRIM_OP_ARGMIN, "argmin");
    return out;
}

static std::tuple<torch::Tensor, torch::Tensor> max_dim_impl(
    const torch::Tensor& self, int64_t dim, bool keepdim) {
    check_vortex_f32(self, "max");
    std::vector<int64_t> one = {dim};
    c10::OptionalArrayRef<int64_t> d = c10::OptionalArrayRef<int64_t>(one);
    check_reduce_dims(self, d, "max");
    auto shape = reduced_shape(self, d, keepdim);
    auto values = torch::empty(shape, self.options());
    auto indices = torch::empty(shape, self.options().dtype(at::kLong));
    index_reduce(self, d, indices, &values, VX_PRIM_OP_ARGMAX, "max");
    return std::make_tuple(values, indices);
}

static std::tuple<torch::Tensor, torch::Tensor> min_dim_impl(
    const torch::Tensor& self, int64_t dim, bool keepdim) {
    check_vortex_f32(self, "min");
    std::vector<int64_t> one = {dim};
    c10::OptionalArrayRef<int64_t> d = c10::OptionalArrayRef<int64_t>(one);
    check_reduce_dims(self, d, "min");
    auto shape = reduced_shape(self, d, keepdim);
    auto values = torch::empty(shape, self.options());
    auto indices = torch::empty(shape, self.options().dtype(at::kLong));
    index_reduce(self, d, indices, &values, VX_PRIM_OP_ARGMIN, "min");
    return std::make_tuple(values, indices);
}

// ---- softmax family -------------------------------------------------------
//
// softmax, log_softmax and logsumexp are one DL kernel with a selector: they
// share the row max, the shifted exponentials and the sum, and differ only in
// what is written. softmax and log_softmax keep every dimension, so unlike a
// reduction they need the moved layout twice -- once to gather a row, once to
// put the answer back.

// softmax's dim is a bare int, not a list, and a 0-d tensor accepts only 0 and
// -1 (torch: "dim 1 out of range [-1, 0]"). Everything else resolves against
// the rank like every other dim in this file.
static int64_t row_dim(const torch::Tensor& self, int64_t dim, const char* name) {
    const int64_t nd = self.dim();
    if (nd == 0) {
        TORCH_CHECK(dim == 0 || dim == -1, "torch_vortex: ", name, " dim ", dim,
                    " is out of range for a 0-D tensor, which takes 0 or -1");
        return 0;
    }
    const int64_t d = dim < 0 ? dim + nd : dim;
    TORCH_CHECK(d >= 0 && d < nd, "torch_vortex: ", name, " dim ", dim,
                " is out of range for a ", nd, "-D tensor");
    return d;
}

// The (rows, cols) view the DL row kernel takes, plus the way back.
//
// A trailing dim is already the view: the tensor is contiguous, so its rows
// are the kernel's rows and the output is written in place. Any other dim has
// to be moved to the end, which costs a copy in and a copy back; the write
// back goes through launch_copy_strided, which is where the 4-dimension cap
// comes from (the same one the reductions' layout has).
struct RowLayout {
    torch::Tensor in;     // contiguous (rows, cols)
    torch::Tensor out;    // contiguous (rows, cols): what the kernel writes
    torch::Tensor shaped; // the same storage as `out`, in the moved shape
    int64_t rows = 0;
    int64_t cols = 0;
    bool moved = false;   // `shaped` has to be copied back into the result
    int64_t dim = 0;
};

// `out` is the caller's result tensor, already allocated with the input's
// shape. The kernel addresses elements, so the only question this answers is
// which buffer those elements live in and in what order.
static RowLayout row_layout(const torch::Tensor& self, torch::Tensor& out,
                            int64_t d, const char* name) {
    RowLayout L;
    L.dim = d;
    const int64_t nd = self.dim();
    if (nd == 0) {
        // A scalar is one row of one column, and it comes back a scalar:
        // torch.softmax(tensor(3.0), 0) is 0-dimensional, not (1,1).
        L.in = self.reshape({1, 1});
        L.out = out;
        L.rows = L.cols = 1;
        return L;
    }
    TORCH_CHECK(nd <= 4, "torch_vortex: ", name, " supports at most 4 "
                "dimensions in v1, got ", nd);
    if (d == nd - 1) {
        L.cols = self.size(nd - 1);
        L.rows = L.cols == 0 ? 0 : self.numel() / L.cols;
        L.in = self;
        L.out = out;
        return L;
    }
    auto moved = self.movedim(d, nd - 1).contiguous();
    L.cols = moved.size(nd - 1);
    L.rows = L.cols == 0 ? 0 : moved.numel() / L.cols;
    L.in = moved.reshape({L.rows, L.cols});
    // Two views of one buffer: the kernel addresses elements, the copy back
    // needs the shape it is copying into.
    L.shaped = torch::empty_like(moved);
    L.out = L.shaped.reshape({L.rows, L.cols});
    L.moved = true;
    return L;
}

// The three DL row entries share a signature, so softmax and log_softmax run
// through one body and differ by which one is passed in.
using RowFn = vx_prim_status (*)(vx_queue_h, uint64_t, uint64_t, uint32_t,
                                 uint32_t);

static torch::Tensor softmax_op(const torch::Tensor& self, int64_t dim,
                                RowFn fn, const char* name) {
    check_vortex_f32(self, name);
    const int64_t d = row_dim(self, dim, name);
    auto out = torch::empty_like(self);
    // Nothing to normalise, and the DL entry point refuses a zero shape rather
    // than accepting an empty job.
    if (self.numel() == 0) {
        ++g_stats.skipped_launches;
        return out;
    }
    RowLayout L = row_layout(self, out, d, name);
    DL_LAUNCH(fn(current_queue(), (uint64_t)(uintptr_t)L.in.data_ptr(),
                 (uint64_t)(uintptr_t)L.out.data_ptr(),
                 u32_dim(L.rows, "softmax rows"),
                 u32_dim(L.cols, "softmax cols")));
    if (!L.moved) {
        return out;
    }
    // The kernel wrote rows x cols in moved order; the answer has to go back
    // to the caller's layout.
    out.movedim(d, self.dim() - 1).copy_(L.shaped);
    return out;
}

// half_to_float asks for the half-input/float-output form of the op, which
// needs half support; the answer here is float32 either way.
static void check_half_to_float(bool half_to_float, const char* name) {
    TORCH_CHECK(!half_to_float, "torch_vortex: ", name, " half_to_float is "
                "unsupported; the input must already be float32 (W3.2)");
}

static torch::Tensor softmax_impl(const torch::Tensor& self, int64_t dim,
                                  bool half_to_float) {
    check_half_to_float(half_to_float, "softmax");
    return softmax_op(self, dim, &vx_prim_softmax, "softmax");
}

static torch::Tensor log_softmax_impl(const torch::Tensor& self, int64_t dim,
                                      bool half_to_float) {
    check_half_to_float(half_to_float, "log_softmax");
    return softmax_op(self, dim, &vx_prim_log_softmax, "log_softmax");
}

static torch::Tensor logsumexp_impl(const torch::Tensor& self,
                                    c10::IntArrayRef dim, bool keepdim) {
    check_vortex_f32(self, "logsumexp");
    c10::OptionalArrayRef<int64_t> d =
        dim.size() == 0 ? std::nullopt : std::make_optional(dim);
    check_reduce_dims(self, d, "logsumexp");
    auto out = torch::empty(reduced_shape(self, d, keepdim), self.options());

    // A reduction that consumes no elements has an answer here, and it is not
    // the kernel's: sum writes its identity and mean writes NaN, and this
    // writes -inf -- the log of an empty sum -- which is what torch returns.
    // Checked before the layout, which refuses a zero row length outright.
    if (self.numel() == 0) {
        if (out.numel() > 0) {
            fill_args_t f = {(uint64_t)(uintptr_t)out.data_ptr(),
                             u32_numel(out, "logsumexp"), (float)-INFINITY, 0};
            launch(h_fill_kernel, f, (uint32_t)((out.numel() + 3) / 4));
        } else {
            ++g_stats.skipped_launches;
        }
        return out;
    }

    auto src = reduce_layout(self, d, "logsumexp");
    TORCH_CHECK(out.numel() == src.size(0), "torch_vortex: logsumexp out has ",
                out.numel(), " elements but the reduction produces ",
                src.size(0));
    DL_LAUNCH(vx_prim_logsumexp(current_queue(),
                                (uint64_t)(uintptr_t)src.data_ptr(),
                                (uint64_t)(uintptr_t)out.data_ptr(),
                                u32_dim(src.size(0), "logsumexp rows"),
                                u32_dim(src.size(1), "logsumexp cols")));
    return out;
}

// ---- elementwise ----------------------------------------------------------
//
// Op-code driven: one kernel per arity, with the operation selected by a field
// in kernels/torch_kernel_args.h. Elementwise kernels keep the simple
// grid-stride shape VOLT compiles correctly -- the note on tv_mm_kernel
// records what happens when one stops being simple.

// PyTorch hands a Python number to the *tensor* overloads: `x * 3.0` arrives
// at mul.Tensor with a 0-dim CPU tensor, not at mul.Scalar. Treating that as a
// scalar is what the CUDA backend does; before this, `x * 3.0` failed with
// "mul.b is on cpu rather than the vortex device".
static bool is_host_scalar(const torch::Tensor& t) {
    return t.device().type() == c10::DeviceType::CPU && t.dim() == 0;
}

// Right-aligned broadcasting, the rule ATen uses. Returns false when the
// shapes do not broadcast; the caller has already produced a better message
// via at::infer_size, so this only has to agree with it. A broadcast
// dimension is a stride of 0, which the kernel reads as "same element".
static bool broadcast_layout(const torch::Tensor& a, const torch::Tensor& b,
                             uint32_t sizes[4], uint32_t a_strides[4],
                             uint32_t b_strides[4], uint32_t& ndim) {
    const int64_t na = a.dim(), nb = b.dim();
    const int64_t nd = std::max(na, nb);
    if (nd > 4) {
        return false;
    }
    ndim = (uint32_t)nd;
    for (int64_t d = 0; d < nd; ++d) {
        const int64_t ia = na - 1 - d, ib = nb - 1 - d;
        const int64_t sa = ia >= 0 ? a.size(ia) : 1;
        const int64_t sb = ib >= 0 ? b.size(ib) : 1;
        if (sa != sb && sa != 1 && sb != 1) {
            return false;
        }
        const int64_t size = std::max(sa, sb);
        const int64_t at = nd - 1 - d;
        sizes[at] = u32_dim(size, "broadcast size");
        a_strides[at] = (sa == 1 && size != 1) ? 0
                        : u32_dim(ia >= 0 ? a.stride(ia) : 0, "broadcast stride");
        b_strides[at] = (sb == 1 && size != 1) ? 0
                        : u32_dim(ib >= 0 ? b.stride(ib) : 0, "broadcast stride");
    }
    return true;
}

static void launch_broadcast_op(const torch::Tensor& a, const torch::Tensor& b,
                                torch::Tensor& out, uint32_t op) {
    uint32_t sizes[4] = {};
    uint32_t a_strides[4] = {};
    uint32_t b_strides[4] = {};
    uint32_t ndim = 0;
    const uint32_t total = u32_numel(out, "elementwise output");
    TORCH_CHECK(broadcast_layout(a, b, sizes, a_strides, b_strides, ndim),
                "torch_vortex: shapes ", a.sizes(), " and ", b.sizes(),
                " do not broadcast, or exceed 4 dimensions");
    DL_LAUNCH(vx_prim_broadcast(current_queue(), (vx_prim_binary_op)op,
                                (uint64_t)(uintptr_t)out.data_ptr(),
                                (uint64_t)(uintptr_t)a.data_ptr(),
                                (uint64_t)(uintptr_t)b.data_ptr(), total, ndim,
                                sizes, a_strides, b_strides));
}

static void launch_elementwise(const torch::Tensor& a, const torch::Tensor& b,
                               torch::Tensor& out, uint32_t op) {
    const uint64_t dst = (uint64_t)(uintptr_t)out.data_ptr();
    const uint32_t n = u32_numel(out, "elementwise output");
    if (is_host_scalar(b)) {
        launch_scalar_op(dst, (uint64_t)(uintptr_t)a.data_ptr(),
                         b.item<float>(), n, op, 0);
    } else if (is_host_scalar(a)) {
        launch_scalar_op(dst, (uint64_t)(uintptr_t)b.data_ptr(),
                         a.item<float>(), n, op, 1);
    } else if (a.sizes() == b.sizes()) {
        // the common case, and the kernel shape that is known-good
        launch_binary_op(dst, (uint64_t)(uintptr_t)a.data_ptr(),
                         (uint64_t)(uintptr_t)b.data_ptr(), n, op);
    } else {
        launch_broadcast_op(a, b, out, op);
    }
}

static void check_elementwise(const torch::Tensor& a, const torch::Tensor& b,
                              const char* name) {
    const bool a_scalar = is_host_scalar(a);
    const bool b_scalar = is_host_scalar(b);
    TORCH_CHECK(!(a_scalar && b_scalar), "torch_vortex: ", name,
                " with two host scalars should have been folded by torch");
    check_vortex_f32(a_scalar ? b : a, name);
    if (!a_scalar && !b_scalar) {
        // at::infer_size is ATen's own broadcasting rule, so a shape it
        // rejects is rejected with ATen's message rather than ours.
        at::infer_size(a.sizes(), b.sizes());
        check_vortex_f32(b, name);
    }
}

static torch::Tensor binary_op(const torch::Tensor& a, const torch::Tensor& b,
                               uint32_t op, const char* name) {
    check_elementwise(a, b, name);
    const bool a_scalar = is_host_scalar(a), b_scalar = is_host_scalar(b);
    auto out = (a_scalar || b_scalar)
                   ? torch::empty_like(a_scalar ? b : a)
                   : torch::empty(at::infer_size(a.sizes(), b.sizes()),
                                  a.options());
    launch_elementwise(a, b, out, op);
    return out;
}

static torch::Tensor& binary_op_(torch::Tensor& self, const torch::Tensor& other,
                                 uint32_t op, const char* name) {
    check_vortex_f32(self, name);
    if (!is_host_scalar(other)) {
        check_vortex_f32(other, name);
        // broadcasting is allowed, but only if it lands exactly on self --
        // growing in place is what the out-of-place form is for
        TORCH_CHECK(at::infer_size(self.sizes(), other.sizes()) == self.sizes(),
                    "torch_vortex: ", name, " would need to grow ", self.sizes(),
                    " to ", at::infer_size(self.sizes(), other.sizes()),
                    ", which in place cannot do");
    }
    launch_elementwise(self, other, self, op);
    return self;
}

// Scale a tensor by a host constant, one kernel, used only for alpha != 1.
static torch::Tensor scaled_by(const torch::Tensor& t, double k) {
    auto out = torch::empty_like(t);
    launch_scalar_op((uint64_t)(uintptr_t)out.data_ptr(),
                     (uint64_t)(uintptr_t)t.data_ptr(), (float)k,
                     u32_numel(out, "scaled"), VX_PRIM_BINARY_MUL, 0);
    return out;
}

// torch.add/sub take an `alpha` scaling the second operand, so these compute
// `a <op> b*alpha`. Implemented as a scale followed by the plain op rather
// than by growing the argument block: two launches on a path that already
// costs one is cheaper than a wider ABI every kernel has to agree on.
//
// `op` is threaded through every branch. Hardcoding ADD in the tensor-tensor
// branch (as this first did) makes torch.sub(a, b, alpha) silently add.
static void check_alpha(const c10::Scalar& alpha) {
    TORCH_CHECK(std::isfinite(alpha.to<double>()),
                "torch_vortex: alpha must be finite");
}

static torch::Tensor alpha_scaled_op(const torch::Tensor& a, const torch::Tensor& b,
                              uint32_t op, const c10::Scalar& alpha,
                              const char* name) {
    check_alpha(alpha);
    const double k = alpha.to<double>();
    if (k == 1.0) {
        return binary_op(a, b, op, name);
    }
    check_elementwise(a, b, name);
    const torch::Tensor scaled = is_host_scalar(b)
                                     ? b
                                     : scaled_by(b, k);
    const float value = is_host_scalar(b)
                            ? (float)(b.item<double>() * k)
                            : (float)k;
    auto out = torch::empty_like(is_host_scalar(a) ? scaled : a);
    if (is_host_scalar(b)) {
        launch_scalar_op((uint64_t)(uintptr_t)out.data_ptr(),
                         (uint64_t)(uintptr_t)a.data_ptr(), value,
                         u32_numel(out, name), op, 0);
    } else if (is_host_scalar(a)) {
        // reverse: the scalar is the left operand, so `a - b*alpha` is not
        // `b*alpha - a`
        launch_scalar_op((uint64_t)(uintptr_t)out.data_ptr(),
                         (uint64_t)(uintptr_t)scaled.data_ptr(),
                         a.item<float>(), u32_numel(out, name), op, 1);
    } else {
        launch_binary_op((uint64_t)(uintptr_t)out.data_ptr(),
                         (uint64_t)(uintptr_t)a.data_ptr(),
                         (uint64_t)(uintptr_t)scaled.data_ptr(),
                         u32_numel(out, name), op);
    }
    return out;
}

static torch::Tensor& alpha_scaled_op_(torch::Tensor& self, const torch::Tensor& other,
                                uint32_t op, const c10::Scalar& alpha,
                                const char* name) {
    check_alpha(alpha);
    const double k = alpha.to<double>();
    if (k == 1.0) {
        return binary_op_(self, other, op, name);
    }
    if (is_host_scalar(other)) {
        check_vortex_f32(self, name);
        launch_scalar_op((uint64_t)(uintptr_t)self.data_ptr(),
                         (uint64_t)(uintptr_t)self.data_ptr(),
                         (float)(other.item<double>() * k),
                         u32_numel(self, name), op, 0);
        return self;
    }
    auto scaled = scaled_by(other, k);
    return binary_op_(self, scaled, op, name);
}

#define VX_BINARY_OP(NAME, OP, LABEL)                                          \
    static torch::Tensor NAME##_impl(const torch::Tensor& a,                   \
                                     const torch::Tensor& b) {                 \
        return binary_op(a, b, OP, LABEL);                                     \
    }                                                                          \
    static torch::Tensor& NAME##__impl(torch::Tensor& self,                    \
                                       const torch::Tensor& other) {           \
        return binary_op_(self, other, OP, LABEL);                             \
    }

VX_BINARY_OP(mul, VX_PRIM_BINARY_MUL, "mul")
VX_BINARY_OP(div, VX_PRIM_BINARY_DIV, "div")
VX_BINARY_OP(maximum, VX_PRIM_BINARY_MAXIMUM, "maximum")
VX_BINARY_OP(minimum, VX_PRIM_BINARY_MINIMUM, "minimum")
#undef VX_BINARY_OP

static torch::Tensor add_impl(const torch::Tensor& a, const torch::Tensor& b,
                              const c10::Scalar& alpha) {
    return alpha_scaled_op(a, b, VX_PRIM_BINARY_ADD, alpha, "add");
}
static torch::Tensor sub_impl(const torch::Tensor& a, const torch::Tensor& b,
                              const c10::Scalar& alpha) {
    // torch.sub(a, b, alpha) is a - alpha*b
    return alpha_scaled_op(a, b, VX_PRIM_BINARY_SUB, alpha, "sub");
}
static torch::Tensor& add__impl(torch::Tensor& self, const torch::Tensor& other,
                                const c10::Scalar& alpha) {
    return alpha_scaled_op_(self, other, VX_PRIM_BINARY_ADD, alpha, "add_");
}
static torch::Tensor& sub__impl(torch::Tensor& self, const torch::Tensor& other,
                                const c10::Scalar& alpha) {
    return alpha_scaled_op_(self, other, VX_PRIM_BINARY_SUB, alpha, "sub_");
}

// The .Scalar overloads. `x + 1.0` reaches add.Tensor with a wrapped 0-dim
// tensor (handled above), but torch.add(x, 1.0, alpha=2) is genuinely
// add.Scalar, and models use it.
#define VX_SCALAR_OP(SCHEMA, NAME, OP, LABEL)                                   \
    static torch::Tensor NAME##_scalar_impl(const torch::Tensor& self,          \
                                            const c10::Scalar& other) {         \
        return binary_op(self, at::scalar_to_tensor(other), OP, LABEL);         \
    }                                                                           \
    static torch::Tensor& NAME##_scalar__impl(torch::Tensor& self,               \
                                              const c10::Scalar& other) {        \
        return binary_op_(self, at::scalar_to_tensor(other), OP, LABEL);        \
    }

VX_SCALAR_OP("mul.Scalar", mul, VX_PRIM_BINARY_MUL, "mul")
VX_SCALAR_OP("div.Scalar", div, VX_PRIM_BINARY_DIV, "div")
#undef VX_SCALAR_OP

static torch::Tensor add_scalar_impl(const torch::Tensor& self,
                                     const c10::Scalar& other,
                                     const c10::Scalar& alpha) {
    return alpha_scaled_op(self, at::scalar_to_tensor(other), VX_PRIM_BINARY_ADD, alpha,
                    "add");
}
static torch::Tensor sub_scalar_impl(const torch::Tensor& self,
                                     const c10::Scalar& other,
                                     const c10::Scalar& alpha) {
    return alpha_scaled_op(self, at::scalar_to_tensor(other), VX_PRIM_BINARY_SUB, alpha,
                    "sub");
}
static torch::Tensor& add_scalar__impl(torch::Tensor& self,
                                       const c10::Scalar& other,
                                       const c10::Scalar& alpha) {
    return alpha_scaled_op_(self, at::scalar_to_tensor(other), VX_PRIM_BINARY_ADD, alpha,
                     "add_");
}
static torch::Tensor& sub_scalar__impl(torch::Tensor& self,
                                       const c10::Scalar& other,
                                       const c10::Scalar& alpha) {
    return alpha_scaled_op_(self, at::scalar_to_tensor(other), VX_PRIM_BINARY_SUB, alpha,
                     "sub_");
}

// ---- unary ----------------------------------------------------------------

static torch::Tensor unary_op(const torch::Tensor& self, uint32_t op,
                              const char* name) {
    check_vortex_f32(self, name);
    auto out = torch::empty_like(self);
    launch_unary_op((uint64_t)(uintptr_t)out.data_ptr(),
                    (uint64_t)(uintptr_t)self.data_ptr(),
                    u32_numel(out, name), op);
    return out;
}

// In place: source and destination are the same buffer, which is safe here
// because each thread reads and writes the same index.
static torch::Tensor& unary_op_(torch::Tensor& self, uint32_t op,
                                const char* name) {
    check_vortex_f32(self, name);
    launch_unary_op((uint64_t)(uintptr_t)self.data_ptr(),
                    (uint64_t)(uintptr_t)self.data_ptr(),
                    u32_numel(self, name), op);
    return self;
}

// gelu takes a second parameter, so it does not fit the one-argument macro.
// the default is the erf form; approximate="tanh" is a different function,
// not a different spelling of the same one.
static uint32_t gelu_op(vx_schema_string_view approximate, const char* name) {
    if (approximate == "none") {
        return VX_PRIM_OP_GELU_ERF;
    }
    if (approximate == "tanh") {
        return VX_PRIM_OP_GELU_TANH;
    }
    TORCH_CHECK(false, "torch_vortex: ", name, " approximate='", approximate,
                "' is unsupported; use 'none' or 'tanh'");
}

static torch::Tensor gelu_impl(const torch::Tensor& self,
                               vx_schema_string_view approximate) {
    return unary_op(self, gelu_op(approximate, "gelu"), "gelu");
}
static torch::Tensor& gelu__impl(torch::Tensor& self,
                                 vx_schema_string_view approximate) {
    return unary_op_(self, gelu_op(approximate, "gelu_"), "gelu_");
}

static torch::Tensor relu_impl(const torch::Tensor& self) {
    return unary_op(self, VX_PRIM_OP_RELU, "relu");
}
static torch::Tensor& relu__impl(torch::Tensor& self) {
    return unary_op_(self, VX_PRIM_OP_RELU, "relu_");
}

#define VX_UNARY_OP(NAME, OP, LABEL)                                           \
    static torch::Tensor NAME##_impl(const torch::Tensor& self) {              \
        return unary_op(self, OP, LABEL);                                      \
    }                                                                          \
    static torch::Tensor& NAME##__impl(torch::Tensor& self) {                  \
        return unary_op_(self, OP, LABEL);                                     \
    }

VX_UNARY_OP(neg, VX_PRIM_OP_NEG, "neg")
VX_UNARY_OP(abs, VX_PRIM_OP_ABS, "abs")
VX_UNARY_OP(exp, VX_PRIM_OP_EXP, "exp")
VX_UNARY_OP(log, VX_PRIM_OP_LOG, "log")
VX_UNARY_OP(sqrt, VX_PRIM_OP_SQRT, "sqrt")
VX_UNARY_OP(rsqrt, VX_PRIM_OP_RSQRT, "rsqrt")
VX_UNARY_OP(sigmoid, VX_PRIM_OP_SIGMOID, "sigmoid")
VX_UNARY_OP(tanh, VX_PRIM_OP_TANH, "tanh")
VX_UNARY_OP(reciprocal, VX_PRIM_OP_RECIPROCAL, "reciprocal")
VX_UNARY_OP(silu, VX_PRIM_OP_SILU, "silu")
#undef VX_UNARY_OP

// ---------------------------------------------------------------------------
// dnn ops (plan P5-02): the ResNet op set on device. All constraints are
// TORCH_CHECKed — no silent CPU fallback.
// ---------------------------------------------------------------------------

static void check_cnn_f32(const torch::Tensor& t, const char* what,
                          int64_t dims) {
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1,
                "torch_vortex: ", what, " is on ", t.device(), " rather than ",
                "the vortex device");
    TORCH_CHECK(t.scalar_type() == at::kFloat,
                "torch_vortex: ", what, " must be float32 in v1");
    TORCH_CHECK(t.is_contiguous(), "torch_vortex: ", what, " must be contiguous");
    TORCH_CHECK(t.dim() == dims, "torch_vortex: ", what, " must be ", dims, "-D");
}

// Output extent of a convolution/pooling window. Named after the argument so
// the error says which one is wrong.
//
// The sign is the point: this used to be uint32_t arithmetic, so an input
// smaller than the kernel wrapped around and produced a huge output size and
// grid instead of an error. PyTorch floors, so a window that does not divide
// evenly is legal and must not be rejected here.
static int64_t window_out(int64_t in, int64_t k, int64_t pad, int64_t stride,
                          bool ceil_mode, const char* what) {
    TORCH_CHECK(k > 0, "torch_vortex: ", what, ": kernel must be > 0, got ", k);
    TORCH_CHECK(stride > 0,
                "torch_vortex: ", what, ": stride must be > 0, got ", stride);
    TORCH_CHECK(pad >= 0,
                "torch_vortex: ", what, ": padding must be >= 0, got ", pad);
    TORCH_CHECK(in + 2 * pad >= k, "torch_vortex: ", what, ": input ", in,
                " with padding ", pad, " is smaller than kernel ", k);
    int64_t out = ceil_mode ? (in + 2 * pad - k + stride - 1) / stride + 1
                            : (in + 2 * pad - k) / stride + 1;
    if (ceil_mode) {
        while (out > 0 && (out - 1) * stride >= in + pad) --out;
    }
    return out;
}


static torch::Tensor convolution_impl(
    const torch::Tensor& input, const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias, c10::IntArrayRef stride,
                                       c10::IntArrayRef padding,
                                       c10::IntArrayRef dilation,
                                       bool transposed,
                                       c10::IntArrayRef output_padding,
                                       int64_t groups) {
    check_cnn_f32(input, "conv.input", 4);
    check_cnn_f32(weight, "conv.weight", 4);
    TORCH_CHECK(!transposed, "torch_vortex: transposed conv unsupported in v1");
    TORCH_CHECK(groups > 0, "torch_vortex: conv groups must be positive");
    TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
                "torch_vortex: conv dilation must have one or two values");
    const int64_t dh = dilation.size() > 0 ? dilation[0] : 1;
    const int64_t dw = dilation.size() > 1 ? dilation[1] : dh;
    TORCH_CHECK(dh > 0 && dw > 0, "torch_vortex: conv dilation must be positive");
    for (auto o : output_padding) TORCH_CHECK(o == 0, "torch_vortex: output_padding must be 0");
    const int64_t sh = stride.size() > 0 ? stride[0] : 1;
    const int64_t sw = stride.size() > 1 ? stride[1] : sh;
    const int64_t ph = padding.size() > 0 ? padding[0] : 0;
    const int64_t pw = padding.size() > 1 ? padding[1] : ph;

    const auto& is = input.sizes();
    const auto& ws = weight.sizes();
    const int64_t co = ws[0], ci_group = ws[1];
    const int64_t kh = ws[2], kw = ws[3];
    TORCH_CHECK(is[1] % groups == 0 && ci_group == is[1] / groups,
                "torch_vortex: conv channel mismatch for groups=", groups);
    TORCH_CHECK((kh - 1) <= (INT64_MAX - 1) / dh &&
                    (kw - 1) <= (INT64_MAX - 1) / dw,
                "torch_vortex: conv effective kernel overflows");
    const int64_t ekh = (kh - 1) * dh + 1;
    const int64_t ekw = (kw - 1) * dw + 1;
    const int64_t ho = window_out(is[2], ekh, ph, sh, false, "conv height");
    const int64_t wo = window_out(is[3], ekw, pw, sw, false, "conv width");

    // One output channel stages all of its weights in LMEM at once, and the DL
    // kernel's ceiling for that is 16384 bytes. This is the real bound on a
    // full-resolution ResNet stem (ci=512, 3x3, fp32 needs 18 KiB), and
    // shrinking the input image does not reduce it. Checked before anything is
    // allocated, so a rejected call does nothing at all.
    const int64_t lmem_needed = ci_group * kh * kw * 4;
    TORCH_CHECK(lmem_needed <= 16384, "torch_vortex: conv needs ", lmem_needed,
                " bytes of local memory to stage one filter (ci=", ci_group, " kh=",
                kh, " kw=", kw, " x 4 bytes), but the DL kernel allows 16384. ",
                "Tile the weights across output channels; a smaller input ",
                "image does not reduce this.");

    auto out = torch::empty({is[0], co, ho, wo}, input.options());
    uint64_t baddr = 0;
    // Composite layers pass None as an *undefined* tensor inside the
    // optional (still has_value) — treat undefined as absent.
    if (bias.has_value() && bias->defined()) {
        check_cnn_f32(*bias, "conv.bias", 1);
        TORCH_CHECK(bias->numel() == co, "torch_vortex: bias size mismatch");
        baddr = (uint64_t)(uintptr_t)bias->data_ptr();
    }
    // An all-empty result is not a launch: the DL kernel refuses n == 0 (and a
    // zero-sized dimension), and there is nothing to compute anyway. The
    // output is still allocated with the right shape, so this matches what
    // PyTorch does rather than being a special case.
    if (out.numel() == 0) {
        return out;
    }
    // The kernel lives in the DL library, so the ATen path and a direct
    // vx_dnn_conv2d call are the *same* kernel rather than two copies that
    // have to be kept in agreement (W3.1).
    const int status = (int)vx_dnn_conv2d_dilated(
        current_queue(), (uint64_t)(uintptr_t)input.data_ptr(),
        (uint64_t)(uintptr_t)weight.data_ptr(), baddr,
        (uint64_t)(uintptr_t)out.data_ptr(), u32_dim(is[0], "conv batch"),
        u32_dim(is[1], "conv ci"), u32_dim(is[2], "conv hi"),
        u32_dim(is[3], "conv wi"), u32_dim(co, "conv co"),
        u32_dim(kh, "conv kh"), u32_dim(kw, "conv kw"), u32_dim(ph, "conv ph"),
        u32_dim(pw, "conv pw"), u32_dim(sh, "conv sh"), u32_dim(sw, "conv sw"),
        u32_dim(groups, "conv groups"), u32_dim(dh, "conv dh"),
        u32_dim(dw, "conv dw"));
    // Not DL_LAUNCH: this one explains what its status codes mean, which the
    // macro has no room for. The accounting below is the same call it makes.
    TORCH_CHECK(status == 0, "torch_vortex: vx_dnn_conv2d failed with status ",
                status, " (2 is bad args, 3 is a shape the DL kernel rejects)");
    note_device_work();
    return out;
}

static torch::Tensor pool_impl(const torch::Tensor& self,
                               c10::IntArrayRef kernel_size,
                               c10::IntArrayRef stride,
                               c10::IntArrayRef padding, uint32_t op,
                               bool ceil_mode, uint32_t divisor) {
    check_cnn_f32(self, "pool.input", 4);
    const int64_t kh = kernel_size[0], kw = kernel_size[1];
    const int64_t sh = stride.size() > 0 ? stride[0] : kh;
    const int64_t sw = stride.size() > 1 ? stride[1] : sh;
    const int64_t ph = padding.size() > 0 ? padding[0] : 0;
    const int64_t pw = padding.size() > 1 ? padding[1] : ph;
    const auto& s = self.sizes();
    const int64_t ho = window_out(s[2], kh, ph, sh, ceil_mode, "pool height");
    const int64_t wo = window_out(s[3], kw, pw, sw, ceil_mode, "pool width");
    auto out = torch::empty({s[0], s[1], ho, wo}, self.options());
    // An all-empty result is not a launch: the DL kernel refuses n == 0 or a
    // zero-sized dimension, and there is nothing to compute anyway.
    if (out.numel() == 0) {
        return out;
    }
    // The shape arithmetic stays here: window_out is bounds-checked and names
    // the argument it rejects, while the DL's is plain unsigned arithmetic.
    // ATen validates, the DL computes.
    DL_LAUNCH(vx_dnn_pool2d_ex(current_queue(),
                            (uint64_t)(uintptr_t)self.data_ptr(),
                            (uint64_t)(uintptr_t)out.data_ptr(),
                            u32_dim(s[0], "pool batch"), u32_dim(s[1], "pool channels"),
                            u32_dim(s[2], "pool input height"),
                            u32_dim(s[3], "pool input width"),
                            u32_dim(kh, "pool kernel height"),
                            u32_dim(kw, "pool kernel width"),
                            u32_dim(ph, "pool padding height"),
                            u32_dim(pw, "pool padding width"),
                            u32_dim(sh, "pool stride height"),
                            u32_dim(sw, "pool stride width"), op, divisor));
    return out;
}

static torch::Tensor max_pool2d_impl(const torch::Tensor& self,
                                     c10::IntArrayRef kernel,
                                     c10::IntArrayRef stride,
                                     c10::IntArrayRef padding,
                                     c10::IntArrayRef dilation,
                                     bool ceil_mode) {
    for (auto d : dilation) TORCH_CHECK(d == 1, "torch_vortex: dilation must be 1");
    TORCH_CHECK(!ceil_mode, "torch_vortex: ceil_mode unsupported in v1");
    return pool_impl(self, kernel, stride, padding, 0, false, 0);
}

static torch::Tensor avg_pool2d_impl(const torch::Tensor& self,
                                     c10::IntArrayRef kernel,
                                     c10::IntArrayRef stride,
                                     c10::IntArrayRef padding,
                                     bool ceil_mode, bool count_include_pad,
                                     std::optional<int64_t> divisor_override) {
    TORCH_CHECK(!ceil_mode, "torch_vortex: ceil_mode unsupported in v1");
    uint32_t divisor = 0;
    if (divisor_override.has_value()) {
        TORCH_CHECK(*divisor_override > 0 && *divisor_override <= UINT32_MAX,
                    "torch_vortex: avg_pool2d divisor_override must fit uint32");
        divisor = (uint32_t)*divisor_override;
    }
    return pool_impl(self, kernel, stride, padding,
                     count_include_pad ? 2u : 1u, false, divisor);
}


static torch::Tensor adaptive_avg_pool2d_impl(const torch::Tensor& self,
                                              c10::SymIntArrayRef output_size) {
    check_cnn_f32(self, "pool.input", 4);
    auto os = sym_to_vec(output_size);
    TORCH_CHECK(os.size() == 2 && os[0] == 1 && os[1] == 1,
                "torch_vortex: adaptive_avg_pool2d only output (1,1) in v1");
    const auto& s = self.sizes();
    return pool_impl(self, {(int64_t)s[2], (int64_t)s[3]}, {}, {}, 1, false, 0);
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> native_batch_norm_impl(
    const torch::Tensor& input, const std::optional<torch::Tensor>& weight,
    const std::optional<torch::Tensor>& bias,
    const std::optional<torch::Tensor>& running_mean,
    const std::optional<torch::Tensor>& running_var, bool training,
    double momentum, double eps) {
    TORCH_CHECK(!training, "torch_vortex: batch_norm training is unsupported "
                "(inference only); training is W8.1 in docs/mydocs/pytorch_plan.md");
    check_cnn_f32(input, "bn.input", 4);
    auto is_def = [](const std::optional<torch::Tensor>& t) {
        return t.has_value() && t->defined();
    };
    TORCH_CHECK(is_def(running_mean) && is_def(running_var),
                "torch_vortex: bn in inference mode needs running_mean and "
                "running_var");
    const auto& s = input.sizes();
    const int64_t c = s[1];
    const int64_t hw = s[2] * s[3];
    const int64_t total = input.numel();
    TORCH_CHECK(total == s[0] * c * hw, "torch_vortex: bn input is not NCHW");

    // Optional affine: both must be present or both absent.
    const bool has_affine = is_def(weight) || is_def(bias);
    TORCH_CHECK(is_def(weight) == is_def(bias),
                "torch_vortex: bn needs weight and bias together or not at all");
    auto check_param = [&](const std::optional<torch::Tensor>& t,
                           const char* what) {
        TORCH_CHECK(t->device().type() == c10::DeviceType::PrivateUse1,
                    "torch_vortex: bn ", what, " is on ", t->device(),
                    " rather than the vortex device");
        TORCH_CHECK(t->scalar_type() == at::kFloat,
                    "torch_vortex: bn ", what, " must be float32");
        TORCH_CHECK(t->is_contiguous(), "torch_vortex: bn ", what,
                    " must be contiguous");
        TORCH_CHECK(t->numel() == c, "torch_vortex: bn ", what, " has ",
                    t->numel(), " elements but the input has ", c, " channels");
    };
    check_param(running_mean, "running_mean");
    check_param(running_var, "running_var");
    if (has_affine) {
        check_param(weight, "weight");
        check_param(bias, "bias");
    }

    if (total == 0) {
        // nothing to compute, and the DL kernel rejects n == 0
        auto empty_aux = torch::empty({0}, input.options());
        return std::make_tuple(torch::empty_like(input), empty_aux, empty_aux);
    }
    auto out = torch::empty_like(input);
    // The DL kernel computes rstd from var + eps itself, so there is no host
    // round-trip here; and it takes weight/bias as optional, which is why the
    // ATen-side optional affine maps onto it without dummy buffers.
    DL_LAUNCH(vx_dnn_bn_affine(
        current_queue(), (uint64_t)(uintptr_t)input.data_ptr(),
        (uint64_t)(uintptr_t)running_mean->data_ptr(),
        (uint64_t)(uintptr_t)running_var->data_ptr(),
        has_affine ? (uint64_t)(uintptr_t)weight->data_ptr() : 0,
        has_affine ? (uint64_t)(uintptr_t)bias->data_ptr() : 0,
        (uint64_t)(uintptr_t)out.data_ptr(), u32_dim(s[0], "bn batch"),
        u32_dim(c, "bn channels"), u32_dim(hw, "bn spatial span"), (float)eps));
    // rstd is computed inside the kernel from running_var, so this path makes
    // no host round-trip and allocates no scratch buffer. The previous version
    // copied running_var to the host, took a sqrt, copied it back, and freed
    // the scratch immediately after an asynchronous launch.
    auto empty_aux = torch::empty({0}, input.options());
    return std::make_tuple(out, empty_aux, empty_aux);
}

// ---- layer norm / rms norm ------------------------------------------------
//
// Both normalise over the trailing dims, which is exactly the (rows, cols)
// view the DL library's row kernels take, so `cols` is the product of the
// normalized dims and `rows` everything in front of them. ATen validates the
// shape and the DL kernel computes, the same division of labour as conv.

// The normalized dims must be the input's trailing ones and the affine, if
// present, must be cols-long. Torch accepts nothing else, and its message for
// a mis-shaped normalized_shape is worth reproducing in substance: it names
// both shapes.
static void check_norm_shape(const torch::Tensor& input,
                             c10::SymIntArrayRef normalized_shape,
                             const char* name, std::vector<int64_t>& ns,
                             int64_t& rows, int64_t& cols) {
    ns = sym_to_vec(normalized_shape);
    const int64_t nd = input.dim();
    const int64_t k = (int64_t)ns.size();
    // Not zero: torch refuses it ("Expected normalized_shape to be at least
    // 1-dimensional"), and accepting it would make every element its own row,
    // which computes a result rather than reporting the mistake.
    TORCH_CHECK(k > 0, "torch_vortex: ", name, " needs a normalized_shape of "
                "at least one dimension");
    TORCH_CHECK(k <= nd, "torch_vortex: ", name, " normalized_shape has ", k,
                " dimensions but the input has ", nd);
    for (int64_t i = 0; i < k; ++i) {
        TORCH_CHECK(input.size(nd - k + i) == ns[i], "torch_vortex: ", name,
                    " normalized_shape=", ns, " must match the input's trailing "
                    "dimensions; got input of size ", input.sizes());
    }
    // Counted rather than divided: a zero-length normalized dimension makes
    // numel/cols meaningless, and the leading dims are the answer anyway.
    rows = 1;
    for (int64_t d = 0; d < nd - k; ++d) rows *= input.size(d);
    cols = 1;
    for (int64_t d = nd - k; d < nd; ++d) cols *= input.size(d);
}

// A norm's affine parameter: the normalized shape, on this device, in this
// dtype. The shape is checked rather than the element count -- a (2,2) weight
// against a normalized_shape of [4] has the right number of elements and is
// still a mistake, and torch refuses it by naming both shapes.
static uint64_t norm_affine_addr(const std::optional<torch::Tensor>& t,
                                 const std::vector<int64_t>& ns,
                                 int64_t cols, const char* what) {
    check_vortex_f32(*t, what);
    TORCH_CHECK(t->sizes().vec() == ns, "torch_vortex: ", what, " must have the "
                "normalized shape; torch expects ", ns, " but got ",
                t->sizes());
    TORCH_CHECK(t->numel() == cols, "torch_vortex: ", what, " has ",
                t->numel(), " elements but the normalized shape has ", cols);
    return (uint64_t)(uintptr_t)t->data_ptr();
}

static bool is_defined(const std::optional<torch::Tensor>& t) {
    return t.has_value() && t->defined();
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
native_layer_norm_impl(const torch::Tensor& input,
                       c10::SymIntArrayRef normalized_shape,
                       const std::optional<torch::Tensor>& weight,
                       const std::optional<torch::Tensor>& bias, double eps) {
    check_vortex_f32(input, "layer_norm.input");
    int64_t rows = 0, cols = 0;
    std::vector<int64_t> ns;
    check_norm_shape(input, normalized_shape, "layer_norm", ns, rows, cols);

    // Optional affine, each half independent: torch takes either alone and
    // treats the absent one as its identity (gamma 1, beta 0). F.layer_norm(x,
    // shape, weight=w) is how a weight without a bias arrives.
    //
    // Validated before anything is allocated, so a rejected call leaves
    // nothing behind rather than three device blocks.
    uint64_t gaddr = 0, baddr = 0;
    if (is_defined(weight)) {
        gaddr = norm_affine_addr(weight, ns, cols, "layer_norm.weight");
    }
    if (is_defined(bias)) {
        baddr = norm_affine_addr(bias, ns, cols, "layer_norm.bias");
    }

    // mean and rstd keep the input's rank with the normalized dims as 1s,
    // which is what torch returns: (2,3,4) with shape [4] gives (2,3,1).
    std::vector<int64_t> stat_shape(input.sizes().begin(), input.sizes().end());
    for (size_t i = stat_shape.size() - normalized_shape.size();
         i < stat_shape.size(); ++i) {
        stat_shape[i] = 1;
    }
    auto out = torch::empty_like(input);
    auto mean = torch::empty(stat_shape, input.options());
    auto rstd = torch::empty(stat_shape, input.options());

    // A batch of zero rows is not a launch: the statistics outputs are empty
    // too, so there is nothing to write and nothing to compute.
    if (rows == 0) {
        ++g_stats.skipped_launches;
        return std::make_tuple(out, mean, rstd);
    }
    if (cols == 0) {
        // A zero-length normalized dimension. Nothing to normalise, so the
        // output is empty -- but the statistics are not: torch answers with
        // mean 0 and rstd NaN here (measured), not with empties.
        fill_args_t m = {(uint64_t)(uintptr_t)mean.data_ptr(), (uint32_t)rows,
                         0.0f, 0};
        launch(h_fill_kernel, m, (uint32_t)((rows + 3) / 4));
        fill_args_t r = {(uint64_t)(uintptr_t)rstd.data_ptr(), (uint32_t)rows,
                         (float)NAN, 0};
        launch(h_fill_kernel, r, (uint32_t)((rows + 3) / 4));
        return std::make_tuple(out, mean, rstd);
    }

    DL_LAUNCH(vx_prim_layernorm(
        current_queue(), (uint64_t)(uintptr_t)input.data_ptr(), gaddr, baddr,
        (uint64_t)(uintptr_t)out.data_ptr(),
        (uint64_t)(uintptr_t)mean.data_ptr(),
        (uint64_t)(uintptr_t)rstd.data_ptr(), u32_dim(rows, "layer_norm rows"),
        u32_dim(cols, "layer_norm cols"), (float)eps));
    return std::make_tuple(out, mean, rstd);
}

static torch::Tensor rms_norm_impl(const torch::Tensor& input,
                                   c10::SymIntArrayRef normalized_shape,
                                   const std::optional<torch::Tensor>& weight,
                                   std::optional<double> eps) {
    check_vortex_f32(input, "rms_norm.input");
    int64_t rows = 0, cols = 0;
    std::vector<int64_t> ns;
    check_norm_shape(input, normalized_shape, "rms_norm", ns, rows, cols);

    uint64_t gaddr = 0;
    if (is_defined(weight)) {
        gaddr = norm_affine_addr(weight, ns, cols, "rms_norm.weight");
    }

    auto out = torch::empty_like(input);
    if (rows == 0 || cols == 0) {
        ++g_stats.skipped_launches;
        return out;
    }
    // eps=None is not eps=0: torch falls back to the dtype's machine epsilon,
    // which is 1.19e-07 for float32. Measured -- rms_norm with None is
    // bit-identical to eps=finfo.eps and differs from eps=0.
    const double e = eps.has_value() ? *eps
                                     : (double)std::numeric_limits<float>::epsilon();
    DL_LAUNCH(vx_prim_rmsnorm(
        current_queue(), (uint64_t)(uintptr_t)input.data_ptr(), gaddr,
        (uint64_t)(uintptr_t)out.data_ptr(), u32_dim(rows, "rms_norm rows"),
        u32_dim(cols, "rms_norm cols"), (float)e));
    return out;
}

// The single matmul launcher. mm, linear and addmm are thin wrappers with
// different epilogues; having one implementation is what stops them from
// disagreeing about which axis is contracted.
static torch::Tensor mm_launch(const torch::Tensor& a, const torch::Tensor& b_in,
                               uint32_t transb, const torch::Tensor* self,
                               uint32_t self_kind, float alpha, float beta) {
    check_cnn_f32(a, "matmul.mat1", 2);
    check_cnn_f32(b_in, "matmul.mat2", 2);
    const int64_t m = a.size(0), k = a.size(1);
    const int64_t k_other = b_in.size(transb ? 1 : 0);
    const int64_t n = b_in.size(transb ? 0 : 1);
    TORCH_CHECK(k == k_other, "torch_vortex: matmul contraction mismatch: mat1 ",
                "is (", m, ", ", k, ") and mat2 is (", b_in.size(0), ", ",
                b_in.size(1), "), so the inner dimensions differ");

    // C starts at beta*self (or zero) and the DL gemm's epilogue adds alpha*A@B
    // to it, so the whole of torch.addmm is one gemm plus this fill. beta == 0
    // must not read self at all: torch.addmm ignores it then, and 0 * Inf is
    // NaN. A (n,) self is broadcast here rather than in the kernel.
    torch::Tensor out = torch::zeros({m, n}, a.options());
    if (self_kind != 0 && beta != 0.0f) {
        torch::Tensor self_full = (self_kind == 1)
                                      ? self->expand({m, n}).contiguous()
                                      : *self;
        out.add_(self_full, beta);
    }

    // A zero-sized dimension is not a launch: the DL rejects it (the KMU
    // derives the CTA shape from the grid and a zero collapses it), and the
    // answer is the epilogue already sitting in `out`.
    if (m == 0 || n == 0 || k == 0) {
        return out;
    }
    DL_LAUNCH(vx_blas_gemm_ex(current_queue(), VX_BLAS_F32, u32_dim(m, "matmul m"),
                           u32_dim(n, "matmul n"), u32_dim(k, "matmul k"), alpha,
                           1.0f, (uint64_t)(uintptr_t)a.data_ptr(),
                           (uint64_t)(uintptr_t)b_in.data_ptr(),
                           (uint64_t)(uintptr_t)out.data_ptr(), transb));
    return out;
}

static torch::Tensor bmm_impl(const torch::Tensor& a,
                               const torch::Tensor& b) {
    check_cnn_f32(a, "bmm.mat1", 3);
    check_cnn_f32(b, "bmm.mat2", 3);
    const int64_t batch = a.size(0), m = a.size(1), k = a.size(2);
    TORCH_CHECK(b.size(0) == batch, "torch_vortex: bmm batch mismatch: ",
                batch, " vs ", b.size(0));
    TORCH_CHECK(b.size(1) == k, "torch_vortex: bmm contraction mismatch: ",
                k, " vs ", b.size(1));
    const int64_t n = b.size(2);
    const uint32_t um = u32_dim(m, "bmm m");
    const uint32_t un = u32_dim(n, "bmm n");
    const uint32_t uk = u32_dim(k, "bmm k");
    // Fill and copy kernels carry element counts in uint32 fields.
    TORCH_CHECK(m == 0 || n == 0 || batch <= UINT32_MAX / m / n,
                "torch_vortex: bmm output element count does not fit in uint32");
    auto out = torch::empty({batch, m, n}, a.options());
    if (batch == 0 || m == 0 || n == 0) {
        ++g_stats.skipped_launches;
        return out;
    }
    if (k == 0) {
        return out.zero_();
    }

    const uint64_t a_stride = (uint64_t)a.stride(0) * sizeof(float);
    const uint64_t b_stride = (uint64_t)b.stride(0) * sizeof(float);
    const uint64_t c_stride = (uint64_t)out.stride(0) * sizeof(float);
    // beta=0 suppresses reads of the uninitialized output in the DL kernel.
    DL_LAUNCH(vx_blas_batched_gemm(
        current_queue(), VX_BLAS_F32, u32_dim(batch, "bmm batch"), um, un, uk,
        1.0f, 0.0f, (uint64_t)(uintptr_t)a.data_ptr(),
        (uint64_t)(uintptr_t)b.data_ptr(), (uint64_t)(uintptr_t)out.data_ptr(),
        a_stride, b_stride, c_stride, 0));
    return out;
}

static torch::Tensor mm_impl_wrap(const torch::Tensor& a,
                                  const torch::Tensor& b) {
    return mm_launch(a, b, 0, nullptr, 0, 1.0f, 0.0f);
}

static torch::Tensor linear_impl(
    const torch::Tensor& input, const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias) {
    check_vortex_f32(input, "linear.input");
    TORCH_CHECK(input.dim() > 0,
                "torch_vortex: linear input must have at least one dimension");
    check_cnn_f32(weight, "linear.weight", 2);
    const int64_t k = input.size(-1);
    TORCH_CHECK(k == weight.size(1),
                "torch_vortex: linear input has ", k,
                " features but weight expects ", weight.size(1));
    int64_t rows = 1;
    for (int64_t i = 0; i + 1 < input.dim(); ++i) rows *= input.size(i);
    auto matrix = input.reshape({rows, k});
    auto out = mm_launch(matrix, weight, 1, nullptr, 0, 1.0f, 0.0f);
    if (bias.has_value() && bias->defined()) {
        check_cnn_f32(*bias, "linear.bias", 1);
        TORCH_CHECK(bias->numel() == out.size(1), "torch_vortex: linear bias has ",
                    bias->numel(), " elements but the output has ", out.size(1),
                    " columns");
        bias_args_t args = {(uint64_t)(uintptr_t)out.data_ptr(),
                            (uint64_t)(uintptr_t)bias->data_ptr(),
                            u32_dim(out.size(0), "linear rows"),
                            u32_dim(out.size(1), "linear columns")};
        launch(h_tv_bias_add_kernel, args, (uint32_t)((out.numel() + 3) / 4));
    }
    auto shape = input.sizes().vec();
    shape.back() = weight.size(0);
    return out.reshape(shape);
}

// torch.addmm: beta*self + alpha*(mat1 @ mat2).
//
// This used to delegate to linear_impl, which computes mat1 @ mat2^T: a
// non-square system was rejected for the wrong reason and a square one
// silently produced mat1 @ mat2^T + self.
static torch::Tensor addmm_impl(const torch::Tensor& self,
                                const torch::Tensor& mat1,
                                const torch::Tensor& mat2,
                                const c10::Scalar& beta,
                                const c10::Scalar& alpha) {
    check_cnn_f32(mat1, "addmm.mat1", 2);
    check_cnn_f32(mat2, "addmm.mat2", 2);
    check_vortex_f32(self, "addmm.self");
    const int64_t m = mat1.size(0), k = mat1.size(1), n = mat2.size(1);
    TORCH_CHECK(mat2.size(0) == k, "torch_vortex: addmm contraction mismatch: ",
                "mat1 is (", m, ", ", k, ") and mat2 is (", mat2.size(0), ", ",
                n, ")");

    // self may be (m, n) or an (n,) row broadcast. A (m, 1) column would
    // broadcast in some other paths but not here: refuse it by name.
    uint32_t self_kind = 0;
    if (!(beta.to<double>() == 0.0 && self.numel() != 0)) {
        if (self.dim() == 1) {
            TORCH_CHECK(self.size(0) == n, "torch_vortex: addmm self is (",
                        self.size(0), ",) but the result is (", m, ", ", n, ")");
            self_kind = 1;
        } else {
            TORCH_CHECK(self.size(0) == m && self.size(1) == n,
                        "torch_vortex: addmm self is (", self.size(0), ", ",
                        self.size(1), ") but the result is (", m, ", ", n,
                        "); a column-vector addend is unsupported in v1");
            self_kind = 2;
        }
    }
    return mm_launch(mat1, mat2, 0, &self, self_kind, alpha.to<float>(),
                     beta.to<float>());
}

// Every registration below goes through a heap-allocated torch::Library that
// is deliberately never destroyed.
//
// The macros build static RegistrationHandleRAII objects, and their
// destructors call back into the Dispatcher to deregister. At process exit
// this .so's statics are destroyed *after* libtorch's own, so that call lands
// in a torn-down Dispatcher and segfaults -- reproducible as soon as a process
// exits with a kernel still queued. The registrations are process-lifetime by
// design, and leaking the handle is how torch intends that (see the note about
// heap construction in torch/library.h).
// ---- reduction call sites -------------------------------------------------

static torch::Tensor sum_impl(const torch::Tensor& self,
                              c10::OptionalArrayRef<int64_t> dim, bool keepdim,
                              std::optional<c10::ScalarType> dtype) {
    check_reduce_dtype(dtype, "sum");
    check_reduce_dims(self, dim, "sum");
    auto out = torch::empty(reduced_shape(self, dim, keepdim), self.options());
    return reduce_into(self, out, dim, VX_PRIM_OP_SUM, "sum");
}
static torch::Tensor& sum_out_impl(const torch::Tensor& self,
                                   c10::OptionalArrayRef<int64_t> dim,
                                   bool keepdim,
                                   std::optional<c10::ScalarType> dtype,
                                   torch::Tensor& out) {
    check_reduce_dtype(dtype, "sum");
    return reduce_into(self, out, dim, VX_PRIM_OP_SUM, "sum");
}

static torch::Tensor mean_impl(const torch::Tensor& self,
                               c10::OptionalArrayRef<int64_t> dim, bool keepdim,
                               std::optional<c10::ScalarType> dtype) {
    check_reduce_dtype(dtype, "mean");
    check_reduce_dims(self, dim, "mean");
    auto out = torch::empty(reduced_shape(self, dim, keepdim), self.options());
    return reduce_into(self, out, dim, VX_PRIM_OP_MEAN, "mean");
}
static torch::Tensor& mean_out_impl(const torch::Tensor& self,
                                    c10::OptionalArrayRef<int64_t> dim,
                                    bool keepdim,
                                    std::optional<c10::ScalarType> dtype,
                                    torch::Tensor& out) {
    check_reduce_dtype(dtype, "mean");
    return reduce_into(self, out, dim, VX_PRIM_OP_MEAN, "mean");
}

// amax's dim is an int[1] (empty means every dimension), not an optional.
static torch::Tensor amax_impl(const torch::Tensor& self, c10::IntArrayRef dim,
                               bool keepdim) {
    c10::OptionalArrayRef<int64_t> d =
        dim.size() == 0 ? std::nullopt : std::make_optional(dim);
    auto out = torch::empty(reduced_shape(self, d, keepdim), self.options());
    return reduce_into(self, out, d, VX_PRIM_OP_MAX, "amax");
}
static torch::Tensor& amax_out_impl(const torch::Tensor& self,
                                    c10::IntArrayRef dim, bool keepdim,
                                    torch::Tensor& out) {
    c10::OptionalArrayRef<int64_t> d =
        dim.size() == 0 ? std::nullopt : std::make_optional(dim);
    return reduce_into(self, out, d, VX_PRIM_OP_MAX, "amax");
}

static torch::Tensor amin_impl(const torch::Tensor& self, c10::IntArrayRef dim,
                               bool keepdim) {
    c10::OptionalArrayRef<int64_t> d =
        dim.size() == 0 ? std::nullopt : std::make_optional(dim);
    auto out = torch::empty(reduced_shape(self, d, keepdim), self.options());
    return reduce_into(self, out, d, VX_PRIM_OP_MIN, "amin");
}
static torch::Tensor& amin_out_impl(const torch::Tensor& self,
                                    c10::IntArrayRef dim, bool keepdim,
                                    torch::Tensor& out) {
    c10::OptionalArrayRef<int64_t> d =
        dim.size() == 0 ? std::nullopt : std::make_optional(dim);
    return reduce_into(self, out, d, VX_PRIM_OP_MIN, "amin");
}

// aten::max with no dim is a full reduction. max(dim=...) returns indices via
// the index-reduction path above.
static torch::Tensor max_impl(const torch::Tensor& self) {
    auto out = torch::empty({}, self.options());
    return reduce_into(self, out, std::nullopt, VX_PRIM_OP_MAX, "max");
}

static torch::Tensor min_impl(const torch::Tensor& self) {
    auto out = torch::empty({}, self.options());
    return reduce_into(self, out, std::nullopt, VX_PRIM_OP_MIN, "min");
}

// sort/topk use one row-wise device kernel. The input dimension is moved to
// the trailing position and made contiguous; values and int64 indices are
// moved back to the caller's dimension after the launch.
static std::tuple<torch::Tensor, torch::Tensor> sort_impl(
    const torch::Tensor& self, int64_t dim, bool descending, int64_t k) {
    check_vortex_f32(self, "sort/topk");
    TORCH_CHECK(self.dim() > 0, "torch_vortex: sort/topk needs at least one dimension");
    const int64_t nd = self.dim();
    if (dim < 0) dim += nd;
    TORCH_CHECK(dim >= 0 && dim < nd, "torch_vortex: sort/topk dim ", dim,
                " is out of range for a ", nd, "-D tensor");
    const auto moved = (dim == nd - 1) ? self : self.movedim(dim, nd - 1);
    const auto src = moved.contiguous();
    const int64_t cols = src.size(nd - 1);
    TORCH_CHECK(cols > 0, "torch_vortex: sort/topk on an empty dimension");
    TORCH_CHECK(k >= 0 && k <= cols, "torch_vortex: topk k=", k,
                " is outside [0, ", cols, "]");
    const int64_t rows = src.numel() / cols;
    std::vector<int64_t> out_shape(src.sizes().begin(), src.sizes().end());
    out_shape.back() = k;
    auto values = torch::empty(out_shape, self.options());
    auto indices = torch::empty(out_shape, self.options().dtype(at::kLong));
    if (rows == 0 || k == 0) {
        ++g_stats.skipped_launches;
    } else {
        sort_args_t args = {};
        args.dst = (uint64_t)(uintptr_t)values.data_ptr();
        args.indices = (uint64_t)(uintptr_t)indices.data_ptr();
        args.src = (uint64_t)(uintptr_t)src.data_ptr();
        args.rows = u32_dim(rows, "sort rows");
        args.cols = u32_dim(cols, "sort columns");
        args.k = u32_dim(k, "sort k");
        args.descending = descending ? 1u : 0u;
        launch(h_sort_kernel, args, args.rows);
    }
    if (dim != nd - 1) {
        values = values.movedim(nd - 1, dim);
        indices = indices.movedim(nd - 1, dim);
    }
    return std::make_tuple(values, indices);
}

static std::tuple<torch::Tensor, torch::Tensor> sort_op_impl(
    const torch::Tensor& self, int64_t dim, bool descending) {
    return sort_impl(self, dim, descending, self.size(dim < 0 ? dim + self.dim() : dim));
}

static std::tuple<torch::Tensor, torch::Tensor> topk_impl(
    const torch::Tensor& self, c10::SymInt k, int64_t dim, bool largest,
    bool sorted) {
    (void)sorted;  // the device result is sorted, which is valid for sorted=false
    return sort_impl(self, dim, largest, k.expect_int());
}

// Defined with the fallbacks, below.
static void vortex_no_fallback(const c10::OperatorHandle& op, c10::Stack* stack);

void register_vortex_ops() {
    auto* m = new torch::Library(torch::Library::IMPL, "aten",
                                 c10::DispatchKey::PrivateUse1, __FILE__, __LINE__);
#define VX_LIB (*m)
#define VX_IMPL(name, fn) VX_LIB.impl(name, fn)
    VX_IMPL("empty.memory_format", &empty_impl);
    VX_IMPL("empty_strided", &empty_strided_impl);
    VX_IMPL("arange", &arange_impl);
    VX_IMPL("arange.start", &arange_start_impl);
    VX_IMPL("arange.start_step", &arange_start_step_impl);
    VX_IMPL("copy_", &copy_impl);
    VX_IMPL("_copy_from", &copy_from_impl);
    VX_IMPL("fill_.Scalar", &fill__impl);
    VX_IMPL("zero_", &zero__impl);
    VX_IMPL("view", &view_impl);
    VX_IMPL("as_strided", &as_strided_impl);
    VX_IMPL("cat", &cat_impl);
    VX_IMPL("cat.out", &cat_out_impl);
    VX_IMPL("stack", &stack_impl);
    VX_IMPL("stack.out", &stack_out_impl);
    VX_IMPL("gather", &gather_impl);
    VX_IMPL("gather.out", &gather_out_impl);
    VX_IMPL("scatter.src", &scatter_impl);
    VX_IMPL("scatter.src_out", &scatter_src_out_impl);
    VX_IMPL("scatter_.src", &scatter__impl);
    VX_IMPL("index_add", &index_add_impl);
    VX_IMPL("index_add.out", &index_add_out_impl);
    VX_IMPL("index_add_", &index_add__impl);
    VX_IMPL("nll_loss_forward", &nll_loss_forward_impl);
    VX_IMPL("sum.dim_IntList", &sum_impl);
    VX_IMPL("sum.IntList_out", &sum_out_impl);
    VX_IMPL("mean.dim", &mean_impl);
    VX_IMPL("mean.out", &mean_out_impl);
    VX_IMPL("amax", &amax_impl);
    VX_IMPL("amax.out", &amax_out_impl);
    VX_IMPL("amin", &amin_impl);
    VX_IMPL("amin.out", &amin_out_impl);
    VX_IMPL("max", &max_impl);
    VX_IMPL("min", &min_impl);
    VX_IMPL("min.dim", &min_dim_impl);
    VX_IMPL("argmin", &argmin_impl);
    VX_IMPL("sort", &sort_op_impl);
    VX_IMPL("topk", &topk_impl);
    VX_IMPL("relu", &relu_impl);
    VX_IMPL("relu_", &relu__impl);
    VX_IMPL("add.Tensor", &add_impl);
    VX_IMPL("sub.Tensor", &sub_impl);
    VX_IMPL("mul.Tensor", &mul_impl);
    VX_IMPL("div.Tensor", &div_impl);
    VX_IMPL("maximum", &maximum_impl);
    VX_IMPL("minimum", &minimum_impl);
    VX_IMPL("add_.Tensor", &add__impl);
    VX_IMPL("sub_.Tensor", &sub__impl);
    VX_IMPL("mul_.Tensor", &mul__impl);
    VX_IMPL("div_.Tensor", &div__impl);
    VX_IMPL("add.Scalar", &add_scalar_impl);
    VX_IMPL("sub.Scalar", &sub_scalar_impl);
    VX_IMPL("mul.Scalar", &mul_scalar_impl);
    VX_IMPL("div.Scalar", &div_scalar_impl);
    VX_IMPL("add_.Scalar", &add_scalar__impl);
    VX_IMPL("sub_.Scalar", &sub_scalar__impl);
    VX_IMPL("mul_.Scalar", &mul_scalar__impl);
    VX_IMPL("div_.Scalar", &div_scalar__impl);
#define VX_REGISTER_UNARY(NAME)                                                \
    VX_IMPL(#NAME, &NAME##_impl);                                              \
    VX_IMPL(#NAME "_", &NAME##__impl)
    VX_REGISTER_UNARY(neg);
    VX_REGISTER_UNARY(abs);
    VX_REGISTER_UNARY(exp);
    VX_REGISTER_UNARY(log);
    VX_REGISTER_UNARY(sqrt);
    VX_REGISTER_UNARY(rsqrt);
    VX_REGISTER_UNARY(sigmoid);
    VX_REGISTER_UNARY(tanh);
    VX_REGISTER_UNARY(reciprocal);
    VX_REGISTER_UNARY(silu);
    VX_IMPL("gelu", &gelu_impl);
    VX_IMPL("gelu_", &gelu__impl);
#undef VX_REGISTER_UNARY
    VX_IMPL("convolution", &convolution_impl);
    VX_IMPL("native_batch_norm", &native_batch_norm_impl);
#if VX_HAS_ACCELERATOR_GUARD_API
    VX_IMPL("native_layer_norm", &native_layer_norm_impl);
    VX_IMPL("rms_norm", &rms_norm_impl);
#endif
    // _softmax and _log_softmax are the schemas; aten::softmax and
    // aten::log_softmax are composites that call them, so the public spellings
    // work without a registration of their own.
    VX_IMPL("_softmax", &softmax_impl);
    VX_IMPL("_log_softmax", &log_softmax_impl);
    VX_IMPL("logsumexp", &logsumexp_impl);
    VX_IMPL("argmax", &argmax_impl);
    VX_IMPL("max.dim", &max_dim_impl);
    VX_IMPL("max_pool2d", &max_pool2d_impl);
    VX_IMPL("avg_pool2d", &avg_pool2d_impl);
#if VX_HAS_ACCELERATOR_GUARD_API
    VX_IMPL("adaptive_avg_pool2d", &adaptive_avg_pool2d_impl);
#endif
    VX_IMPL("mm", &mm_impl_wrap);
    VX_IMPL("bmm", &bmm_impl);
    VX_IMPL("linear", &linear_impl);
    VX_IMPL("addmm", &addmm_impl);

    auto* ag = new torch::Library(torch::Library::IMPL, "_",
                                  c10::DispatchKey::AutogradPrivateUse1,
                                  __FILE__, __LINE__);
    ag->fallback(torch::CppFunction::makeFallthrough());

    auto* fb = new torch::Library(torch::Library::IMPL, "_",
                                  c10::DispatchKey::PrivateUse1,
                                  __FILE__, __LINE__);
    fb->fallback(torch::CppFunction::makeFromBoxedFunction<&vortex_no_fallback>());

#undef VX_IMPL
#undef VX_LIB
}

// ---------------------------------------------------------------------------
// Backend fallbacks
// ---------------------------------------------------------------------------

// The factory ops are registered on the PrivateUse1 key only (see the
// TORCH_LIBRARY_IMPL(aten, PrivateUse1) block above). There is deliberately no
// BackendSelect registration: registering one replaces ATen's own
// backend-select kernel process-wide, which used to route *every* non-vortex
// factory call to at::detail::empty_cpu — torch.empty(3), torch.zeros,
// torch.arange and device='meta' included. PrivateUse1 alone is sufficient:
// ATen resolves a vortex request to this key, and CPU/Meta requests to their
// own, with no extension involvement.

// Consulted only when the dispatch table has no entry for the operator at
// all; operators with a real PrivateUse1 kernel or a CompositeImplicitAutograd
// math kernel never reach it. Its job is to make "this ran on the CPU"
// impossible to miss — the README used to advertise a
// `torch_vortex.fallback_counter` that never existed.
static void vortex_no_fallback(const c10::OperatorHandle& op,
                               c10::Stack* stack) {
    (void)stack;
    TORCH_CHECK(false,
                "torch_vortex: operator ", op.schema().operator_name(),
                " has no vortex implementation; refusing to fall back to the "
                "CPU. Register it in src/vortex_ext.cpp or remove it from the "
                "model path.");
}

// Autograd pass-through for every registered op. Inference only: a gradient
// taken through a vortex tensor is discarded rather than computed, which
// torch cannot distinguish from a correct zero gradient. Training is W8.1 of
// docs/mydocs/pytorch_plan.md; until then this is the honest boundary.
// Namespace-wide fallbacks must use the catch-all TORCH_LIBRARY_IMPL(_, ...).




// ---------------------------------------------------------------------------
// Module init
// ---------------------------------------------------------------------------

void load_ops(const std::string& vxbin_path, const std::string& dl_dir) {
    // Two image sets now: this backend's own (torch_all.vxbin) and the DL
    // library's, which the unified ops launch. Every image has its own link
    // address (sw/common/module_slots.mk), so they coexist in one process --
    // that used to be impossible and is why the ops and dnn kernels were
    // merged into one image.
    //

    // Idempotent for the same image, loud for a different one.
    //
    // Re-importing the Python package (importlib.reload resets its globals)
    // used to call this again, and a second hipModuleLoad of the same image
    // fails with "address range overlaps with existing allocation" -- the
    // process-global registration had already happened, so the correct answer
    // is to do nothing. Loading a *different* image into the same process is
    // the multi-module problem (W2.4 of docs/mydocs/pytorch_plan.md): the
    // kernels were resolved from the first image, so pretending to switch
    // would silently run the wrong code.
    static std::string loaded_path;
    if (!loaded_path.empty()) {
        TORCH_CHECK(loaded_path == vxbin_path,
                    "torch_vortex: this process already loaded the kernel "
                    "image ", loaded_path, ", so it cannot also load ",
                    vxbin_path, ". Two images in one process is W2.4 of "
                    "docs/mydocs/pytorch_plan.md; until then load one.");
        return;
    }

    c10::SetAllocator(c10::DeviceType::PrivateUse1, &vortex_allocator());
    VX_CHECK(hipInit(0));
    // From here on the allocator's deleter can release through the queue
    // instead of freeing immediately (see vortex_free).
    g_device_ready = true;
    std::atexit(vortex_at_exit);

    // Cached once: conv's LMEM staging is checked against sharedMemPerBlock
    // before every launch, and the runtime's own error for an oversized block
    // is an opaque VX_ERR_INVALID_VALUE.
    hipDeviceProp_t prop;
    VX_CHECK(hipGetDeviceProperties(&prop, 0));
    g_shared_mem_per_block = (int64_t)prop.sharedMemPerBlock;

    // The DL library speaks vortex2.h and needs *this* process's device, not
    // one of its own: a second device context would break ordering with
    // everything launched through HIP. The handles come from the HIP layer,
    // which is the only thing that owns them.
    void* vxdev = nullptr;
    VX_CHECK(hipGetVxDevice(&vxdev));
    g_dl_device = (vx_device_h)vxdev;
    const auto dl_image = [&](const char* name) {
        return dl_dir + "/" + name + ".vxbin";
    };
    DL_CHECK(vx_dnn_init(g_dl_device, dl_image("dnn").c_str()));
    DL_CHECK(vx_blas_init(g_dl_device, dl_image("blas").c_str()));
    DL_CHECK(vx_prim_init(g_dl_device, dl_image("prim").c_str()));

    VX_CHECK(hipModuleLoad(&g_ops_module, vxbin_path.c_str()));
#define TORCH_KERNEL_RESOLVE(name, type, mbx, lmem)                            \
    VX_CHECK(hipModuleGetFunction(&h_##name, g_ops_module, #name));
    TORCH_KERNEL_TABLE(TORCH_KERNEL_RESOLVE)
#undef TORCH_KERNEL_RESOLVE
    loaded_path = vxbin_path;
}

// The argument-block sizes this extension was compiled with, keyed by kernel
// name. The Python side compares these against <vxbin>.meta.json, which
// gen_metadata wrote from the same header at build time: if the image and the
// host disagree, that is a stale vxbin and it must fail at init rather than
// produce wrong numbers.
std::map<std::string, int64_t> arg_sizes_impl() {
    std::map<std::string, int64_t> out;
#define TORCH_KERNEL_SIZE(name, type, mbx, lmem) out[#name] = (int64_t)sizeof(type);
    TORCH_KERNEL_TABLE(TORCH_KERNEL_SIZE)
#undef TORCH_KERNEL_SIZE
    return out;
}

// The one barrier implementation. The Python device module used to reach
// hipDeviceSynchronize through its own ctypes handle, which meant two places
// had to agree about what synchronising means.
void device_synchronize_impl() {
    VX_CHECK(hipDeviceSynchronize());
    note_device_barrier();
}

// Process-exit teardown, registered with atexit from load_ops.
//
// Without it the simx simulator's worker thread is still running
// Processor::run() -> SimPlatform::tick() when the process tears down, and it
// faults the moment anything it walks is freed -- which is why a process that
// exited with a kernel still queued dumped core while one that read the result
// back did not. Worker exit races are the plan's F23.
//
// Synchronising first matters: hipDeviceReset releases every buffer, and the
// device must not still be reading them. atexit runs before static
// destruction, so the runtime is still alive here (its own teardown waits on
// the simulator future).
void vortex_at_exit() {
    if (!g_device_ready.exchange(false)) {
        return;
    }
    device_synchronize_impl();
    // The DL modules hold their own refs on the device and their images; the
    // extension owns their lifetime because nothing in sw/dl finalizes them
    // (it has no lifetime hooks by construction). Order matters: finalize
    // first, then reset, or the device goes away underneath them.
    vx_blas_finalize();
    vx_dnn_finalize();
    vx_prim_finalize();
    hipDeviceReset();
}

std::map<std::string, int64_t> device_properties_impl() {
    hipDeviceProp_t prop;
    VX_CHECK(hipGetDeviceProperties(&prop, 0));
    return {
        {"shared_mem_per_block", (int64_t)prop.sharedMemPerBlock},
        {"max_threads_per_block", (int64_t)prop.maxThreadsPerBlock},
        {"warp_size", (int64_t)prop.warpSize},
        {"total_global_mem", (int64_t)prop.totalGlobalMem},
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    register_vortex_ops();
    m.def("load_ops", &load_ops,
          "initialize the vortex backend (allocator + kernel image)");
    m.def("stats", &stats_snapshot,
          "device counters: launches, transfer bytes, allocations, syncs");
    m.def("reset_stats", &stats_reset, "zero the counters returned by stats()");
    m.def("arg_sizes", &arg_sizes_impl,
          "argument-block sizes this extension was compiled with");
    m.def("device_synchronize", &device_synchronize_impl,
          "block until every queued command on the device has retired");
    m.def("device_properties", &device_properties_impl, "cached device limits");
}
