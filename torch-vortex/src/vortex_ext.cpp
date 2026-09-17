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

#include <atomic>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <c10/core/Allocator.h>
#include <ATen/EmptyTensor.h>
#include <ATen/ops/view_native.h>
#include <c10/core/Device.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

extern "C" {
#include <hip/hip_runtime_api.h>
}

// The single definition of every kernel argument block, shared with the device
// compiler. Never declare an argument struct in this file.
#include "torch_kernel_args.h"

#define VX_CHECK(expr)                                                        \
    do {                                                                      \
        hipError_t _e = (expr);                                               \
        TORCH_CHECK(_e == hipSuccess, "torch_vortex: " #expr " failed: ",     \
                    hipGetErrorString(_e));                                   \
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

// Called after anything that drains the queue host-side.
void note_barrier() {
    g_barrier_epoch.store(g_launch_epoch.load());
}

bool work_since_last_barrier() {
    return g_launch_epoch.load() != g_barrier_epoch.load();
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
    g_stats.frees = 0;
    g_stats.immediate_frees = 0;
    g_stats.blocking_syncs = 0;
    g_stats.host_numeric_ops = 0;
}

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
    VX_CHECK(hipModuleLaunchKernel(f, gx, gy, gz, bx, 1, 1, lmem, nullptr,
                                   nullptr, extra));
    ++g_stats.launches;
    ++g_launch_epoch;
}

// The previous calling convention (a single pointer in kernelParams, with the
// size read from image metadata) is deliberately not used anywhere any more.

} // namespace

// ---------------------------------------------------------------------------
// Allocator: device memory via hipMalloc (vx_buffer_create under it)
// ---------------------------------------------------------------------------

namespace {

void vortex_free(void* ctx);

struct VortexAllocator final : public c10::Allocator {
    c10::DataPtr allocate(size_t n) override {
        void* p = nullptr;
        if (n != 0) {
            VX_CHECK(hipMalloc(&p, n));
            ++g_stats.allocations;
            g_stats.allocated_bytes += n;
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

VortexAllocator g_vortex_allocator;  // static lifetime (SetAllocator is non-owning)

// ---------------------------------------------------------------------------
// Device guard: single device, single implicit stream
// ---------------------------------------------------------------------------

struct VortexGuardImpl final : public c10::impl::DeviceGuardImplInterface {
    c10::DeviceType type() const override {
        return c10::DeviceType::PrivateUse1;
    }
    c10::Device exchangeDevice(c10::Device d) const override {
        auto old = current_;
        if (d.index() >= 0) {
            TORCH_CHECK(d.type() == c10::DeviceType::PrivateUse1 && d.index() == 0,
                        "torch_vortex: only device 0 exists");
        }
        return old;
    }
    void setDevice(c10::Device d) const override {
        if (d.index() >= 0) {
            TORCH_CHECK(d.type() == c10::DeviceType::PrivateUse1 && d.index() == 0,
                        "torch_vortex: only device 0 exists");
        }
    }
    c10::Device getDevice() const override {
        return c10::Device(c10::DeviceType::PrivateUse1, 0);
    }
    c10::DeviceIndex deviceCount() const noexcept override { return 1; }
    void uncheckedSetDevice(c10::Device d) const noexcept override {
        (void)d;  // single device; nothing to switch
    }
    c10::Stream getStream(c10::Device) const noexcept override {
        return getDefaultStream(c10::Device(c10::DeviceType::PrivateUse1, 0));
    }
    c10::Stream exchangeStream(c10::Stream s) const noexcept override {
        auto old = stream_;
        stream_ = s;
        return old;
    }
    c10::Stream getDefaultStream(c10::Device) const override {
        return c10::Stream(c10::Stream::DEFAULT,
                           c10::Device(c10::DeviceType::PrivateUse1, 0));
    }
    c10::Stream getNewStream(c10::Device, int) const override {
        return getDefaultStream(c10::Device(c10::DeviceType::PrivateUse1, 0));
    }

   private:
    static inline c10::Device current_ = c10::Device(c10::DeviceType::PrivateUse1, 0);
    static inline c10::Stream stream_ =
        c10::Stream(c10::Stream::DEFAULT, c10::Device(c10::DeviceType::PrivateUse1, 0));
};

} // namespace

C10_REGISTER_GUARD_IMPL(PrivateUse1, VortexGuardImpl);

// ---------------------------------------------------------------------------
// Op kernel image
// ---------------------------------------------------------------------------

namespace {

// The argument blocks (binary_args_t, fill_args_t, conv_args_t, ...) are
// defined once in kernels/torch_kernel_args.h and shared verbatim with the
// device compiler. Never declare one locally in this file.

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

void launch_binary(hipFunction_t f, uint64_t dst, uint64_t a, uint64_t b,
                   uint32_t n) {
    binary_args_t args = {dst, a, b, n, 0};
    launch(f, args, (n + 3) / 4);
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
        c10::IntArrayRef(sizes), &g_vortex_allocator, kVortexDispatchKeys,
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
        &g_vortex_allocator, kVortexDispatchKeys, dtype);
    return torch::Tensor(std::move(base));
}

// copy_: same dtype, same shape, contiguous destination, offset 0, distinct
// storage. Everything else is refused by name.
//
// The previous version memcpy'd self.nbytes() bytes whatever the source was:
// a dtype change became a bit-pattern copy, a smaller source was over-read
// (a host out-of-bounds read on H2D), and a non-contiguous tensor copied
// storage order rather than its logical contents.
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

    TORCH_CHECK(self.scalar_type() == src.scalar_type(),
                "torch_vortex: copy_ does not convert dtypes yet (",
                src.scalar_type(), " -> ", self.scalar_type(),
                "); casting is W3.2/W4.1 in docs/mydocs/pytorch_plan.md");
    TORCH_CHECK(self.sizes() == src.sizes(), "torch_vortex: copy_ shape mismatch: ",
                self.sizes(), " <- ", src.sizes());
    TORCH_CHECK(self.is_contiguous(),
                "torch_vortex: copy_ into a non-contiguous vortex tensor is "
                "unsupported in v1 (strided destinations are W3.2)");
    TORCH_CHECK(self.storage_offset() == 0,
                "torch_vortex: copy_ into a vortex tensor at storage_offset ",
                self.storage_offset(), " is unsupported in v1");

    // A non-contiguous CPU source is cheap to fix on the host and is a
    // reasonable thing to write (t.t().to("vortex")), so materialise it rather
    // than either refusing or silently copying storage order.
    torch::Tensor src_contig = src;
    if (!src_dev && !src.is_contiguous()) {
        src_contig = src.contiguous();
        ++g_stats.host_numeric_ops;
    }

    const int64_t bytes = self.numel() * self.element_size();
    if (bytes == 0) {
        return self;
    }
    if (non_blocking) {
        // Honest rather than silent: hipMemcpy enqueues and then waits on its
        // own completion event, so this call blocks whatever the flag says.
        // Counted so a model that depends on overlap is visible.
        ++g_stats.blocking_syncs;
    }
    VX_CHECK(hipMemcpy(self.data_ptr(), src_contig.data_ptr(), (size_t)bytes, kind));
    // hipMemcpy enqueues and then waits on its own completion event; the queue
    // is FIFO, so everything enqueued before it has retired by the time it
    // returns. The allocator's fast path relies on knowing that.
    note_barrier();
    switch (kind) {
        case hipMemcpyHostToDevice: g_stats.h2d_bytes += (uint64_t)bytes; break;
        case hipMemcpyDeviceToHost: g_stats.d2h_bytes += (uint64_t)bytes; break;
        default: g_stats.d2d_bytes += (uint64_t)bytes; break;
    }
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
static torch::Tensor view_impl(const torch::Tensor& self,
                               c10::SymIntArrayRef sym_sizes) {
    TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
                "torch_vortex: view on non-vortex tensor");
    auto out = at::native::view_symint(self, sym_sizes);
    TORCH_CHECK(out.device().type() == c10::DeviceType::PrivateUse1,
                "torch_vortex: view produced a tensor on ", out.device());
    return out;
}

// relu, non-inplace. PyTorch's relu is max(x, 0) with two properties a naive
// `x > 0 ? x : 0` gets wrong: NaN propagates, and -0.0 stays -0.0. It also
// must not touch its input — this used to run the in-place kernel on self and
// return self, so a residual branch sharing the input was silently modified.
static torch::Tensor relu_impl(const torch::Tensor& self) {
    check_vortex_f32(self, "relu");
    auto out = torch::empty_like(self);
    unary_args_t args = {(uint64_t)(uintptr_t)out.data_ptr(),
                         (uint64_t)(uintptr_t)self.data_ptr(),
                         (uint32_t)self.numel(), 0};
    launch(h_relu_out_kernel, args, (uint32_t)((self.numel() + 3) / 4));
    return out;
}

static torch::Tensor& relu__impl(torch::Tensor& self) {
    check_vortex_f32(self, "relu_");
    fill_args_t args = {(uint64_t)(uintptr_t)self.data_ptr(),
                        (uint32_t)self.numel(), 0.0f, 0};
    launch(h_relu_kernel, args, (uint32_t)((self.numel() + 3) / 4));
    return self;
}

static torch::Tensor add_impl(const torch::Tensor& a, const torch::Tensor& b,
                              const c10::Scalar& alpha) {
    check_vortex_f32(a, "add.a");
    check_vortex_f32(b, "add.b");
    TORCH_CHECK(alpha.to<double>() == 1.0,
                "torch_vortex: add alpha != 1 unsupported in v1");
    TORCH_CHECK(a.sizes() == b.sizes(), "torch_vortex: add needs equal shapes");
    auto out = torch::empty_like(a);
    launch_binary(h_add_kernel, (uint64_t)(uintptr_t)out.data_ptr(),
                  (uint64_t)(uintptr_t)a.data_ptr(),
                  (uint64_t)(uintptr_t)b.data_ptr(), (uint32_t)a.numel());
    return out;
}

static torch::Tensor mul_impl(const torch::Tensor& a, const torch::Tensor& b) {
    check_vortex_f32(a, "mul.a");
    check_vortex_f32(b, "mul.b");
    TORCH_CHECK(a.sizes() == b.sizes(), "torch_vortex: mul needs equal shapes");
    auto out = torch::empty_like(a);
    launch_binary(h_mul_kernel, (uint64_t)(uintptr_t)out.data_ptr(),
                  (uint64_t)(uintptr_t)a.data_ptr(),
                  (uint64_t)(uintptr_t)b.data_ptr(), (uint32_t)a.numel());
    return out;
}


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
                          const char* what) {
    TORCH_CHECK(k > 0, "torch_vortex: ", what, ": kernel must be > 0, got ", k);
    TORCH_CHECK(stride > 0,
                "torch_vortex: ", what, ": stride must be > 0, got ", stride);
    TORCH_CHECK(pad >= 0,
                "torch_vortex: ", what, ": padding must be >= 0, got ", pad);
    TORCH_CHECK(in + 2 * pad >= k, "torch_vortex: ", what, ": input ", in,
                " with padding ", pad, " is smaller than kernel ", k);
    return (in + 2 * pad - k) / stride + 1;
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

static void launch_conv(const conv_args_t& args) {
    // one CTA per (output row, output channel, sample); 16 threads per row;
    // all of one filter's weights staged in LMEM
    launch(h_tv_conv2d_kernel, args, args.ho, args.co, args.n, 16,
           (uint32_t)(args.ci * args.kh * args.kw * 4));
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
    TORCH_CHECK(groups == 1, "torch_vortex: grouped conv unsupported in v1");
    for (auto d : dilation) TORCH_CHECK(d == 1, "torch_vortex: dilation must be 1");
    for (auto o : output_padding) TORCH_CHECK(o == 0, "torch_vortex: output_padding must be 0");
    const int64_t sh = stride.size() > 0 ? stride[0] : 1;
    const int64_t sw = stride.size() > 1 ? stride[1] : sh;
    const int64_t ph = padding.size() > 0 ? padding[0] : 0;
    const int64_t pw = padding.size() > 1 ? padding[1] : ph;

    const auto& is = input.sizes();
    const auto& ws = weight.sizes();
    const int64_t co = ws[0], ci = ws[1];
    const int64_t kh = ws[2], kw = ws[3];
    TORCH_CHECK(is[1] == ci, "torch_vortex: conv channel mismatch");
    const int64_t ho = window_out(is[2], kh, ph, sh, "conv height");
    const int64_t wo = window_out(is[3], kw, pw, sw, "conv width");

    // One output channel stages all of its weights in LMEM at once. This is
    // the real ceiling on a full-resolution ResNet stem (ci=512, 3x3, fp32
    // needs 18 KiB against a 16 KiB default): shrinking the input image does
    // not reduce it, so say so instead of hanging.
    const int64_t lmem_needed = ci * kh * kw * 4;
    if (g_shared_mem_per_block > 0) {
        TORCH_CHECK(lmem_needed <= g_shared_mem_per_block, "torch_vortex: conv ",
                    "needs ", lmem_needed, " bytes of local memory to stage one ",
                    "filter (ci=", ci, " kh=", kh, " kw=", kw, " x 4 bytes), but ",
                    "only ", g_shared_mem_per_block, " bytes are available. ",
                    "Tile the weights across output channels; a smaller input ",
                    "image does not reduce this.");
    }

    auto out = torch::empty({is[0], co, ho, wo}, input.options());
    uint64_t baddr = 0;
    // Composite layers pass None as an *undefined* tensor inside the
    // optional (still has_value) — treat undefined as absent.
    if (bias.has_value() && bias->defined()) {
        check_cnn_f32(*bias, "conv.bias", 1);
        TORCH_CHECK(bias->numel() == co, "torch_vortex: bias size mismatch");
        baddr = (uint64_t)(uintptr_t)bias->data_ptr();
    }
    conv_args_t args = {(uint64_t)(uintptr_t)input.data_ptr(),
                        (uint64_t)(uintptr_t)weight.data_ptr(), baddr,
                        (uint64_t)(uintptr_t)out.data_ptr(),
                        u32_dim(is[0], "conv batch"), u32_dim(ci, "conv ci"),
                        u32_dim(is[2], "conv hi"), u32_dim(is[3], "conv wi"),
                        u32_dim(co, "conv co"), u32_dim(ho, "conv ho"),
                        u32_dim(wo, "conv wo"), u32_dim(kh, "conv kh"),
                        u32_dim(kw, "conv kw"), u32_dim(ph, "conv ph"),
                        u32_dim(pw, "conv pw"), u32_dim(sh, "conv sh"),
                        u32_dim(sw, "conv sw"), baddr != 0};
    launch_conv(args);
    return out;
}

static torch::Tensor pool_impl(const torch::Tensor& self,
                               c10::IntArrayRef kernel_size,
                               c10::IntArrayRef stride,
                               c10::IntArrayRef padding, uint32_t op) {
    check_cnn_f32(self, "pool.input", 4);
    const int64_t kh = kernel_size[0], kw = kernel_size[1];
    const int64_t sh = stride.size() > 0 ? stride[0] : kh;
    const int64_t sw = stride.size() > 1 ? stride[1] : sh;
    const int64_t ph = padding.size() > 0 ? padding[0] : 0;
    const int64_t pw = padding.size() > 1 ? padding[1] : ph;
    const auto& s = self.sizes();
    const int64_t ho = window_out(s[2], kh, ph, sh, "pool height");
    const int64_t wo = window_out(s[3], kw, pw, sw, "pool width");
    auto out = torch::empty({s[0], s[1], ho, wo}, self.options());
    pool_args_t args = {(uint64_t)(uintptr_t)self.data_ptr(),
                        (uint64_t)(uintptr_t)out.data_ptr(),
                        u32_dim(s[0], "pool batch"),
                        u32_dim(s[1], "pool channels"),
                        u32_dim(s[2], "pool input height"),
                        u32_dim(s[3], "pool input width"),
                        u32_dim(ho, "pool output height"),
                        u32_dim(wo, "pool output width"),
                        u32_dim(kh, "pool kernel height"),
                        u32_dim(kw, "pool kernel width"),
                        u32_dim(ph, "pool padding height"),
                        u32_dim(pw, "pool padding width"),
                        u32_dim(sh, "pool stride height"),
                        u32_dim(sw, "pool stride width"), op};
    // one CTA per (output row, channel, sample)
    launch(h_tv_pool2d_kernel, args, args.ho, args.c, args.n, 16);
    return out;
}

static torch::Tensor max_pool2d_impl(const torch::Tensor& self,
                                     c10::IntArrayRef kernel,
                                     c10::IntArrayRef stride,
                                     c10::IntArrayRef padding,
                                     c10::IntArrayRef dilation,
                                     bool ceil_mode) {
    TORCH_CHECK(!ceil_mode, "torch_vortex: ceil_mode unsupported in v1");
    for (auto d : dilation) TORCH_CHECK(d == 1, "torch_vortex: dilation must be 1");
    return pool_impl(self, kernel, stride, padding, 0);
}


static torch::Tensor adaptive_avg_pool2d_impl(const torch::Tensor& self,
                                              c10::SymIntArrayRef output_size) {
    check_cnn_f32(self, "pool.input", 4);
    auto os = sym_to_vec(output_size);
    TORCH_CHECK(os.size() == 2 && os[0] == 1 && os[1] == 1,
                "torch_vortex: adaptive_avg_pool2d only output (1,1) in v1");
    const auto& s = self.sizes();
    return pool_impl(self, {(int64_t)s[2], (int64_t)s[3]}, {}, {}, 1);
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

    auto out = torch::empty_like(input);
    bn_args_t args = {(uint64_t)(uintptr_t)input.data_ptr(),
                      (uint64_t)(uintptr_t)running_mean->data_ptr(),
                      (uint64_t)(uintptr_t)running_var->data_ptr(),
                      has_affine ? (uint64_t)(uintptr_t)weight->data_ptr() : 0,
                      has_affine ? (uint64_t)(uintptr_t)bias->data_ptr() : 0,
                      (uint64_t)(uintptr_t)out.data_ptr(),
                      u32_numel(input, "bn input"), u32_dim(c, "bn channels"),
                      u32_dim(hw, "bn spatial span"), (float)eps,
                      has_affine ? 1u : 0u};
    launch(h_tv_bn_affine_kernel, args, (uint32_t)((total + 3) / 4));
    // rstd is computed inside the kernel from running_var, so this path makes
    // no host round-trip and allocates no scratch buffer. The previous version
    // copied running_var to the host, took a sqrt, copied it back, and freed
    // the scratch immediately after an asynchronous launch.
    auto empty_aux = torch::empty({0}, input.options());
    return std::make_tuple(out, empty_aux, empty_aux);
}

// The single matmul launcher. mm, linear and addmm are thin wrappers with
// different epilogues; having one implementation is what stops them from
// disagreeing about which axis is contracted.
static torch::Tensor mm_launch(const torch::Tensor& a, const torch::Tensor& b,
                               uint32_t transb, const torch::Tensor* self,
                               uint32_t self_kind, float alpha, float beta) {
    check_cnn_f32(a, "mm.mat1", 2);
    check_cnn_f32(b, "mm.mat2", 2);
    const int64_t k = a.size(1);
    const int64_t k_other = b.size(transb ? 1 : 0);
    TORCH_CHECK(k == k_other, "torch_vortex: matmul contraction mismatch: mat1 ",
                "is (", a.size(0), ", ", k, ") and mat2 is (", b.size(0), ", ",
                b.size(1), "), so the inner dimensions differ");
    const int64_t m = a.size(0), n = b.size(transb ? 0 : 1);
    auto out = torch::empty({m, n}, a.options());
    mm_args_t args = {(uint64_t)(uintptr_t)a.data_ptr(),
                      (uint64_t)(uintptr_t)b.data_ptr(),
                      (uint64_t)(uintptr_t)out.data_ptr(),
                      u32_dim(m, "matmul m"), u32_dim(n, "matmul n"),
                      u32_dim(k, "matmul k"), transb};
    // K == 0 is legal and needs no special case: the accumulation loop does
    // not execute and the kernel writes zeros.
    launch(h_tv_mm_kernel, args, (uint32_t)((n + 15) / 16),
           (uint32_t)((m + 15) / 16), 1, 16, 1024);

    // torch.addmm's alpha/beta/self scaling is a separate in-place pass rather
    // than an epilogue in tv_mm_kernel: that branch inside the matmul kernel
    // makes VOLT accumulate wrongly for any n that is not a multiple of 16.
    // See tv_mm_epilogue_kernel and W5.4 of docs/mydocs/pytorch_plan.md.
    if ((self_kind != 0 || alpha != 1.0f) && out.numel() != 0) {
        mm_epilogue_args_t ep = {
            (uint64_t)(uintptr_t)out.data_ptr(),
            self ? (uint64_t)(uintptr_t)self->data_ptr() : 0,
            u32_dim(m, "epilogue m"), u32_dim(n, "epilogue n"), alpha, beta,
            self_kind, 0};
        launch(h_tv_mm_epilogue_kernel, ep, (uint32_t)((out.numel() + 3) / 4));
    }
    return out;
}

static torch::Tensor mm_impl_wrap(const torch::Tensor& a,
                                  const torch::Tensor& b) {
    return mm_launch(a, b, 0, nullptr, 0, 1.0f, 0.0f);
}

static torch::Tensor linear_impl(
    const torch::Tensor& input, const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias) {
    // v1 is 2-D only. A 1-D input has a different output rank, and quietly
    // treating it as 2-D is worse than refusing it.
    check_cnn_f32(input, "linear.input", 2);
    check_cnn_f32(weight, "linear.weight", 2);
    TORCH_CHECK(input.size(1) == weight.size(1),
                "torch_vortex: linear input has ", input.size(1),
                " features but weight expects ", weight.size(1));
    auto out = mm_launch(input, weight, 1, nullptr, 0, 1.0f, 0.0f);  // x @ w^T
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
    return out;
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

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("empty.memory_format", &empty_impl);
    m.impl("empty_strided", &empty_strided_impl);
    m.impl("copy_", &copy_impl);
    m.impl("_copy_from", &copy_from_impl);
    m.impl("fill_.Scalar", &fill__impl);
    m.impl("zero_", &zero__impl);
    m.impl("view", &view_impl);
    m.impl("relu", &relu_impl);
    m.impl("relu_", &relu__impl);
    m.impl("add.Tensor", &add_impl);
    m.impl("mul.Tensor", &mul_impl);
    m.impl("convolution", &convolution_impl);
    m.impl("native_batch_norm", &native_batch_norm_impl);
    m.impl("max_pool2d", &max_pool2d_impl);
    m.impl("adaptive_avg_pool2d", &adaptive_avg_pool2d_impl);
    m.impl("mm", &mm_impl_wrap);
    m.impl("linear", &linear_impl);
    m.impl("addmm", &addmm_impl);
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
TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&vortex_no_fallback>());
}

// ---------------------------------------------------------------------------
// Module init
// ---------------------------------------------------------------------------

void load_ops(const std::string& vxbin_path, const std::string& dnn_path) {
    // One combined image (ops + dnn kernels): the simx backend loads modules
    // at a fixed base address, so separate images would overlap. `dnn_path` is
    // accepted for call compatibility but ignored.
    (void)dnn_path;

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

    c10::SetAllocator(c10::DeviceType::PrivateUse1, &g_vortex_allocator);
    VX_CHECK(hipInit(0));
    // From here on the allocator's deleter can release through the queue
    // instead of freeing immediately (see vortex_free).
    g_device_ready = true;

    // Cached once: conv's LMEM staging is checked against sharedMemPerBlock
    // before every launch, and the runtime's own error for an oversized block
    // is an opaque VX_ERR_INVALID_VALUE.
    hipDeviceProp_t prop;
    VX_CHECK(hipGetDeviceProperties(&prop, 0));
    g_shared_mem_per_block = (int64_t)prop.sharedMemPerBlock;

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
    note_barrier();
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
