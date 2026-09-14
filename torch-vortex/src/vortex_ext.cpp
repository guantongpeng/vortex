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

#include <c10/core/Allocator.h>
#include <ATen/EmptyTensor.h>
#include <c10/core/Device.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

extern "C" {
#include <hip/hip_runtime_api.h>
}

#define VX_CHECK(expr)                                                        \
    do {                                                                      \
        hipError_t _e = (expr);                                               \
        TORCH_CHECK(_e == hipSuccess, "torch_vortex: " #expr " failed: ",     \
                    hipGetErrorString(_e));                                   \
    } while (0)

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
        }
        return c10::DataPtr(p, p, &vortex_free,
                            c10::Device(c10::DeviceType::PrivateUse1, 0));
    }
    void copy_data(void* dest, const void* src, std::size_t count) const override {
        // Device-to-device byte copy through the host-visible buffer
        // mapping (same-address-space model on current backends).
        VX_CHECK(hipMemcpy(dest, src, count, hipMemcpyDeviceToDevice));
    }
};

void vortex_free(void* ctx) {
    if (ctx != nullptr) {
        hipFree(ctx);
    }
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

struct BinaryArgs {   // mirrors kernels/ops.hip binary_args_t (host side)
    uint64_t dst;
    uint64_t a;
    uint64_t b;
    uint32_t n;
    uint32_t pad;
};
struct FillArgs {
    uint64_t dst;
    uint32_t n;
    float value;
    uint32_t pad;
};

hipModule_t g_ops_module = nullptr;
hipFunction_t g_add = nullptr;
hipFunction_t g_mul = nullptr;
hipFunction_t g_fill = nullptr;

void launch_binary(hipFunction_t f, uint64_t dst, uint64_t a, uint64_t b,
                   uint32_t n) {
    BinaryArgs args = {dst, a, b, n, 0};
    void* params[1] = {&args};
    uint32_t blocks = (n + 3) / 4;
    VX_CHECK(hipModuleLaunchKernel(f, blocks, 1, 1, 4, 1, 1, 0, nullptr,
                                   params, nullptr));
}

void check_vortex_f32(const torch::Tensor& t, const char* what) {
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1,
                "torch_vortex: ", what, " must be a vortex tensor");
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

static torch::Tensor empty_impl(c10::SymIntArrayRef sym_size,
                                std::optional<c10::ScalarType> dtype_opt,
                                std::optional<c10::Layout> layout_opt,
                                std::optional<c10::Device> device_opt,
                                std::optional<bool> pin_memory_opt,
                                std::optional<c10::MemoryFormat> memory_format_opt) {
    TORCH_CHECK(!layout_opt.has_value() || *layout_opt == c10::Layout::Strided,
                "torch_vortex: strided layout only");
    TORCH_CHECK(!device_opt.has_value() ||
                    device_opt->type() == c10::DeviceType::PrivateUse1,
                "torch_vortex: wrong device in empty()");
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
    TORCH_CHECK(!device_opt.has_value() ||
                    device_opt->type() == c10::DeviceType::PrivateUse1,
                "torch_vortex: wrong device in empty_strided()");
    auto dtype = dtype_opt.value_or(at::kFloat);
    auto sizes = sym_to_vec(sym_size);
    auto strides = sym_to_vec(sym_stride);
    auto base = at::detail::empty_strided_generic(
        c10::IntArrayRef(sizes), c10::IntArrayRef(strides),
        &g_vortex_allocator, kVortexDispatchKeys, dtype);
    return torch::Tensor(std::move(base));
}

static torch::Tensor& copy_impl(torch::Tensor& self, const torch::Tensor& src,
                                bool non_blocking) {
    const auto kind = self.is_privateuseone()
                          ? (src.is_cpu() ? hipMemcpyHostToDevice
                                          : hipMemcpyDeviceToDevice)
                          : hipMemcpyDeviceToHost;
    const int64_t bytes =
        self.nbytes() ? self.nbytes() : (int64_t)self.numel() * self.element_size();
    if (bytes != 0) {
        VX_CHECK(hipMemcpy(self.data_ptr(), src.data_ptr(), (size_t)bytes, kind));
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
    FillArgs args = {(uint64_t)(uintptr_t)self.data_ptr(),
                     (uint32_t)self.numel(), value.to<float>(), 0};
    void* params[1] = {&args};
    uint32_t n = (uint32_t)self.numel();
    VX_CHECK(hipModuleLaunchKernel(g_fill, (n + 3) / 4, 1, 1, 4, 1, 1, 0,
                                   nullptr, params, nullptr));
    return self;
}

static torch::Tensor& zero__impl(torch::Tensor& self) {
    return fill__impl(self, c10::Scalar(0.0));
}

static torch::Tensor add_impl(const torch::Tensor& a, const torch::Tensor& b,
                              const c10::Scalar& alpha) {
    check_vortex_f32(a, "add.a");
    check_vortex_f32(b, "add.b");
    TORCH_CHECK(alpha.to<double>() == 1.0,
                "torch_vortex: add alpha != 1 unsupported in v1");
    TORCH_CHECK(a.sizes() == b.sizes(), "torch_vortex: add needs equal shapes");
    auto out = torch::empty_like(a);
    launch_binary(g_add, (uint64_t)(uintptr_t)out.data_ptr(),
                  (uint64_t)(uintptr_t)a.data_ptr(),
                  (uint64_t)(uintptr_t)b.data_ptr(), (uint32_t)a.numel());
    return out;
}

static torch::Tensor mul_impl(const torch::Tensor& a, const torch::Tensor& b) {
    check_vortex_f32(a, "mul.a");
    check_vortex_f32(b, "mul.b");
    TORCH_CHECK(a.sizes() == b.sizes(), "torch_vortex: mul needs equal shapes");
    auto out = torch::empty_like(a);
    launch_binary(g_mul, (uint64_t)(uintptr_t)out.data_ptr(),
                  (uint64_t)(uintptr_t)a.data_ptr(),
                  (uint64_t)(uintptr_t)b.data_ptr(), (uint32_t)a.numel());
    return out;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("empty.memory_format", &empty_impl);
    m.impl("empty_strided", &empty_strided_impl);
    m.impl("copy_", &copy_impl);
    m.impl("_copy_from", &copy_from_impl);
    m.impl("fill_.Scalar", &fill__impl);
    m.impl("zero_", &zero__impl);
    m.impl("add.Tensor", &add_impl);
    m.impl("mul.Tensor", &mul_impl);
}

// Factory ops dispatch through BackendSelect: route vortex-device calls to
// the implementations above and chain everything else to the CPU factory.
static torch::Tensor bs_empty_impl(
    c10::SymIntArrayRef sym_size, std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt, std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
    if (device_opt.has_value() &&
        device_opt->type() == c10::DeviceType::PrivateUse1) {
        return empty_impl(sym_size, dtype_opt, layout_opt, device_opt,
                          pin_memory_opt, memory_format_opt);
    }
    auto sizes = sym_to_vec(sym_size);
    return torch::Tensor(at::detail::empty_cpu(
        c10::IntArrayRef(sizes), dtype_opt, layout_opt, device_opt,
        pin_memory_opt, memory_format_opt));
}

static torch::Tensor bs_empty_strided_impl(
    c10::SymIntArrayRef sym_size, c10::SymIntArrayRef sym_stride,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt, std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
    if (device_opt.has_value() &&
        device_opt->type() == c10::DeviceType::PrivateUse1) {
        return empty_strided_impl(sym_size, sym_stride, dtype_opt, layout_opt,
                                  device_opt, pin_memory_opt);
    }
    auto sizes = sym_to_vec(sym_size);
    auto strides = sym_to_vec(sym_stride);
    return torch::Tensor(at::detail::empty_strided_cpu(
        c10::IntArrayRef(sizes), c10::IntArrayRef(strides), dtype_opt,
        layout_opt, device_opt, pin_memory_opt));
}

TORCH_LIBRARY_IMPL(aten, BackendSelect, m) {
    m.impl("empty.memory_format", &bs_empty_impl);
    m.impl("empty_strided", &bs_empty_strided_impl);
}

// ---------------------------------------------------------------------------
// Module init
// ---------------------------------------------------------------------------

void load_ops(const std::string& vxbin_path) {
    c10::SetAllocator(c10::DeviceType::PrivateUse1, &g_vortex_allocator);
    VX_CHECK(hipInit(0));
    VX_CHECK(hipModuleLoad(&g_ops_module, vxbin_path.c_str()));
    VX_CHECK(hipModuleGetFunction(&g_add, g_ops_module, "add_kernel"));
    VX_CHECK(hipModuleGetFunction(&g_mul, g_ops_module, "mul_kernel"));
    VX_CHECK(hipModuleGetFunction(&g_fill, g_ops_module, "fill_kernel"));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("load_ops", &load_ops, "initialize the vortex backend (allocator + ops image)");
}
