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

#include <cmath>

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
hipFunction_t g_relu = nullptr;

// dnn image (plan P5-02): conv2d / pooling / bn / mm / bias add
struct ConvArgs {
    uint64_t in, weight, bias, out;
    uint32_t n, ci, hi, wi, co, ho, wo;
    uint32_t kh, kw, ph, pw, sh, sw, has_bias;
};
struct PoolArgs {
    uint64_t in, out;
    uint32_t n, c, hi, wi, ho, wo;
    uint32_t kh, kw, ph, pw, sh, sw, op;
};
struct BnArgs {
    uint64_t in, mean, rstd, weight, bias, out;
    uint32_t total, c;
};
struct MmArgs {
    uint64_t a, b, out;
    uint32_t m, n, k;
    uint32_t transb;
};
struct BiasArgs {
    uint64_t out, bias;
    uint32_t m, n;
};

hipModule_t g_dnn_module = nullptr;
hipFunction_t g_conv2d = nullptr;
hipFunction_t g_pool2d = nullptr;
hipFunction_t g_bn = nullptr;
hipFunction_t g_mm = nullptr;
hipFunction_t g_bias_add = nullptr;

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

// view on contiguous tensors: pure metadata (new sizes, fresh TensorImpl
// over the same storage). Offset-0 views only — enough for flatten-style
// reshapes; strided/offset views stay unsupported loudly.
static torch::Tensor view_impl(const torch::Tensor& self,
                               c10::SymIntArrayRef sym_sizes) {
    TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
                "torch_vortex: view on non-vortex tensor");
    TORCH_CHECK(self.is_contiguous(),
                "torch_vortex: view on non-contiguous tensor unsupported in v1");
    auto sizes = sym_to_vec(sym_sizes);
    auto impl = c10::make_intrusive<c10::TensorImpl>(
        c10::Storage(self.storage()), self.key_set(), self.dtype());
    impl->set_sizes_contiguous(c10::IntArrayRef(sizes));
    return torch::Tensor(std::move(impl));
}

static torch::Tensor relu_impl(const torch::Tensor& self) {
    check_vortex_f32(self, "relu");
    FillArgs args = {(uint64_t)(uintptr_t)self.data_ptr(),
                     (uint32_t)self.numel(), 0.0f, 0};
    void* params[1] = {&args};
    uint32_t n = (uint32_t)self.numel();
    VX_CHECK(hipModuleLaunchKernel(g_relu, (n + 3) / 4, 1, 1, 4, 1, 1, 0,
                                   nullptr, params, nullptr));
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


// ---------------------------------------------------------------------------
// dnn ops (plan P5-02): the ResNet op set on device. All constraints are
// TORCH_CHECKed — no silent CPU fallback.
// ---------------------------------------------------------------------------

static void check_cnn_f32(const torch::Tensor& t, const char* what,
                          int64_t dims) {
    TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1,
                "torch_vortex: ", what, " must be a vortex tensor");
    TORCH_CHECK(t.scalar_type() == at::kFloat,
                "torch_vortex: ", what, " must be float32 in v1");
    TORCH_CHECK(t.is_contiguous(), "torch_vortex: ", what, " must be contiguous");
    TORCH_CHECK(t.dim() == dims, "torch_vortex: ", what, " must be ", dims, "-D");
}

static void launch_conv(hipFunction_t f, uint64_t in, uint64_t w, uint64_t b,
                        uint64_t out, uint32_t n, uint32_t ci, uint32_t hi,
                        uint32_t wi, uint32_t co, uint32_t kh, uint32_t kw,
                        uint32_t ph, uint32_t pw, uint32_t sh, uint32_t sw) {
    const uint32_t ho = (hi + 2 * ph - kh) / sh + 1;
    const uint32_t wo = (wi + 2 * pw - kw) / sw + 1;
    ConvArgs args = {in, w, b, out, n, ci, hi, wi, co, ho, wo,
                     kh, kw, ph, pw, sh, sw, b != 0};
    void* params[1] = {&args};
    VX_CHECK(hipModuleLaunchKernel(f, ho, co, n, 16, 1, 1,
                                   ci * kh * kw * 4, nullptr, params, nullptr));
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
    const uint32_t sh = stride.size() > 0 ? (uint32_t)stride[0] : 1;
    const uint32_t sw = stride.size() > 1 ? (uint32_t)stride[1] : sh;
    const uint32_t ph = padding.size() > 0 ? (uint32_t)padding[0] : 0;
    const uint32_t pw = padding.size() > 1 ? (uint32_t)padding[1] : ph;

    const auto& is = input.sizes();
    const auto& ws = weight.sizes();
    const uint32_t co = (uint32_t)ws[0], ci = (uint32_t)ws[1];
    const uint32_t kh = (uint32_t)ws[2], kw = (uint32_t)ws[3];
    TORCH_CHECK((uint32_t)is[1] == ci, "torch_vortex: conv channel mismatch");
    const uint32_t ho = ((uint32_t)is[2] + 2 * ph - kh) / sh + 1;
    const uint32_t wo = ((uint32_t)is[3] + 2 * pw - kw) / sw + 1;

    auto out = torch::empty({is[0], (int64_t)co, (int64_t)ho, (int64_t)wo},
                            input.options());
    uint64_t baddr = 0;
    // Composite layers pass None as an *undefined* tensor inside the
    // optional (still has_value) — treat undefined as absent.
    if (bias.has_value() && bias->defined()) {
        check_cnn_f32(*bias, "conv.bias", 1);
        TORCH_CHECK(bias->numel() == co, "torch_vortex: bias size mismatch");
        baddr = (uint64_t)(uintptr_t)bias->data_ptr();
    }
    launch_conv(g_conv2d, (uint64_t)(uintptr_t)input.data_ptr(),
                (uint64_t)(uintptr_t)weight.data_ptr(), baddr,
                (uint64_t)(uintptr_t)out.data_ptr(), (uint32_t)is[0], ci,
                (uint32_t)is[2], (uint32_t)is[3], co, kh, kw, ph, pw, sh, sw);
    return out;
}

static torch::Tensor pool_impl(const torch::Tensor& self,
                               c10::IntArrayRef kernel_size,
                               c10::IntArrayRef stride,
                               c10::IntArrayRef padding, uint32_t op) {
    check_cnn_f32(self, "pool.input", 4);
    const uint32_t kh = (uint32_t)kernel_size[0], kw = (uint32_t)kernel_size[1];
    const uint32_t sh = stride.size() > 0 ? (uint32_t)stride[0] : kh;
    const uint32_t sw = stride.size() > 1 ? (uint32_t)stride[1] : sh;
    const uint32_t ph = padding.size() > 0 ? (uint32_t)padding[0] : 0;
    const uint32_t pw = padding.size() > 1 ? (uint32_t)padding[1] : ph;
    const auto& s = self.sizes();
    const uint32_t ho = ((uint32_t)s[2] + 2 * ph - kh) / sh + 1;
    const uint32_t wo = ((uint32_t)s[3] + 2 * pw - kw) / sw + 1;
    auto out = torch::empty({s[0], s[1], (int64_t)ho, (int64_t)wo}, self.options());
    PoolArgs args = {(uint64_t)(uintptr_t)self.data_ptr(),
                     (uint64_t)(uintptr_t)out.data_ptr(),
                     (uint32_t)s[0], (uint32_t)s[1], (uint32_t)s[2], (uint32_t)s[3],
                     ho, wo, kh, kw, ph, pw, sh, sw, op};
    void* params[1] = {&args};
    VX_CHECK(hipModuleLaunchKernel(g_pool2d, ho, (uint32_t)s[1], (uint32_t)s[0],
                                   16, 1, 1, 0, nullptr, params, nullptr));
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
    TORCH_CHECK(!training, "torch_vortex: batch_norm training unsupported (inference only)");
    check_cnn_f32(input, "bn.input", 4);
    auto is_def = [](const std::optional<torch::Tensor>& t) {
        return t.has_value() && t->defined();
    };
    TORCH_CHECK(is_def(weight) && is_def(bias) && is_def(running_mean) &&
                    is_def(running_var),
                "torch_vortex: bn needs weight/bias/running stats (inference)");
    const auto c = input.size(1);
    TORCH_CHECK(weight->numel() == c && bias->numel() == c &&
                running_mean->numel() == c && running_var->numel() == c,
                "torch_vortex: bn parameter size mismatch");

    // rstd = 1/sqrt(var + eps) computed on host into a device scratch.
    std::vector<float> varv((size_t)c);
    VX_CHECK(hipMemcpy(varv.data(), running_var->data_ptr(), varv.size() * 4,
                       hipMemcpyDeviceToHost));
    std::vector<float> rstdv((size_t)c);
    for (size_t i = 0; i < varv.size(); ++i) rstdv[i] = 1.0f / std::sqrt(varv[i] + (float)eps);
    void* rstd_dev = nullptr;
    VX_CHECK(hipMalloc(&rstd_dev, varv.size() * 4));
    VX_CHECK(hipMemcpy(rstd_dev, rstdv.data(), varv.size() * 4, hipMemcpyHostToDevice));

    auto out = torch::empty_like(input);
    BnArgs args = {(uint64_t)(uintptr_t)input.data_ptr(),
                   (uint64_t)(uintptr_t)running_mean->data_ptr(),
                   (uint64_t)(uintptr_t)rstd_dev,
                   (uint64_t)(uintptr_t)weight->data_ptr(),
                   (uint64_t)(uintptr_t)bias->data_ptr(),
                   (uint64_t)(uintptr_t)out.data_ptr(),
                   (uint32_t)input.numel(), (uint32_t)c};
    void* params[1] = {&args};
    uint32_t total = (uint32_t)input.numel();
    VX_CHECK(hipModuleLaunchKernel(g_bn, (total + 3) / 4, 1, 1, 4, 1, 1, 0,
                                   nullptr, params, nullptr));
    hipFree(rstd_dev);
    auto empty_aux = torch::empty({0}, input.options());
    return std::make_tuple(out, empty_aux, empty_aux);
}

static torch::Tensor mm_impl(const torch::Tensor& a, const torch::Tensor& b,
                             uint32_t transb) {
    check_cnn_f32(a, "mm.a", 2);
    check_cnn_f32(b, "mm.b", 2);
    TORCH_CHECK(a.size(1) == b.size(transb ? 1 : 0),
                "torch_vortex: mm shape mismatch");
    auto out = torch::empty({a.size(0), b.size(transb ? 0 : 1)}, a.options());
    MmArgs args = {(uint64_t)(uintptr_t)a.data_ptr(),
                   (uint64_t)(uintptr_t)b.data_ptr(),
                   (uint64_t)(uintptr_t)out.data_ptr(),
                   (uint32_t)a.size(0), (uint32_t)out.size(1), (uint32_t)a.size(1),
                   transb};
    void* params[1] = {&args};
    VX_CHECK(hipModuleLaunchKernel(g_mm, (uint32_t)((out.size(1) + 15) / 16),
                                   (uint32_t)((a.size(0) + 15) / 16), 1, 16, 1,
                                   1, 1024, nullptr, params, nullptr));
    return out;
}

static torch::Tensor mm_impl_wrap(const torch::Tensor& a,
                                  const torch::Tensor& b) {
    return mm_impl(a, b, 0);
}

static torch::Tensor linear_impl(
    const torch::Tensor& input, const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias) {
    check_cnn_f32(input, "linear.input", 2);
    check_cnn_f32(weight, "linear.weight", 2);
    auto out = mm_impl(input, weight, 1);  // x @ w^T
    if (bias.has_value() && bias->defined()) {
        check_cnn_f32(*bias, "linear.bias", 1);
        TORCH_CHECK(bias->numel() == out.size(1), "torch_vortex: bias size mismatch");
        BiasArgs args = {(uint64_t)(uintptr_t)out.data_ptr(),
                         (uint64_t)(uintptr_t)bias->data_ptr(),
                         (uint32_t)out.size(0), (uint32_t)out.size(1)};
        void* params[1] = {&args};
        uint32_t total = (uint32_t)out.numel();
        VX_CHECK(hipModuleLaunchKernel(g_bias_add, (total + 3) / 4, 1, 1, 4, 1,
                                       1, 0, nullptr, params, nullptr));
    }
    return out;
}

static torch::Tensor addmm_impl(const torch::Tensor& self,
                                const torch::Tensor& mat1,
                                const torch::Tensor& mat2,
                                const c10::Scalar& beta, const c10::Scalar& alpha) {
    TORCH_CHECK(beta.to<double>() == 1.0 && alpha.to<double>() == 1.0,
                "torch_vortex: addmm beta/alpha != 1 unsupported in v1");
    TORCH_CHECK(self.dim() == 1 && self.size(0) == mat2.size(1),
                "torch_vortex: addmm self must be the (n,) bias vector in v1");
    return linear_impl(mat1, mat2, self);
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

// Autograd pass-through for every registered op (inference-only v1: the
// composite autograd kernels would need cross-device sync, so gradient
// support is explicitly out of scope and the key just falls through).
// Namespace-wide fallbacks must use the catch-all TORCH_LIBRARY_IMPL(_, ...).
TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, BackendSelect, m) {
    m.impl("empty.memory_format", &bs_empty_impl);
    m.impl("empty_strided", &bs_empty_strided_impl);
}

// ---------------------------------------------------------------------------
// Module init
// ---------------------------------------------------------------------------

void load_ops(const std::string& vxbin_path, const std::string& dnn_path) {
    // One combined image (ops + dnn kernels): the simx backend loads
    // modules at a fixed base address, so separate images would overlap.
    // `dnn_path` is accepted for call compatibility but ignored.
    (void)dnn_path;
    c10::SetAllocator(c10::DeviceType::PrivateUse1, &g_vortex_allocator);
    VX_CHECK(hipInit(0));
    VX_CHECK(hipModuleLoad(&g_ops_module, vxbin_path.c_str()));
    VX_CHECK(hipModuleGetFunction(&g_add, g_ops_module, "add_kernel"));
    VX_CHECK(hipModuleGetFunction(&g_mul, g_ops_module, "mul_kernel"));
    VX_CHECK(hipModuleGetFunction(&g_fill, g_ops_module, "fill_kernel"));
    VX_CHECK(hipModuleGetFunction(&g_relu, g_ops_module, "relu_kernel"));
    VX_CHECK(hipModuleGetFunction(&g_conv2d, g_ops_module, "tv_conv2d_kernel"));
    VX_CHECK(hipModuleGetFunction(&g_pool2d, g_ops_module, "tv_pool2d_kernel"));
    VX_CHECK(hipModuleGetFunction(&g_bn, g_ops_module, "tv_bn_affine_kernel"));
    VX_CHECK(hipModuleGetFunction(&g_mm, g_ops_module, "tv_mm_kernel"));
    VX_CHECK(hipModuleGetFunction(&g_bias_add, g_ops_module, "tv_bias_add_kernel"));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("load_ops", &load_ops, "initialize the vortex backend (allocator + ops/dnn images)");
}
