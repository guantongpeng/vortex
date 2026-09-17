# P5.1 torch-vortex:PyTorch PrivateUse1 后端

> **历史文档(2026-09-14, torch 2.4.1 / Python 3.8)。**
> 当前状态见 [p5_03_m0_m1_baseline.md](p5_03_m0_m1_baseline.md):后端已迁移到
> torch 2.14.0+cpu / Python 3.10,算子语义按 `pytorch_plan.md` 的 W1 重修。
> 下文第 4 节的"torch 2.4 踩坑记录"是当时的实测记录,**其中第 7 条在 2.14 上
> 已经反过来**(`rename_privateuse1_backend` 单独不再生成 `.vortex()`),照它
> 推断会踩坑;现行结论在 p5_03 第 1 节。

## 目标

完成计划 P5-01:PyTorch PrivateUse1 注册为 `vortex` 设备 —— allocator、device guard、copy、基础 ATen op,数据真实驻留 Vortex 设备内存、算子真实运行在设备 kernel 上(经 SimX 验证)。

## 架构

```
torch eager(aten::add/copy_/fill_/zero_/empty)
   │ PrivateUse1 dispatch key("vortex" via rename_privateuse1_backend)
   ▼
torch_vortex(C++ extension,torch.utils.cpp_extension 在线编译)
   │ allocator: hipMalloc/hipFree
   │ copy_:     hipMemcpy(H2D/D2H/D2D)
   │ add/mul/fill_/zero_: hipModuleLaunchKernel
   ▼
libhip_vortex → vortex2.h → simx/rtlsim/xrt
```

- [`torch-vortex/src/vortex_ext.cpp`](../../torch-vortex/src/vortex_ext.cpp):`c10::SetAllocator(PrivateUse1)`(hipMalloc 挂底 `vx_buffer_create`)+ `C10_REGISTER_GUARD_IMPL`(单设备、单隐式流)+ `empty/empty_strided`(PrivateUse1 与 **BackendSelect** 双注册,非 vortex 设备链回 `empty_cpu`)+ `copy_/_copy_from`(全方向)+ `add.Tensor/mul.Tensor/fill_.Scalar/zero_`(launch `kernels/torch_ops.vxbin` 中的 KMU kernel)。**v1 显式约束(TORCH_CHECK,不静默回退):FP32、contiguous、add alpha=1**。
- [`torch-vortex/kernels/ops.hip`](../../torch-vortex/kernels/ops.hip):add/mul/fill 的 KMU kernel(hipcc-vortex `--kernel-lib=vortex2` + VXKMDATA 元数据)。
- [`torch-vortex/torch_vortex/__init__.py`](../../torch-vortex/torch_vortex/__init__.py):rename → 在线编译扩展 → `load_ops`(allocator+镜像)→ `torch._register_device_module("vortex")`(device_count/synchronize 经 ctypes);仓库/构建树自定位(向上搜索标志文件,不信任被污染的 `VORTEX_HOME` 环境)。
- [`torch-vortex/tests/test_basic.py`](../../torch-vortex/tests/test_basic.py):7 项 —— 注册、分配、往返、fill、add、mul、非 F32 显式报错。

## torch 2.4 踩坑记录(全部实测修复)

1. `aten::empty_strided` **没有 `.memory_format` 变体**(与 empty 不同),注册名写错会静默/崩溃。
2. factory 经 **BackendSelect** 键分发:必须双注册(PrivateUse1 + BackendSelect),后者按 device 参数路由、非本设备的调用链回 `at::detail::empty_cpu`。
3. `Tensor.to()` 走 `aten::_copy_from`(返回 Tensor,canonical 签名 `dst` 为 const&)而非 `copy_`;两者都要注册。
4. `copy_`/`_copy_from`/`fill_` 等 mutable op **返回 Tensor**;kernel 签名与 schema 不匹配会在 load 时 `Mismatch in kernel C++ signatures` 崩溃。
5. `DeviceGuardImplInterface` 必须实现全部纯虚:`type/exchangeDevice/getDevice/setDevice/uncheckedSetDevice/getStream/exchangeStream/deviceCount`。
6. `c10::Allocator::copy_data` 在 2.4 是纯虚(必须有);`Device` 有效性用 `index() >= 0`(无 isValid)。
7. 2.4 无 `generate_tensor_methods_for_privateuse1_backend`/`.is_vortex` 属性(2.5+);`t.device.type` 报告为 rename 后的 `"vortex"` 字符串。

## 验证记录(2026-09-14,SimX 后端,rv64)

```
$ python3 -m pytest tests/test_basic.py -q
7 passed in 17.43s
```

冒烟(显式数值):65 元素随机 `add`/`mul` 与 CPU 参考 rtol=1e-6 全对;`torch.full(device='vortex')` 往返一致;`vortex:0` 设备名正确。

## 约束与后续

1. v1 op 集刻意最小;Tier A 扩展(view/reshape 类不需要 kernel,后续按模型 trace 统计补)。
2. 单隐式流;`torch.cuda.Stream` 风格 API 未暴露(libhip_vortex 已具备,包装属下一节点)。
3. autograd:注册键集含 AutogradPrivateUse1,backward 走 CompositeImplicitAutograd(CPU)的显式同步策略未实现 —— 训练不可用,推理 eager 路径成立。
4. P5-02(ResNet eager)需要 conv/bn/pool/linear 算子集 —— 依赖 P3 第二层算子;届时 `torch.compile`/export 才有图可落。
5. 扩展在线编译产物在 `~/.cache/torch_extensions`;CI 中应改为 setup.py 预编译。
