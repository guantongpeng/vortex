# P1.2 HIP function metadata 映射

## 目标

让 native HIP 的 autotune/loader 可以在 launch 前读取 kernel 资源信息，而不必重新解析 VXKMDATA 或访问 runtime 私有句柄。本节点先实现稳定且能由当前 `vx_kernel_info_t` 支持的 `hipFuncGetAttribute` 子集。

## 接口和映射

`sw/hip/include/hip/hip_runtime_api.h` 新增与 HIP ABI 同值的 `hipFunction_attribute` 枚举和 `hipFuncGetAttribute` 声明。`libhip_vortex` 通过 `vx_kernel_get_info` 和设备 capability 查询提供：

| HIP attribute | Vortex 来源 | 语义 |
|---|---|---|
| `hipFuncAttributeMaxThreadsPerBlock` | `VX_CAPS_NUM_THREADS × VX_CAPS_NUM_WARPS` | 单 CTA 的设备上限 |
| `hipFuncAttributeSharedSizeBytes` | `static_lmem_bytes` | metadata 声明的静态 local/shared 使用量 |
| `hipFuncAttributeMaxDynamicSharedSizeBytes` | `VX_CAPS_LOCAL_MEM_SIZE - static_lmem_bytes` | 当前设备可提供的动态余量 |
| `hipFuncAttributeNumRegs` | `registers` | metadata 声明的寄存器数 |

`hipFuncAttributeLocalSizeBytes`、PTX/binary version、cache mode、constant memory 和 shared-memory carveout 当前没有可靠的 VXKMDATA 来源，明确返回 `hipErrorNotSupported`。静态 local/shared 的命名差异不能用猜测字段掩盖。

## 验证

`tests/hip_native/module_api/main.cpp` 在获取 `vecadd_kernel` 后检查：

- max threads 为正数；
- 静态 shared 为 0，动态 shared 余量为正数；
- 未支持的 local-size attribute 返回 `hipErrorNotSupported`；
- 后续 module launch、stream/event 和数值校验仍然通过。

rv32/rv64 的 device capability 查询都经过同一接口，避免把 host pointer width 当作设备属性。

## 后续

- 为 compiler metadata 增加 constant/local memory、PTX/ISA 版本和 cache policy 的可验证字段；
- 把 max block/LMEM/ISA metadata 传给 Triton autotune，拒绝无法满足设备资源的候选配置；
- 在 PyTorch loader 中缓存 module/function attributes，并在 shape/dtype dispatch 前做 capability gate；
- 补 `hipFuncGetAttributes`（复数接口）和 occupancy API 的支持范围决策，unsupported 路径保持显式错误。
