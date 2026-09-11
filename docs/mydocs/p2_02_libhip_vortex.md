# P2.2 libhip_vortex：HIP 主机 API 子集

## 目标

建立 HIP 主机运行时到 `vortex2.h` 的薄适配层（计划 P2-02），覆盖 device/context、memory、stream、event、memcpy/memset、module/launch 六个模块的最小子集，并用一个端到端测试验证：主机程序通过 HIP API 加载 P2-01 工具链产出的 KMU kernel 镜像并运行 vecadd。

## 交付物

- [`sw/hip/include/hip/hip_runtime_api.h`](../../sw/hip/include/hip/hip_runtime_api.h)：主机侧 C ABI。错误码（CUDA 对齐编号）、`hipDeviceProp_t`（warpSize/mem/lmem 来自 capability 查询）、`HIP_LAUNCH_PARAM_*` 常量。
- [`sw/hip/src/hip_vortex.cpp`](../../sw/hip/src/hip_vortex.cpp)：实现，直接链接 `libvortex.so`，无私有后端路径：
  - **device/context**：单设备 primary context（`hipInit/hipSetDevice` 惰性 `vx_device_open`；`hipDeviceReset` 全量释放）；`hipGetDeviceProperties` 从 `VX_CAPS_*` 填充。
  - **memory**：`hipMalloc` → `vx_buffer_create` + `vx_buffer_address`，返回设备指针值；地址→handle 表由互斥锁保护。`hipHostMalloc` → `vx_buffer_map`。`hipMemcpy` 按方向走 `vx_enqueue_write/read/copy` 并等待完成；`hipMemset` → `vx_enqueue_fill_buffer`。
  - **stream/event**：`hipStream_t` 包 `vx_queue_h`（全部带 `VX_QUEUE_PROFILING_ENABLE`，否则 `hipEventElapsedTime` 无时间戳）；`hipEventRecord` → `vx_enqueue_signal`（异步、时间线计数递增），`hipEventSynchronize` → `vx_event_wait_value` 后取 profiling；`hipEventElapsedTime` 先同步再取两条 record 的 `end_ns` 差。
  - **module/launch**：`hipModuleLoad/GetFunction` → `vx_module_load_file` + `vx_module_get_kernel`（按名解析 `__vx_kentry_*`）；`hipModuleLaunchKernel` 组装 `vx_launch_info_t`（kernel + args blob + ndim/grid/block/lmem），kernelParams 路径的 args_size 取自镜像 VXKMDATA 元数据（P1.3 的首个消费者），extra 路径支持 `HIP_LAUNCH_PARAM_BUFFER_POINTER/SIZE`。
  - **错误模型**：每个 `vx_result_t` 经单一表转换，线程局部 last error；launch 按异步语义不在提交点等待。
- [`sw/hip/Makefile`](../../sw/hip/Makefile)：`libhip_vortex.so`（构建树内，链接 `-lvortex`，rpath 指向 runtime）；顶层 `Makefile.in` 增加 `hip:` 目标。
- 设备侧（`third_party/hip-vortex`）新增 **KMU 模式**：`hipcc-vortex --kernel-lib=vortex2` 为 TU 加 `-DHIP_VORTEX_KMU`、为 `vx_start.S` 加 `-DKMU_ENABLE`、链接 `libvortex2.a`。该模式下 `threadIdx/blockIdx/blockDim/gridDim` 来自 `vx_spawn2.h` 的 CSR，hostless 的 heap/launch 机制被编译排除，kernel 使用单指针参数块 ABI。
- [`tests/hip_native/module_api/`](../../tests/hip_native/module_api/)：`kernel.hip`（KMU vecadd）+ `main.cpp`（主机验收）+ Makefile（含 VXKMDATA sidecar 生成）。

## 参数 ABI 与 RV32/RV64 宽度

kernelParams 是“单指针参数块”约定：`kernelParams[0]` 指向参数结构体，大小优先取镜像元数据 `args_size`，缺失时要求调用者用 `extra` 显式给出（返回 `hipErrorInvalidValue`，不猜测）。

验收中发现并修复了计划 §5.1 预警的指针宽度问题：64 位主机把 `uintptr_t`（8 字节）指针写进参数块，rv32 设备按 4 字节读取，输出大面积错误（256 元素中 255 个 mismatch）。修复：测试主机侧以 `HIP_TEST_DEV_PTR_WIDTH` 用设备宽度整数打包，Makefile 的 `ARGS_SIZE` 相应为 rv64=32、rv32=20。这一结论对所有上层（Triton driver、PyTorch）都适用：**参数块必须按设备 XLEN 打包，主机 `size_t` 不能直接复制**。

## 验证记录（2026-09-11）

rv64（build_dl64）与 rv32（build_dl32），SimX 后端：

```
device=vortex sm=1 warpSize=4 clock=400000kHz mem=8192MB lmem=16384   (rv64)
device=vortex sm=1 warpSize=4 clock=400000kHz mem=4096MB lmem=16384   (rv32)
kernel elapsed: rc=hipSuccess ms=28.960 / 30.308
PASSED
```

覆盖：`hipInit/GetDeviceCount/GetDeviceProperties`（warpSize=4 断言，CSR 查询路径）、`hipMalloc×3/hipMemset/hipMemcpy` H2D+D2H、`hipModuleLoad/GetFunction("vecadd_kernel")/LaunchKernel`（kernelParams + 元数据 args_size）、`hipStreamCreate/hipEventRecord×2/hipEventSynchronize/hipEventElapsedTime/hipEventDestroy/hipStreamDestroy`、`hipModuleUnload/hipFree/hipDeviceReset`。256 元素数值全对。

完整 hip_native 套件（hostless vecadd + module_api）rv64/rv32 双档 `run-simx` 全部 PASSED。单元测试：hipcc 7 例、probe 4 例 OK。

## 限制与后续

1. **未实现**（按计划显式留给后续节点）：graphs、texture/surface、peer/multi-device、hiprtc、主机侧 `hipLaunchKernelGGL`（多参数 kernel 需要设备侧 trampoline 或编译器支持）、`hipMallocAsync`（可基于 P1.1 的 `vx_enqueue_free` 做，但 pool 策略未定）。
2. 单队列（`VX_CAPS_CP_NUM_QUEUES=1` 默认档）：多个 `hipStream_t` 语义正确但不会重叠；并发性能要等 CP 多队列档。
3. `hipEventElapsedTime` 的精度依赖后端 profiling 时钟（SimX 为仿真时间，单位 ms 正确但数值无性能含义）。
4. module 卸载/重载与 kernel refcount 的压力测试、`hipMemcpyDefault` 的混合 host-pinned 场景尚未单测。
5. rtlsim 后端的运行（`run-rtlsim` 入口已就绪）留待 P2-03 与 parity 基线一起做。
