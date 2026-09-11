# P1.2 CP capability contract 开发记录

## 目标

让 HIP stream/graph、Triton driver 和 PyTorch runtime 读取命令处理器的真实能力。此前 `vortex2.h` 只公开 GPU 拓扑和 ISA；CP 的 queue/ring/QMD/DRAW 信息只在 `Device` 内部使用，上层只能猜测。

## 实现

- [`sw/runtime/include/vortex2.h`](../../sw/runtime/include/vortex2.h) 新增六个 `VX_CAPS_CP_*` ID。
- [`sw/runtime/common/device.cpp`](../../sw/runtime/common/device.cpp) 保存 `CP_DEV_CAPS` 原始值并在 `vx_device_query` 解码。
- [`sw/runtime/common/vortex2_internal.h`](../../sw/runtime/common/vortex2_internal.h) 增加每设备缓存字段。
- [`tests/runtime/dl_capabilities.cpp`](../../tests/runtime/dl_capabilities.cpp) 输出新增字段。
- [`tests/runtime/test_basic.cpp`](../../tests/runtime/test_basic.cpp) 验证 queue 数和 ring size 不为零，防止 ABI 回归。

字段来源是 CP register 0x008：低 8 位 queue 数、[15:8] ring size log2、[23:16] AXI TID 宽度、bit25 DRAW、bit26 QMD、bit27 MMU fault report。全 1 的异常读值按已有逻辑转换为零并保守关闭可选能力。

## 验证

```bash
cd build_dl64
../configure --xlen=64 --tooldir=/data/vortex-tools
make -C sw/runtime/stub
make -C tests/runtime test_basic dl_capabilities
```

后端可用时运行：

```bash
VORTEX_DRIVER=simx LD_LIBRARY_PATH=$PWD/sw/runtime tests/runtime/test_basic
VORTEX_DRIVER=simx LD_LIBRARY_PATH=$PWD/sw/runtime \
  tests/runtime/dl_capabilities | python3 -m json.tool
```

当前环境仍因 Ramulator 构建所需的 `yaml-cpp` 无法下载而不能启动 SimX；本节点的 host 编译和符号链接验证应先通过，SimX/rtlsim 数值验证在依赖恢复后补做。CP capability 不应在上层以配置宏复制一份。

## 2026-09-11 补充验证

网络恢复后 SimX 后端构建成功，`make -C tests/runtime run-simx` 中 `test_basic` 的 CP capability 断言（queues 与 ring_log2 非零）在真实设备路径上通过。`dl_capabilities` 实际输出 `cp_num_queues=1`、`cp_ring_size_log2=16`、`cp_axi_tid_width=6`、DRAW/QMD/MMU-fault-report 均为 1，与 CP register 0x008 的解码一致。上层（HIP stream、Triton driver）应以这些查询值为准；单队列是当前默认配置的事实，不代表 ABI 上限。
