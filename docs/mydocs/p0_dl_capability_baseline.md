# P0.2 DL capability baseline 开发记录

## 目标

把运行时实际公布的 Vortex 能力导出为机器可读 JSON，作为 DL kernel、HIP runtime、Triton autotune 和 PyTorch device 初始化的输入。这个节点只读取现有 `vortex2.h` 能力，不把“配置文件声明”冒充成“设备实际能力”。TCU dtype bitmap、并发 queue 数等扩展能力会在对应 runtime contract 节点实现后再增加字段。

## 实现

- [`tests/runtime/dl_capabilities.cpp`](../../tests/runtime/dl_capabilities.cpp) 枚举现有 `VX_CAPS_*`，调用 `vx_device_count/open/query/release`，输出稳定的 JSON。
- [`tests/runtime/Makefile`](../../tests/runtime/Makefile) 将工具纳入 runtime 测试集合，因此 SimX/XRT 运行入口都会执行它。
- 输出字段直接来自 `vortex2.h` 和 backend `vx_device_query`，没有读取 `VX_config.toml` 或硬编码设备参数。

## 预期输出

```json
{
  "device_count": 1,
  "capabilities": {
    "num_threads": 4,
    "num_warps": 4,
    "num_cores": 1,
    "isa_flags": 0
  }
}
```

实际数值随 configure 和 backend 改变；CI 应保存完整输出作为构建产物，而不是把某个机器的数值写死为 golden。

## 验证命令

```bash
cd build_dl64
../configure --xlen=64 --tooldir=/data/vortex-tools
make -C tests/runtime dl_capabilities
VORTEX_DRIVER=simx LD_LIBRARY_PATH=$PWD/sw/runtime \
  tests/runtime/dl_capabilities > dl_capabilities.json
python3 -m json.tool dl_capabilities.json
```

成功条件是 JSON 可解析、所有现有 capability 查询成功、设备句柄正常释放。工具遇到没有设备或查询错误时返回非零并输出错误原因。

## 当前验证结果与阻断

2026-09-11（晚）：网络恢复后重跑 Ramulator FetchContent（yaml-cpp、spdlog 等全部成功），`make -C sim simx` 与 `make -C sw/runtime simx` 完成，SimX 后端首次可运行。`make -C tests/runtime run-simx` 全套通过（exit 0），包括 test_basic、test_async（8 个子用例）、test_async_free、test_timeline_events、test_module_kernel（含真实 `kernel.vxbin` 的 load/launch/refcount）和 dl_capabilities。

dl_capabilities 在 build_dl64（1 core/4 warps/4 threads 默认档）的实际输出：

```json
{
  "device_count": 1,
  "capabilities": {
    "num_threads": 4,
    "num_warps": 4,
    "num_cores": 1,
    "global_mem_size": 8589934592,
    "local_mem_size": 16384,
    "isa_flags": 221200257320,
    "cp_num_queues": 1,
    "cp_ring_size_log2": 16,
    "cp_axi_tid_width": 6,
    "cp_supports_draw": 1,
    "cp_supports_qmd": 1,
    "cp_mmu_fault_report": 1
  }
}
```

（完整字段还包括 cache_line_size、num_mem_banks、mem_bank_size、num_clusters、socket_size、issue_width、clock_rate_mhz=400、peak_mem_bw_mb_s=460000、vm_support=0。）

## P0.2 完成判定

SimX 运行验证已补做并通过：JSON 可解析、所有 capability 查询成功、设备句柄正常释放。rtlsim parity 与 FPGA 档的运行仍属后续节点（dl_rtl/dl_fpga profile），不在本节点范围内。
