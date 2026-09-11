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

2026-09-11：已执行 RV32/RV64 configure，并初始化仓库固定 submodule。`sw/runtime/stub` 和 `tests/runtime/dl_capabilities` 在两个 XLEN 的默认配置下均编译成功。完整 `make -s` 在构建 Ramulator 2.0 时需要从 GitHub 获取 `yaml-cpp`，当前环境网络不可达，因此尚未能构建 SimX backend 或运行该工具。失败位置为 `third_party/ramulator/CMakeLists.txt:40 FetchContent_MakeAvailable(yaml-cpp)`；这不是 capability 工具的编译失败。待 `yaml-cpp` 可用或预构建 Ramulator 安装后，必须按上面的命令补跑，并把 JSON 和 SimX/rtlsim/HIP 基线结果加入本记录。

## P0.2 完成判定

代码和 Makefile 已完成，RV32/RV64 host 编译通过；节点在 SimX/rtlsim 上的完成状态保持“待依赖解除后验证”，不提前标记为通过。下一节点不能依赖一个未实际运行的 capability 字段。
