# P0.3 native HIP SimX↔rtlsim parity

## 目标

功能 case 通过只能说明每个后端各自能运行。这个节点增加第一条 native HIP parity 证据：同一个 module API workload、同一个 RV32 配置，必须在 SimX 和 rtlsim 上退休相同数量的指令，周期差控制在统一 CI 容差内。

## 实现

- `tests/hip_native/module_api/main.cpp` 在 kernel/event 完成并校验输出后，通过 `hipGetVxDevice` 调用公共 `vx_device_dump_perf`，输出标准 `PERF: instrs=…, cycles=…, IPC=…` 摘要。
- `ci/testcase.py` 的 `make-run` 执行器现在使用 parity runner 传入的 driver 覆盖目标中的 `{driver}`。此前一个 `check: model_parity` 的 make-run case 会把 SimX leg 错误地再次执行成 rtlsim。
- `ci/testcases/hip_native.yaml` 新增 `parity_module_api`，固定 RV32、`tolerance: 0.05`；case 本身 pinned 到 rtlsim，runner 自动执行 SimX 对照 leg。

## 验收条件

`ci/test_runner.py` 对两条 leg 的最后一条设备级 PERF 摘要做比较：

1. `instrs` 必须精确相等；不允许用周期容差掩盖功能/控制流差异；
2. `abs(rtlsim_cycles - simx_cycles) / rtlsim_cycles <= 5%`；
3. 任一 leg 缺少 PERF 摘要或 make target 失败都直接失败。

验证命令（在 `build_dl32`，先重新 `configure`）：

```bash
../configure --xlen=32 --tooldir=/data/vortex-tools
env -u DEBUG CCACHE_DISABLE=1 make -s -C sw/runtime stub
env -u DEBUG CCACHE_DISABLE=1 make -s -C sw/runtime rtlsim
VX_XLEN=32 pytest ci -m 'hip_native and model_parity' --strict-markers
```

## 当前边界

这是一条 module API parity smoke，不代表 reduction/GEMM/stream overlap/A-extension、RV64 或 XRT 已建立 parity。A-extension atomic 的 rtlsim leg 仍未登记；后续每增加一个 parity case，都要记录 workload、配置、周期容差和两边的 PERF 原文。
