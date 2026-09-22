# P0.3 native HIP functional CI 收口

## 目标

把 `tests/hip_native` 的可执行入口登记到统一 testcase 目录，并让每个 case 的后端标签与实际验证证据一致。这个节点只收口功能测试矩阵；SimX↔rtlsim 的 retired-instruction parity、A-extension 的 RTL 基线和 XRT 集成仍由后续节点负责。

## 当前矩阵

| case | SimX | rtlsim | 配置/说明 |
|---|---:|---:|---|
| `vecadd` | ✓ | ✓ | native HIP 编译、module load、launch、copy |
| `module_api` | ✓ | ✓ | module/function API 和 KMU kernel |
| `reduction` | ✓ | ✓ | warp shuffle reduction |
| `sgemm` | ✓ | ✓ | FP32 FPU GEMM |
| `stream_overlap` | ✓ | ✓ | 双 stream、event wait、异步 copy |
| `atomics` | ✓ | — | `-DVX_CFG_EXT_A_ENABLE`；A-extension rtlsim 尚无独立基线 |
| `rtc_stub` | ✓ | — | 只有明确的 unsupported stub 断言；`run-rtlsim` 实际是 SimX 别名 |
| `bigargs` | ✓ | ✓ | RV64 4816 B / RV32 4812 B 参数块必须在入队前返回 `hipErrorInvalidValue`；rtlsim 只验证主机侧拒绝路径 |

`ci/testcases/hip_native.yaml` 中的 `drivers` 只列出已经具有真实对应入口和证据的后端。特别是，不能因为一个 Makefile 提供了同名目标，就把一个 SimX 别名标成 rtlsim 覆盖。`bigargs` 的 rtlsim 单元不执行 kernel，而是确认 rtlsim runtime 初始化后仍在提交前返回相同 HIP 错误。

## `bigargs` 的语义

运行时参数上限是公开 ABI 常量 `VX_KERNEL_ARGS_MAX_BYTES`（4096 字节）。`bigargs` 保留一个 4816 字节的 VXKMDATA 镜像，但现在把超过上限视为负向契约测试：

1. 通过 `hipModuleLaunchKernel` 提交带 metadata 的参数块；
2. 断言立即返回 `hipErrorInvalidValue`；
3. 再用显式 HIP buffer/size 参数提交 4097 B，断言同样立即返回 `hipErrorInvalidValue`；
4. 验证输出 buffer 没有被修改，再释放 module、device allocation 和 runtime。

HIP native 的每个 Makefile 都把 `.hip`/`main.cpp` 依赖指向源码树。`configure` 对 `tests/*` 只复制 Makefile，因此从空 build 目录运行时不会依赖手工复制 kernel 源文件；`stream_overlap` 同时用 host `static_assert` 检查 RV32/RV64 参数大小。

这使 HIP 层行为与 runtime 的 `vx_enqueue_launch`/批量命令校验保持一致，也避免测试继续依赖已经移除的“超大参数块暂存”路径。

## 验证命令

在已配置的 build 目录执行：

```bash
python3 ci/testcase.py lint
python3 ci/testcase.py matrix --drivers=simx,rtlsim --tier=smoke,full
make -C sw/runtime stub
make -C sw/runtime rtlsim
make -C tests/hip_native/bigargs clean
make -C tests/hip_native/bigargs run-simx
make -C tests/hip_native/bigargs run-rtlsim
```

native HIP 编译依赖 `TOOLDIR` 中的 LLVM、RISC-V sysroot、picolibc 和 compiler-rt；缺少这些输入时先运行 `python3 ci/hip_native_probe.py --strict`，不要把工具链缺失误判为 kernel 或 runtime 回归。

本节点还修复了 `ci/testcases/dl.yaml` 中 7 个已有 `make-run` case 缺少 `target` 的 lint 错误；统一入口现在能成功展开 677 个 case。

## 未完成项

- 为 A-extension 建立独立的 rtlsim build/config，并把 `atomics` 加入该矩阵。
- 为至少一个 HIP native workload 增加 `check: model_parity`，输出 SimX/rtlsim 的 retired instructions 和周期。
- 继续补齐 XRT/AFU 入口；rtlsim 不能替代完整硬件集成验证。
- 增加 chipStar 同源对照时，保留 native HIP 与 chipStar 两条路径的独立标签和日志。
