# P2.1a 原生 HIPVortex 工具链输入探针

## 背景

仓库当前的 HIP 测试通过 chipStar→SPIR-V→PoCL 路径运行。P2.1 的 native HIP 路径需要独立的 clang driver、HIP headers/device bitcode、RISC-V sysroot、Vortex linker script 和 `vxbin.py`。如果缺少其中任一项，继续修改 kernel 或 runtime 只会得到误导性的失败。

## 交付物

`ci/hip_native_probe.py` 检查这些输入并输出 schema=1 的 JSON。它不下载依赖，也不把 chipStar 当作 native HIP 的替代品。HIP headers 通过 `HIP_VORTEX_INCLUDE` 指定，必须包含 `hip/hip_runtime.h`；LLVM 工具预期位于 `$TOOLDIR/llvm-vortex/bin`，sysroot 位于 `$TOOLDIR/riscv{32,64}-unknown-elf`。

## 用法

```bash
python3 ci/hip_native_probe.py --tooldir /data/vortex-tools
python3 ci/hip_native_probe.py --tooldir /data/vortex-tools --json
python3 ci/hip_native_probe.py --tooldir /data/vortex-tools --strict
```

`--strict` 适合 CI gate；普通模式用于开发机诊断，即使缺项也返回 0。JSON 可作为后续 `hipcc-vortex` 配置阶段的输入，避免硬编码绝对路径。

## 验证

```bash
python3 -m unittest discover -s ci -p 'test_hip_native_probe.py'
python3 -m py_compile ci/hip_native_probe.py
```

当前节点只建立前置条件探测，不声称 native HIP 已可编译。下一节点应在独立目录（建议 `third_party/hip-vortex`）加入可 upstream 的 clang driver wrapper、HIP include 版本锁定和 device bitcode 构建；成功标准是 `hipcc-vortex --version`、`--print-resource-dir` 和一个 `__global__` 编译命令均能复现并记录完整命令行。
