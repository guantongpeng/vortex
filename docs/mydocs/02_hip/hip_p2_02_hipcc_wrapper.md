# P2.1b `hipcc-vortex` 设备编译命令

`ci/hipcc_vortex.py` 是 native HIPVortex driver 的第一块可执行代码。它把 `--offload-arch=vortex32|vortex64` 映射到 `riscv{32,64}-unknown-elf`、Vortex LLVM 的 `+xvortex` target feature 和当前 VOLT 的 `rv{32,64}imaf{,d}` ISA，并强制生成设备对象（`-c`）。

```bash
python3 ci/hipcc_vortex.py --tooldir /data/vortex-tools \
  --offload-arch=vortex64 --dry-run kernel.hip -o kernel.o
```

执行前会检查 P2.1a 探针中的 clang 和两个 RISC-V sysroot。`--dry-run` 输出可复制的完整命令，适合 AI agent 保存到构建日志；缺输入时返回 1 并指出缺少的键。当前阶段明确拒绝隐式链接，避免把 host `libstdc++` 或 chipStar SPIR-V runtime 错误地混入设备对象。

验证：

```bash
python3 -m unittest discover -s ci -p 'test_hipcc_vortex.py'
python3 -m py_compile ci/hipcc_vortex.py
```

下一步要接入 HIP headers/device bitcode 和 `vxbin.py` 链接后处理，并增加真实 `__global__` kernel 的 rv32/rv64 编译验收；在这些输入安装前不要把 wrapper 标记为 native HIP 可用。

**2026-09-11 更新**：该验收已在 P2.1c 完成，见 [`hip_p2_03_hip_headers.md`](hip_p2_03_hip_headers.md)。本文件的 wrapper 描述已被扩展：编译模式加入 `-mabi`/`-mcmodel=medany`/`-x c++`（`.hip` 扩展名会自动触发 clang 的 HIP 模式，其 AMDGPU 选项对 RISC-V 非法）与探针解析的 sysroot/gcc-toolchain 路径；新增链接模式（两遍 `vx_start.S` + `link<N>.ld` + picolibc/compiler-rt + `vxbin.py`）和 `--build-dir`/`--configs` 参数。
