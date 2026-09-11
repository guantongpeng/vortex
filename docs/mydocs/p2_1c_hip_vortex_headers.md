# P2.1c HIP headers 与 vxbin 直编直跑

## 目标

完成 P2-01 的验收：真实 `__global__` kernel 经 `hipcc-vortex` 编译为 rv32/rv64 对象、链接为 `.vxbin`，并在 SimX 上运行通过（vecadd 直编直跑）。这是原生 HIPVortex 路径上第一个可执行的 HIP 程序。

## 关键发现

1. **VOLT clang 20.1.8 内置 HIP 模式，但 `-x hip` 对 RISC-V 目标不可用**：clang driver 在 HIP 模式下注入 `-amdgpu-internalize-symbols` 等 AMDGPU 专用 LLVM 选项，RISC-V 后端解析即失败；`.hip` 扩展名还会自动触发该模式。因此 `hipcc-vortex` 强制 `-x c++`，HIP 表面由 `third_party/hip-vortex` 头文件映射，保持 VOLT 无补丁。
2. **`annotate("vortex.kernel")` 已被 VOLT 后端支持**：带该注解的函数自动生成 `__vx_kentry_<name>` 别名，`vxbin.py` 将其写入 VXSYMTAB，运行时 `vx_module_get_kernel` 可按名解析。`__global__` 宏因此定义为 `extern "C"` + 该注解，kernel 名保持非 mangled。
3. **设备上不能用 picolibc 的 `printf`**：其 stdio 惰性分配缓冲会走 `_sbrk`，而设备 stub 的 `_sbrk` 执行 `ebreak`（`sw/kernel/src/vx_syscalls.c`），程序直接挂起且无任何输出。头文件将 `printf` 重定向到 tinyprintf 的 `vx_printf`（MMIO、零分配）。
4. **探针 sysroot 布局修订**（同日提交）：真实工具链的 sysroot 位于 `riscv{N}-gnu-toolchain/riscv{N}-unknown-elf`，`libc{N}`（picolibc）与 `libcrt{N}`（compiler-rt baremetal）是链接必需输入；探针 schema 2 输出已解析路径。

## 交付物

- [`third_party/hip-vortex/include/hip/hip_runtime.h`](../../third_party/hip-vortex/include/hip/hip_runtime.h)：执行空间宏、`warpSize`（CSR 查询）、错误枚举、hostless `hipMalloc/hipFree/hipMemcpy/hipMemset/hipDeviceSynchronize`、`hipLaunchKernelGGL`（ArgPack 递归参数包 + 每签名 trampoline → `vx_spawn_threads`）、warp intrinsics（`__shfl_*`/`__ballot`/`__all`/`__any`/`__syncwarp`）、atomics（`atomicAdd/Max/Exch/CAS`，float 为 CAS 循环）、数学近似入口、`hipGetDeviceCount`。
- [`third_party/hip-vortex/include/hip/hip_vector_types.h`](../../third_party/hip-vortex/include/hip/hip_vector_types.h)：`dim3`（镜像 `dim3_t` 布局，union 不能做基类）与 HIP 向量类型。
- [`ci/hipcc_vortex.py`](../../ci/hipcc_vortex.py)：
  - 编译模式：`--target/--sysroot/--gcc-toolchain` 来自探针 `paths`、`-march/-mabi`（rv32: `rv32imaf/ilp32f`，rv64: `rv64imafd/lp64d`）、`-mcmodel=medany`、`-x c++`、hip-vortex 与 kernel include、`gen_config.py` 展开的 `-DVX_CFG_*`；
  - 链接模式（输出 `.vxbin` 或输入含 `.o`）：复刻 `tests/kernel/common.mk` 的两遍 `vx_start.S` 编译（`kernel_startup.sh` 检测 NEED_GP/TLS/INITFINI）、`link<N>.ld` + `libvortex.a` + picolibc + compiler-rt 静态链接、`vxbin.py` 转换；
  - `--build-dir` 指向已 configure 的构建目录（`VX_types.h` 与 `libvortex.a` 来源）。
- [`tests/hip_native/`](../../tests/hip_native/)：`vecadd.hip` + Makefile（`run-simx`/`run-rtlsim` 入口），`tests/Makefile` 增加 `hip_native` 目标（不进默认 `all`，因为依赖 TOOLDIR 输入，由 `ci/hip_native_probe.py --strict` 把关）。

## 参数传递设计

`hipLaunchKernelGGL(k, grid, block, shmem, stream, args...)` 在设备上展开为：

```
ArgPack<Args...> 递归参数包（不依赖 libstdc++ tuple）
        ↓
LaunchCtx{fn, args} 栈上上下文
        ↓
vx_spawn_threads(3, grid, block, launch_trampoline<Fn,Args...>, &ctx)
        ↓
每个线程: trampoline 解包 → k(a0, a1, ...)
```

`PackGet<I,...>` 采用继承递归解析（包装式实现的返回类型会错位到子包首元素，已修复并记录）。`__global__` 函数的 kentry 别名让同一镜像同时兼容后续 `libhip_vortex` 的按名 launch（P2-03）。

## 验证记录（2026-09-11）

rv64（build_dl64，默认 1 core/4 warps/4 threads 档）：

```bash
make -C tests/hip_native run-simx
# Running vecadd.vxbin...
# #0: hip_native vecadd: warpSize=4
# #0: PASSED
```

rv32（build_dl32，同一默认档）：

```bash
make -C tests/hip_native run-simx
# Running vecadd.vxbin...
# #0: hip_native vecadd: warpSize=4
# #0: PASSED
```

单元测试与语法检查：

```bash
python3 -m unittest discover -s ci -p 'test_hipcc_vortex.py'    # 6 tests OK
python3 -m unittest discover -s ci -p 'test_hip_native_probe.py' # 4 tests OK
python3 -m py_compile ci/hipcc_vortex.py ci/hip_native_probe.py
```

对象级符号验证：`llvm-nm` 显示 `__vx_kentry_vecadd_kernel`（非 mangled）。

## 限制与后续

1. rtlsim 运行未在本节点执行（verilator 构建耗时），`run-rtlsim` 入口已就绪，应在下个节点与 parity 基线一起补做。
2. 三尖括号语法、静态 `__shared__`、stream/event 语义见 README 的“Not supported”清单；对应能力分别在 clang driver 改造、KMU launch 模型和 `libhip_vortex`（P2-02/P2-03）节点交付。
3. `hip::half`/bfloat16 向量类型尚未加入；P3 的 GEMM 节点需要时一并定义，避免在无 TCU dtype 验证的情况下提前承诺 ABI。
4. warp intrinsics/atomics 只有编译级覆盖（vecadd 未执行它们）；P2-03 的 conformance smoke 会为每个 intrinsic 增加运行验证。
