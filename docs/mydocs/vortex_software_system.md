# Vortex 软件系统详解（Vortex 3.0）

> 本文基于本仓库（`VORTEX_VERSION=3.0`，git `v3.0-911-gbba80f8de`）的实际源码整理，
> 全面介绍 Vortex GPGPU 的软件系统：工具链、构建与配置系统、主机端运行时与后端驱动、
> 命令处理器（CP）、设备端内核运行库、二进制格式（vxbin）、图形软件栈、高层编程模型
> 接入（OpenCL/Vulkan/HIP）、模拟器以及测试与 CI 体系。
>
> 注意：Vortex 3.0 相对上游经典版本是一次大规模重构。上游旧机制（如 `libvx`、
> `vx_driver_*.cpp`、`vx_proxy`、`.vhdr`/tcpsim、`vx_generate_coremap`、`sw/kernel/src/mex`、
> `libvxe`、`ecall` 系统调用接口等）在本树中已不存在，本文只描述当前实际存在的机制。

---

## 目录

1. [概述](#1-概述)
2. [软件栈总体架构](#2-软件栈总体架构)
3. [目录结构总览](#3-目录结构总览)
4. [工具链](#4-工具链)
5. [构建与配置系统](#5-构建与配置系统)
6. [主机端运行时（sw/runtime）](#6-主机端运行时swruntime)
7. [命令处理器（CP）与命令环](#7-命令处理器cp与命令环)
8. [设备端内核运行库（sw/kernel）](#8-设备端内核运行库swkernel)
9. [链接与 vxbin 二进制格式](#9-链接与-vxbin-二进制格式)
10. [主机/设备共享层（sw/common）](#10-主机设备共享层swcommon)
11. [图形软件栈（sw/gfx）](#11-图形软件栈swgfx)
12. [高层编程模型接入：OpenCL / Vulkan / HIP](#12-高层编程模型接入opencl--vulkan--hip)
13. [模拟器软件（sim/）](#13-模拟器软件sim)
14. [测试与 CI 体系](#14-测试与-ci-体系)
15. [端到端流程示例：vecadd](#15-端到端流程示例vecadd)
16. [常用环境变量参考](#16-常用环境变量参考)
17. [延伸阅读](#17-延伸阅读)

---

## 1. 概述

Vortex 是 Georgia Tech 开源的**全栈开源 RISC-V GPGPU**：它以最小化扩展的方式给
RISC-V 加上 SIMT（单指令多线程）GPU 扩展，并实现了从编译器、运行时、驱动到 RTL、
FPGA/ASIC 的完整软硬件栈。软件系统的职责是：

- **编译**：把 C/C++/OpenCL C/SPIR-V/HIP 内核编译成 RISC-V + Vortex SIMT 扩展的
  设备可执行映像（`.vxbin`）；
- **装载与调度**：主机端运行时把映像与数据装入设备显存，通过**命令处理器（CP）**
  和**内核管理单元（KMU，硬件）**发起内核启动（grid/block/cluster 维度、参数传递、
  局部内存大小等）；
- **多后端适配**：同一套 API 与应用，可透明地运行在 C++ 功能模拟器（simx）、
  Verilator RTL 仿真（rtlsim）、Intel OPAE FPGA、Xilinx XRT FPGA、AMD V80 ASIC
  （SLASH/VRT）、gem5、FireSim 等后端上；
- **高层 API**：通过 PoCL 支持 OpenCL 1.2，通过 Mesa（lavapipe + vortexpipe）支持
  Vulkan，通过 chipStar 支持 HIP，另有原生 C/C++ 内核编程模型；
- **设备服务**：设备侧提供启动代码、printf 控制台、newlib 桩、线程/块索引 CSR、
  warp 级内联指令（投票、洗牌、屏障等）以及张量/光追/图形等专用单元的编程接口。

**版本与依赖锚点**（`VERSION` 文件）：

| 项 | 值 |
|---|---|
| `VORTEX_VERSION` | 3.0 |
| `TOOLCHAIN_REV` | v3.0.1（预编译工具链 bundle 标签） |
| `GEM5_REV` | v25.0.0.1 |

---

## 2. 软件栈总体架构

Vortex 软件栈分为**主机侧（host）**与**设备侧（device）**两半，通过一个极小的
硬件契约（CP 寄存器通道 + CP 可见主机内存 + 设备 MMIO）连接：

```
┌─────────────────────────────── 主机侧（x86/ARM Linux） ───────────────────────────────┐
│  应用 / 测试                                                                            │
│   ├── 原生 C/C++（tests/regression, tests/graphics, ...）                               │
│   ├── OpenCL 应用 ──→ PoCL（ICD, 运行时 JIT 内核）                                      │
│   ├── HIP 应用 ────→ chipStar ─→ PoCL                                                   │
│   └── Vulkan 应用 ─→ Mesa lavapipe + vortexpipe（Gallium 驱动, NIR→LLVM→vxbin）         │
│                                        │                                                │
│  Vortex 运行时 libvortex.so（sw/runtime/stub + sw/runtime/common）                      │
│   · vortex2.h 异步 API（CUDA Driver API 风格）/ vortex.h 旧同步 API                    │
│   · Device/Queue/Buffer/Module/Kernel/Event、CP 命令环、DMA、影子页表、性能计数        │
│                                        │ dlopen（$VORTEX_DRIVER）                       │
│  后端传输层 libvortex-<driver>.so（sw/runtime/{simx,rtlsim,opae,xrt,aved,gem5,firesim}）│
│   · 仅 7 个回调：dev_open/close、cp_reg_read/write、host_mem_alloc/free/pull/push      │
└────────────────────────────────┬───────────────────────────────────────────────────────┘
                                 │ CP 寄存器（MMIO）+ CP 可见主机内存（命令环/参数/DMA 暂存）
┌────────────────────────────────▼───────────────────────────────────────────────────────┐
│  设备侧（RISC-V GPGPU）                                                                  │
│  命令处理器 CP ── 命令环取指：MEM_*/DCR_*/LAUNCH/FENCE/EVENT_*/CACHE_FLUSH/QMD/DRAW      │
│  内核管理单元 KMU（硬件 CTA 调度） ── 按 grid/block/cluster 逐 CTA 启动                  │
│  core/cluster 网格 + L1/L2/L3 + 本地内存 + 固定功能单元（TEX/RASTER/OM/RTU/DXA/TCU）     │
│                                                                                          │
│  设备软件映像 .vxbin（VOLT clang 编译，sw/kernel 运行库链接）                             │
│   · vx_start.S 启动序言 → CTA 入口分发 → 用户内核（__kernel）                            │
│   · CSR/DCR 索引、barrier、printf 环、newlib 桩、TLS                                    │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

三条主机↔设备契约（在 `sw/runtime/common/callbacks.h` 与 `VX_types.toml` 中定义）：

1. **CP 寄存器通道**（32 位 MMIO 读/写）：写 `Q_TAIL` 即"按门铃"提交命令，
   读 `Q_SEQNUM` 轮询命令完成；
2. **CP 可见主机内存**：命令环、内核参数暂存、DMA 暂存区都放在主机内存中，
   由 CP 的 `m_axi_host` 主端口跨 PCIe/QDMA 访问；
3. **设备 MMIO**：退出码（`VX_MEM_IO_EXIT_CODE`）、printf 控制台环
   （`VX_MEM_IO_COUT_ADDR`）、性能计数 CSR 等。

---

## 3. 目录结构总览

| 路径 | 内容 |
|---|---|
| `configure` / `config.mk.in` / `Makefile.in` | 源外（out-of-tree）构建配置系统 |
| `VX_config.toml` / `VX_types.toml` | 微架构配置与软硬件 ABI 契约的**单一事实来源** |
| `sw/kernel/` | 设备端运行库（`libvortex.a` / `libvortex2.a`）与启动代码 |
| `sw/runtime/` | 主机端运行时（`libvortex.so`）与各后端驱动库 |
| `sw/common/` | 主机/设备共享层：ABI 头、软件模型（oracle）、公共工具 |
| `sw/gfx/` | 设备端图形前端内核与 SIMT 软件回退（供 Mesa vortexpipe 使用） |
| `sim/` | 模拟器：simx（C++ 功能模型）、rtlsim（Verilator）等 |
| `hw/` | RTL 源码（含 CP、KMU 的 RTL 实现） |
| `tests/` | 测试集（regression/opencl/hip/vulkan/graphics/raytracing/riscv/mpi/...） |
| `ci/` | 工具链安装脚本、pytest 测试目录（CI v2）、blackbox 启动器、性能门禁 |
| `miscs/` | docker/apptainer、补丁（musl 等）、vortex.cmake |
| `third_party/` | 子模块：softfloat、ramulator2、cvfpu、hardfloat、cocogfx |
| `docs/` | 仓库文档（`designs/` 设计文档、`kb/`、`proposals/` 等） |

---

## 4. 工具链

### 4.1 组件清单

工具链不随仓库构建，而是由 `ci/toolchain_install.sh`（由 `configure` 从
`ci/toolchain_install.sh.in` 生成）从
`vortexgpgpu/vortex-toolchain-prebuilt`（标签 `TOOLCHAIN_REV=v3.0.1`）下载预编译
分片包并拼装安装到 `$TOOLDIR`（默认 `~/tools`）。组件如下：

| 组件 | 用途 | 版本/来源 |
|---|---|---|
| **VOLT LLVM**（`$TOOLDIR/llvm-vortex`） | 设备代码编译器：RISC-V 后端 + Vortex SIMT 扩展（`+xvortex` 目标特性），识别 `annotate("vortex.kernel")`、`__UNIFORM__` 等标注 | `vortex_3.x` 分支 ≈ LLVM 20.1.8 |
| **riscv-gnu-toolchain**（`riscv32/64`） | 设备运行库与 riscv-tests 用的 GCC/ binutils | 预编译 bundle |
| **libc（musl 移植）**（`libc32/64`） | 设备 C 库（打 `miscs/patches/musl_libc.patch`） | 预编译 |
| **libcrt（compiler-rt builtins）** | 设备链接所需的 `libclang_rt.builtins-riscv*.a` | 预编译 |
| **PoCL**（`$TOOLDIR/pocl`） | OpenCL 平台，内置 Vortex 设备后端，运行时 JIT 内核 | `vortex_3.x` 分支（POCL 7.0 基础） |
| **chipStar**（`$TOOLDIR/chipstar`） | HIP 运行时（构建于 PoCL 之上） | `vortex_3.x` 分支 |
| **mesa-vortex**（`$TOOLDIR/mesa-vortex`） | Vulkan/图形栈：lavapipe ICD + vortexpipe Gallium 驱动 | `vortex_3.x` 分支 |
| **Verilator**（`$TOOLDIR/verilator`） | RTL 仿真器（rtlsim） | 上游 master 构建 |
| **sv2v / Yosys / OpenSTA** | ASIC/综合流程 | 预编译 |
| **gem5**（可选，`ci/gem5_install.sh.in`） | 全系统模拟后端 | `v25.0.0.1` |
| **SST / OpenMPI**（可选，`ci/sst_install.sh.in`） | SST 协同仿真 | 15.1.0 |

基础系统依赖由 `ci/install_dependencies.sh` 安装（build-essential、ccache、
opencl-headers、ocl-icd 等，可选 `--vulkan`、`--gem5`、`--mpi`）。
源码级构建各组件的文档见 `docs/building_toolchain.md`；打包侧脚本是
`ci/toolchain_prebuilt.sh.in`。

### 4.2 设备代码编译要点（native 路径）

设备编译器为 VOLT clang，关键参数（见 `tests/*/common.mk`）：

- `--target=riscv$(XLEN)-unknown-elf --sysroot=$(RISCV_SYSROOT)`，
  `-Xclang -target-feature -Xclang +xvortex`（SIMT 扩展）、`+zicond`；
- `-march=rv{32,64}imafd[c]`（按配置可含 `zacas`）、`-mabi=ilp32f/lp64d`；
- `-mcmodel=medany -nostartfiles -nostdlib -O3`；
- 链接脚本 `sw/kernel/scripts/link$(XLEN).ld`，起始地址
  `STARTUP_ADDR`（rv32 为 `0x80000000`，rv64 为 `0x180000000`）；
  链接 `-lc -lm`（musl libc）+ `libclang_rt.builtins` + `libvortex{,2}.a`。

（注意：本树没有旧的 `-mgpu` 开关，目标三元组 + `+xvortex` 特性已取代其作用。）

---

## 5. 构建与配置系统

### 5.1 configure 与源外构建

`configure` 是 autoconf 风格的 bash 脚本，**在独立 build 目录中运行**：

```bash
git clone --recursive <vortex> && cd vortex
sudo ci/install_dependencies.sh && ./ci/toolchain_install.sh
mkdir build && cd build && ../configure --xlen=64 --tooldir=$HOME/tools
make            # 依序构建 third_party → hw+sim → sw → tests
```

- 仅 4 类参数：`--xlen=32|64`、`--tooldir=`、`--osversion=`、`--prefix/--installdir=`；
- `configure` 把源码树中的 `Makefile`/`*.mk`/`*.in` 复制到 build 目录并做
  `@VORTEX_HOME@ @XLEN@ @TOOLDIR@ @OSVERSION@ ...` 替换，源文件继续留在源码树中
  通过 `VORTEX_HOME` 引用——**多棵构建树可共存**，不再需要 `toolchain_env.sh`
  之类的环境设置脚本（3.0 已移除，工具路径全部以绝对路径烘焙进 `config.mk`）；
- 对根目录每个 `*.toml` 各生成两份产物：
  `hw/VX_config.vh`、`hw/VX_types.vh`（Verilog）与 `sw/VX_config.h`、`sw/VX_types.h`（C++）。

`make install` 会安装出**类 CUDA/ROCm 的 SDK 目录**：`kernel/include` +
`kernel/lib<XLEN>`（设备库）、`runtime/include` + `runtime/lib`（`libvortex.so`
及各后端 `.so`）、`lib/pkgconfig`（`vortex-runtime.pc` / `vortex-kernel.pc`），
供 mesa、pocl、chipstar 等下游仅通过 `$VORTEX_PATH` + pkg-config 集成。

### 5.2 VX_config.toml / VX_types.toml 与 gen_config.py

两个根级 TOML 是"单一事实来源"，由 `ci/gen_config.py`（自包含 Python，无第三方
依赖）解析：

- **`VX_config.toml`** —— 微架构参数，命名空间 `VX_CFG_*`：
  - `[platform]`：`NUM_CLUSTERS`、`NUM_CORES`、`SOCKET_SIZE`、icache/dcache/lmem/
    l2/l3 使能等；
  - `[isa]`：VM、`EXT_{M,F,D,C,A,ZICOND,ZACAS,TCU,DMA,DXA,TEX,RASTER,OM,RTU}_ENABLE`、
    `VLEN`；
  - `[pipeline]`：`NUM_WARPS`（默认 4）、`NUM_THREADS`（默认 4）、`ISSUE_WIDTH`
    （表达式 `up(NUM_WARPS/16)`）、`NUM_BARRIERS`、GPR/VGPR bank 等；
  - `[l1cache][l2cache][l3cache][lmem]`：容量/相联度/策略/MSHR/延迟（写回策略自动
    推导：仅当某级是 LLC 且是唯一一致性点时才写回）；
  - `[tcu]`（类型枚举 DPI|DSP|BHF|TFR|FPNEW、数据类型/稀疏支持）、`[dxa]`、`[rtu]`、
    `[tex]`、`[raster]`、`[om]` 等单元配置；
  - `[[enum]]` 声明（如 XLEN、FPU_TYPE、TCU_TYPE）会同时产出 `-DNAME=value` 与
    `-DNAME_<value>` 伴随宏，供 `ifdef` 梯使用。
- **`VX_types.toml`** —— 软硬件 **ABI 契约**，命名空间 `VX_*`：
  - 全部 **CSR 地址表**（线程/线程束/核 ID、CTA 系列、MPM 性能计数基址等）；
  - 全部 **DCR 地址表**（KMU 启动寄存器、MMU/SATP、TEX/RASTER/OM/RTU/DXA 配置）；
  - **内存映射**（`[memmap]`：栈、页表、IO、OM 片元导出孔径、printf 控制台环，
    多为 XLEN 条件表达式）；
  - VM 页表几何（SV32/SV39）、纹理/光栅格式枚举、RTU 槽位/状态、trap 原因。

**三态输出**：`--format cflags`（解析成一组 `-DVX_CFG_*/-DVX_*` 编译宏）、
`--format cpp`（`VX_config.h`/`VX_types.h`，默认未解析模式：`#ifndef` 保护、可被
`-D` 覆盖）、`--format verilog`（`VX_config.vh` 等，机制相同）。
值支持 `"expr: ..."` Python 表达式（`$NAME` 引用、`min/max/up/clamp/pow/clog2`
助手）；每个 `X_ENABLE` 自动派生整型 `X_ENABLED` 镜像。

### 5.3 构建时覆盖架构配置

微架构不再是 Makefile 变量，而是在**任意构建/运行命令**上用 `CONFIGS` 覆盖：

```bash
CONFIGS="-DVX_CFG_NUM_CORES=2 -DVX_CFG_NUM_THREADS=8" make -C tests/regression/vecadd run-rtlsim
# 或者用统一启动器：
./ci/blackbox.sh --driver=rtlsim --cores=2 --threads=8 --app=vecadd
```

`CONFIGS` 经 `gen_config.py --format cflags` 投影成一致的 `-D` 宏，同时作用于
设备内核、主机程序、运行时库与模拟器构建；各构建目录的 `config.stamp` 会在
配置变化时强制重建。

---

## 6. 主机端运行时（sw/runtime）

### 6.1 目录与分发机制

| 目录 | 产物 | 说明 |
|---|---|---|
| `sw/runtime/stub/` | `libvortex.so` | **API 分发器**：首次 `vx_device_open()` 时读环境变量 `VORTEX_DRIVER`（默认 `simx`），`dlopen("libvortex-<name>.so")` 并取得回调表 |
| `sw/runtime/common/` | （编入 libvortex.so） | 后端无关的运行时核心（约 5800 行 C++） |
| `sw/runtime/simx/` | `libvortex-simx.so` | C++ 功能模拟器后端（链 `libsimx.so`） |
| `sw/runtime/rtlsim/` | `libvortex-rtlsim.so` | Verilator RTL 仿真后端（链 `librtlsim.so`） |
| `sw/runtime/opae/` | `libvortex-opae.so` | Intel OPAE/CCI-P FPGA 后端 |
| `sw/runtime/xrt/` | `libvortex-xrt.so` | Xilinx XRT 后端（Alveo，`vortex_afu.xclbin`） |
| `sw/runtime/aved/` | `libvortex-aved.so` | AMD/Xilinx V80 ASIC 后端（VRT 库 + QDMA/SLASH） |
| `sw/runtime/gem5/` | `libvortex-gem5.so`（按 xlen 命名） | gem5 全系统后端（PIO/PIN 虚地址窗口） |
| `sw/runtime/firesim/` | `libvortex-firesim.so` | FireSim 后端（内存/控制跨传输层） |

`sw/runtime/include/` 安装的公共头：`vortex2.h`（**规范异步 API**）、
`vortex.h`（旧同步 API）、`graphics.h`（RASTER setup/binning 主机 API）、
`dxa.h`、`raytrace.h`（RTU CW-BVH 构建与派发）、`tensor_mx.h`/`tensor_sp.h`
（MX/稀疏张量单元主机侧模拟器）。

### 6.2 vortex2.h —— 规范异步 API（CUDA Driver API 风格）

头文件注释明确："所有上层翻译层（POCL、chipStar、未来的 Vulkan/CUDA/HIP/Metal/
OpenGL）都应直接面向本 API"。句柄：`vx_device_h`、`vx_buffer_h`、`vx_queue_h`、
`vx_event_h`、`vx_module_h`、`vx_kernel_h`。要点：

- **设备**：`vx_device_open/retain/release`；`vx_device_query(VX_CAPS_*)`
  （核数/线程束数/线程数/cluster/socket 尺寸、全局/本地内存、 cacheline、
  ISA 能力位、issue 宽度、内存 bank、VM 支持等）；`vx_device_mpm_query`
  （按核读 MPM 性能 CSR，`core_id==0xffffffff` 时跨核求和）；
  `vx_device_dump_perf`（打印 PERF 报告）；`vx_device_max_occupancy_grid`。
- **缓冲**：`vx_buffer_create/reserve`（reserve 用于在指定设备地址"预订"内存，
  如模块映像）、`vx_buffer_map/unmap`（主机镜像暂存）、`vx_buffer_address/access`。
  内存标志：`VX_MEM_READ/WRITE/READ_WRITE`、`VX_MEM_PIN_MEMORY`、
  `VX_MEM_PHYS`（物理地址、不做 VA 翻译）、**`VX_MEM_HOST`**（分配在 CP 的
  `m_axi_host` 主机内存孔径中——命令环、DMA 暂存都用它）。
- **模块/内核**（对应 cuModule/cuFunction）：`vx_module_load_file/load_bytes`
  装载 `.vxbin`；`vx_module_get_kernel(name)` 按符号表取内核；
  `vx_kernel_address` 取入口 PC（RASTER 片元着色器描述符需要它）；
  `vx_kernel_get_max_block_size`。
- **队列**：`vx_queue_create`（优先级、`VX_QUEUE_PROFILING_ENABLE`）、
  `vx_queue_flush/finish`。每队列一个**工作线程**，命令入队延迟与执行解耦，
  保持 OpenCL in-order 语义。
- **异步 enqueue**（全部接收等待列表、返回事件）：
  `vx_enqueue_launch`（携带 `vx_launch_info_t`：内核句柄或旧式 NULL、
  `args_host` 参数块指针（由运行时暂存到设备 scratch）、ndim/grid/block/
  lmem_size/cluster_dim）；`vx_enqueue_commands`（DCR 写 + 启动的**批量提交**，
  类似 NVIDIA pushbuffer）；`vx_enqueue_draw`（图形单命令绘制）；
  `vx_enqueue_copy/read/write`（含 `_rect` 三维跨距变体）、
  `vx_enqueue_fill_buffer`、`vx_enqueue_map/unmap`、`vx_enqueue_barrier`、
  `vx_enqueue_dcr_write/read`、`vx_enqueue_signal/wait_value`。
- **事件**：Vulkan 时间线信号量模型——`vx_event_signal(value)`（单调计数）、
  `vx_event_wait_value(s)`、`vx_event_get_profiling`（queued/submit/start/end 时间戳）。

`vortex.h` 是旧同步 API（`vx_dev_open`、`vx_mem_alloc`、`vx_copy_to_dev`、
`vx_start`、`vx_ready_wait` 等），由 `common/legacy_runtime.cpp` 包装到 vortex2
之上实现，`vx_upload_kernel_file` 解析 vxbin 头、预订 VMA 区间、代码段 RX、
全局区 RW、BSS 清零。

### 6.3 运行时核心（sw/runtime/common/）

内部 C++ 类（`vortex2_internal.h`，引用计数）：`Device`、`Buffer`、`Module`、
`Kernel`、`Queue`、`Event`、`Platform`、`CallbacksAdapter`。核心机制：

- **CP 初始化（`device.cpp: cp_init`）**：在 CP 可见主机内存中分配命令环
  （默认 64 KiB）、head 行与完成槽；编程 CP 队列 0 寄存器（`CP_Q_RING_BASE_*`、
  `CP_Q_HEAD_ADDR_*`、`CP_Q_CMPL_ADDR_*`、`CP_Q_RING_SIZE_LOG2`、`CP_Q_CONTROL=1`）；
  **从 CP 幸存的 `Q_SEQNUM` 恢复**（head/retire 计数可跨进程存活）；从
  `CP_DEV_CAPS` 发现能力（bit24 VM、bit25 DRAW、bit26 QMD、bit27 MMU 故障上报）。
- **提交原语**：`cp_submit_dcr_write/read`、`cp_submit_launch`、
  `cp_submit_launch_qmd`（KMU 描述符重放，一条命令 ≈18 次 DCR 写）、
  `cp_submit_draw`、`cp_submit_cache_flush`（每次 launch 后追加，类 AMD
  ACQUIRE_MEM）、`cp_batch_begin/end`（只发一次 doorbell、只轮询一次完成——
  图形多阶段流水的关键优化）。完成轮询 `Q_SEQNUM`（默认 10 s 停滞告警）。
- **DMA 路由**：`dev_write/dev_read/dev_copy` 是所有设备内存搬运的唯一路由
  （模块映像、参数、COUT、rect/fill/map 全部走它）——**CP 是唯一 DMA 引擎，
  运行时从不直接触碰设备内存**，一律发 `CMD_MEM_*` 描述符。
- **内存簿记**：设备缓冲只是主机侧 `vortex::MemoryAllocator`（`sw/common/mem_alloc.h`）
  中的地址记账；`VX_MEM_HOST` 分配走 `platform()->host_mem_alloc` 并登记
  `{host_ptr, cp_addr, size}`。
- **VM 管理（`vm.cpp`）**：VM 开启时维护**主机影子页表**，`ensure_mmu_satp()`
  在首次启动前编程 `CP_SATP_LO/HI`，`check_mmu_fault()` 启动后读故障上报 DCR。
- **COUT 控制台**：等待完成期间轮询 `VX_MEM_IO_COUT_ADDR` 的 64 个每-hart
  环形缓冲（`wr[64] | rd[64] | data[64][512] | lost[64]`，共 33536 B），按
  `#slot:` 前缀打印，可与运行中内核并发输出。
- **参数池**：4 KB 设备 scratch 槽位空闲列表，暂存内核参数块。
- **性能（`perf.cpp`）**：按 MPM 类别（core/icache/dcache/l2/l3/mem/tcu/raster）
  汇总打印 CI 解析的 `PERF: instrs=..., cycles=..., IPC=...` 报告。

### 6.4 后端 HAL 与七个后端

后端契约极小（`common/callbacks.h` + `callbacks.inc`，模板 `vx_dev_init` 填充
`callbacks_t` 表）：

```c
// 每个 libvortex-<name>.so 只需实现：
dev_open / dev_close          // 设备生命周期
cp_reg_read / cp_reg_write    // CP 寄存器通道（32 位 MMIO；偏移与 RTL CP regfile 一致）
host_mem_alloc / host_mem_free// CP 可见主机内存（返回 CPU 指针 + 设备侧地址）
host_mem_pull / host_mem_push // 一致性钩子（一致型后端为 no-op，影子型做精确同步）
```

各后端一句话概括：

- **simx**：进程内 C++ 功能 GPU（`libsimx.so`）。统一内存——`host_mem_alloc`
  即 `aligned_alloc`，`cp_addr == 指针`；DRAM 钩子把落在已注册主机区间的地址
  路由到主机分配，其余走进程内 RAM。CP 是软件模型 `sim/common/cmd_processor.cpp`。
- **rtlsim**：Verilator 包装的 Vortex RTL（`librtlsim.so`），同样的软件 CP 模型
  驱动 RTL；`dcr_read` 会等待后台运行 future，避免与 Verilator 状态竞争。
- **opae**：`driver.cpp` dlopen `libopae-c.so`，按 AFU UUID 枚举，MMIO 0x1000 起
  的 CP 寄存器空间；主机内存 = CCI-P 共享缓冲（`fpgaPrepareBuffer` + IOVA）。
- **xrt**：加载 `vortex_afu.xclbin`，AP102 风格控制协议（RESET→IDLE）；主机内存 =
  host-only XRT BO（`bo.address()` 即 CP 可见地址）。所有 XRT 调用包在
  `XRT_TRY/XRT_CATCH` 里，异常不得穿越 C 边界。
- **aved**（最大，约 1465 行）：V80 ASIC + VRT 库。sysfs 扫描 SLASH 控制功能
  `10ee:50c1` 定位 BDF；入口静默（quiesce）轮询 `CP_STATUS_BUSY`；三种主机内存
  模式——(1) 硬件 DMA 缓冲（一致）；(2) **staged 影子模式**（当 CP 的
  `m_axi_host` 直达内存 bank 而非 PCIe 从桥时，设备显存中放影子，用
  `host_mem_push/pull` 按门铃/序号水位精确同步，延迟释放暂存到 CP 可证明退役
  之后——修复过 V80 硅片上的数据丢失问题）；(3) avedsim 进程内仿真模式。
- **gem5**：CP 作为 gem5 SimObject 原生运行，运行时是"模拟客户机"程序。两个固定
  虚址窗口：`PIO_BASE 0x20000000`（CP 寄存器，仅 32 位访问）与
  `PIN_BASE 0x100000000`（4 GB 设备显存恒等映射）；"CP 可见主机内存"从 PIN
  窗口顶部的专用 64 MB 孔径中切出。
- **firesim**：无硬件 CP——功能性 CP 模型 + dram 钩子跨 FireSim 传输层
  （`firesim_sim::mem_read/...`），主机内存留在本地进程。

### 6.5 scope（硬件调试）

`common/scope.h/.cpp` 实现 ILA 调试探针协议（2 个 MMIO 寄存器），在 CP
launch-wait 轮询循环中顺带排空采样数据；各后端用 `SCOPE` 宏按需启用。

---

## 7. 命令处理器（CP）与命令环

CP（RTL 在 `hw/rtl/cp`，软件孪生模型在 `sim/common/cmd_processor.cpp`，二者命令
ABI 一致，设计文档 `docs/designs/command_processor.md`）是主机与 GPU 之间唯一的
控制面：

- **命令环**位于 CP 可见主机内存，每行 64 字节（一个 cacheline，最多 5 条命令）；
  命令 `cmd_t` = 操作码/标志/保留 + 3×u64 参数；
- **提交**：主机追加命令后写 `Q_TAIL`（doorbell）；**完成**：读 `Q_SEQNUM`
  （已退役命令的序号计数器，跨进程存活）；
- **命令操作码**：`OP_MEM_WRITE / OP_MEM_READ / OP_MEM_COPY`（DMA）、
  `OP_DCR_WRITE / OP_DCR_READ`（设备配置寄存器）、`OP_LAUNCH`（内核启动：
  一组 KMU DCR 写 + 触发）、`OP_LAUNCH_QMD`（用驻留的 QMD 描述符重放启动）、
  `OP_FENCE`、`OP_EVENT_SIGNAL / OP_EVENT_WAIT`（时间线事件，0 值等待）、
  `OP_CACHE_FLUSH`、`OP_DRAW`（图形绘制，驻留描述符由 CP 展开）；
- **批量提交**（`cp_batch_begin/end`）：追加式批量，单 doorbell + 单完成轮询，
  图形流水（VS→binning→FS）一次提交整段；
- **KMU DCR**：`VX_DCR_KMU_STARTUP_ADDR0/1`（入口 PC）、`ARG0/1`（参数指针）、
  `BLOCK_DIM_*/GRID_DIM_*/CLUSTER_DIM_*`、`LMEM_SIZE`、`BLOCK_SIZE`、
  `WARP_STEP_*` 等，由运行时（legacy `vx_start` 或 vortex2 `enqueue_launch`）
  编程后再触发 `OP_LAUNCH`。

---

## 8. 设备端内核运行库（sw/kernel）

### 8.1 布局与两种产物

- `sw/kernel/Makefile` 构建两个静态库（同一份源码）：
  - **`libvortex.a`** —— 传统模型：**软件调度器**（`vx_spawn_threads`）在设备上
    把 CUDA 风格的 block 分派到 warp/核，`blockIdx/threadIdx` 是 TLS 变量；
  - **`libvortex2.a`** —— **KMU 模型**（同源码加 `-DKMU_ENABLE`）：由硬件
    **内核管理单元**逐 CTA 启动内核入口，索引直接来自 CTA CSR，支持 CTA cluster
    与多入口 vxbin。测试用 `KERNEL_LIB := vortex|vortex2` 选择。
- 目录：`include/`（设备头，安装进 SDK sysroot：`vx_intrinsics.h`、`vx_spawn.h`、
  `vx_spawn2.h`、`vx_barrier.h`、`vx_dxa.h`、`vx_graphics.h`、`vx_raytrace.h`、
  `vx_tensor.h`、`vx_print.h`）；`src/`（启动与运行时源码）；`scripts/`
  （`link32/64.ld`、`vxbin.py`、`kernel_startup.sh`）。

注意：上游的 `mex` 模块库（`vx_mem.c`/`vx_math.c`/...）已不存在，其角色并入
`vx_intrinsics.h` 与各专用头；也没有 `libvxe` 嵌入式 libc——设备运行库很薄
（约 1.4k 行 C/asm），静态链接 musl 移植的 newlib 风格 libc + tinyprintf；
**全树没有任何 `ecall` 系统调用**，主机↔设备交互全部走 MMIO 与 DCR。

### 8.2 启动流程（vx_start.S）

**KMU 模型（`__vx_cta_entry`）**——每个 CTA 的共享序言：

1. 若开启 VM：编程 `satp`（页表基址 PPN + SV39/SV32 模式位）；栈区与页表区为
   MMU 旁路，无栈亦可安全执行；
2. `gp ← __global_pointer`；`sp ← VX_MEM_STACK_BASE_ADDR - (mhartid << VX_MEM_STACK_LOG2_SIZE)`
   （每 hart 独立栈区）；
3. TLS：`tp ← _end + mhartid × __tls_block_size`（步长 = `.tdata+.tbss`，
   修复过与 `__tbss_size` 混用的重叠 bug），再 `call __init_tls`；
4. `call __libc_init_array`（全局构造）；
5. **分发窗口**：`csrr s11, VX_CSR_CTA_ENTRY`（内核入口 PC）+
   `csrr a0, VX_CSR_MSCRATCH`（参数指针）+ `jalr s11`（`norvc`，恰好 4 字节）。
   硬件调度器通过**把 PC 回卷到这 20 字节窗口**为同一 warp 槽位重入下一个 CTA；
   返回后 `wsync; tmc x0` 排空并退役 warp。

多入口 `.vxbin` 为每个内核带一个 `.vx_entry` 桩（`lla s11, <kernel>; j __vx_cta_entry`），
由 `vxbin.py` 打入 `VXSYMTAB` 尾表（配合 VOLT 为 `annotate("vortex.kernel")`
函数发出的 `__vx_kentry_<name>` 别名）。设计文档：
`docs/designs/kernel_entry_and_dispatch.md`、`docs/designs/cta_dispatch_architecture.md`。

**传统模型（`_start`）**：用 `wspawn` 在全部 warp 上运行寄存器初始化
（SATP/gp/sp/tp，`tmc` 掩码控制激活），初始化每 warp TLS 与 init array，然后
`call main; tail exit`；`_Exit` 写 MMIO `VX_MEM_IO_EXIT_CODE` 后 `wsync; tmc x0`。

`scripts/kernel_startup.sh` 对**已链接的 ELF** 做两遍编译探测：按 `.sdata/.sbss`、
`.tdata/.tbss`、`init_array` 段大小决定 `vx_start.S` 重编译所需的
`-DNEED_GP / -DNEED_TLS / -DNEED_INITFINI`。

### 8.3 newlib 桩与 printf（vx_syscalls.c / vx_print.*）

- 桩：`_close/_fstat/_isatty/_lseek/_open/_read/_kill → -1`；`_getpid → vx_hart_id()`；
  `_write` 逐字符走 `vx_putchar`；`_sbrk` 触发 `ebreak` 并返回 0 ——**设备上没有堆**，
  需要动态内存的内核自带分配器（如 tests/kernel/vecadd 的静态池 bump 指针）。
- `vx_putchar`（`vx_print.S`）：**有损每-hart 环形控制台**。写 `VX_MEM_IO_COUT_ADDR`
  处的字符环（64 槽 × 512 B），生产者存字节 → `fence ow,ow`（release）→ 发布
  `wr+1`；环满则**丢弃该字节**并 `amoadd.w` 递增 `lost[slot]`，绝不阻塞
  （CUDA/HIP printf 语义；修复过旧"满则自旋"路径的死锁）。主机在等待完成期间
  排空这些环。
- `vx_printf/vx_vprintf`：tinyprintf 格式化；`vx_putint/vx_putfloat` 经
  `vx_serial`（`vx_serial.S`）**warp 内串行化**——用 `split/join` 保证同一时刻
  只有一个 lane 执行回调，使发散 lane 的 printf 输出有序。

### 8.4 SIMT 指令内联（vx_intrinsics.h）

Vortex 对 RISC-V 的自定义扩展（opcode CUSTOM0=0x0B / CUSTOM1）在此头文件中完整
映射为内联函数：

- **warp 控制（CUSTOM0，funct7 区分）**：
  `vx_tmc(mask)`（线程掩码置位）、`vx_wspawn(num_warps, fn)`（派生 warp）、
  `vx_split(pred)` / `vx_join(sp)`（发散路径压栈/汇合——立即后支配点重汇合机制）、
  `vx_barrier(id, count)`（warp 组屏障）、`vx_wsync()`（warp 内汇合）、
  `vx_pred/vx_pred_n`（谓词化）；
- **异步屏障**（funct3=6）：`vx_barrier_arrive / vx_barrier_wait`（分离式
  到达/等待）、`vx_barrier_expect_tx(count)`（预登记 N 个事务完成事件——使
  DXA 异步拷贝的完成能够"释放"屏障）；
- **身份/配置 CSR**：`vx_thread_id/warp_id/core_id`（0xCC0–CC2）、
  `vx_active_warps/threads`（0xCC3/CC4，LLVM 也读）、
  `vx_num_threads/warps/cores/barriers`（0xFC0–FC4）、`VX_CSR_LOCAL_MEM_BASE`；
- **协作 lane 操作**（funct3=1）：投票 `vx_vote_all/any/uni/ballot`；洗牌
  `vx_shfl_up/down/bfly/idx`；像素 quad 助手（`vx_quad_ddx/ddy` 派生导数）；
  **`vx_wgather`**（CUSTOM1，R4 型，按 lane 收集寄存器——DXA 参数分发与
  4×4 矩阵转置的基础）；
- **打包加载**（funct3=4）：`vx_packlb_f/packlh_f`（4×byte / 2×half 打包，供
  张量片加载）；
- 时间：`vx_rdcycle`（RV32 高低位重读循环）、`vx_rdcycle_sync_begin/end/diff`
  （profiling 对）；`vx_fence() = fence iorw,iorw`；
- `__UNIFORM__ = annotate("vortex.uniform")`：VOLT 一致性/发散分析标注。

### 8.5 两种编程模型（vx_spawn.h / vx_spawn2.h）

**KMU 模型（vx_spawn2.h，现行主流，regression 约 108 个测试在用）**：

```cpp
#include <vx_spawn2.h>

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {   // annotate("vortex.kernel")
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;    // 代理对象读 CTA CSR
    ...
}
```

`__kernel` = `extern "C" __attribute__((annotate("vortex.kernel"), used, retain))`；
`ThreadIdx/BlockIdx/BlockDim/GridDim` 的 `x/y/z` 转换为
`csr_read_nv(VX_CSR_CTA_{THREAD,BLOCK}_{ID,DIM}_{X,Y,Z})`；CTA 相关：
`get_local_group_id() = VX_CSR_CTA_ID`、`get_sub_group_id() = VX_CSR_CTA_RANK`、
`get_num_sub_groups() = VX_CSR_CTA_SIZE`、cluster 维度经
`VX_CSR_CTA_CLUSTER_SIZE` 计算；`__local_mem() = VX_CSR_CTA_LMEM_ADDR`；
`__syncthreads()` 映射到 `vx_barrier`。

**传统模型（vx_spawn.h，约 21 个 `*_v1` 测试保留）**：主机传 grid/block 维度，
设备上 `vx_spawn_threads` 计算 warp→block 映射，用 `vx_wspawn` 派生 warp 执行
`process_thread_groups` 回调，逐 block 设置 TLS 的 `blockIdx/threadIdx` 后调用
用户函数；部分 warp 用 `vx_tmask` 处理不满编线程；调度变量为 TLS，grid/block
维度为全局一致量。

### 8.6 屏障库（vx_barrier.h）

- `vortex::barrier(id, num_warps)` —— CTA 内：硬件 bar 槽号 `(id<<8)|cta_id`；
  `arrive()/wait(phase)` 分离式、`arrive_and_wait()`、`expect_tx(count)`；
- `vortex::gbarrier(id, num_cores)` —— 跨核全局屏障（id 置 bit31）；
- `vortex::group_barrier(id, num_peers)` —— cluster 内跨 CTA 屏障（不嵌 CTA id，
  所有 peer 共享一个硬件槽——用于 DXA 多播前的会合）；
- 异步相：`vx_barrier_arrive/wait`（funct3=6）与 `vx_barrier_expect_tx`
  （`rs2[31]=1`），配合 DXA 异步拷贝实现"数据到达即放行"。

### 8.7 专用单元头

- **`vx_dxa.h`（异步数据搬运引擎）**：`vx_dxa_issue_{1..5}d[_multicast]_wg`
  （CUSTOM0 funct7=3），参数（smem 地址、`(barrier_id<<4)|desc_slot` 打包 meta、
  0–4 维坐标、CTA 多播掩码）经 `vx_wgather` 分布到各 lane 槽位；描述符
  （基址/尺寸/步长/meta）由主机编程进 `VX_DCR_DXA_*`（最多 16 槽）。C++ 封装
  `vortex::dxa_multicast_*` 把"expect_tx + 双屏障"惯用法固化，避免违反
  掩码↔事务数不变量。
- **`vx_graphics.h`（固定功能图形）**：纹理采样 `vx_tex(stage,u,v,lod)`
  （CUSTOM1 funct3=5）、输出合并导出 `vx_om_export(addr,color,depth,mask)`
  （funct3=3，对 `VX_MEM_OM_BASE_ADDR` 孔径的 posted store）、片元图章 CSR
  （`vx_frag_pos/pid/load`，光栅引擎携带 launch 推送）、着色器内自动 LOD
  （`vx_tex_auto_lod`，来自 `sw/common/vx_tex_lod.h`）。
- **`vx_raytrace.h`（RTU）**：`vx_rt_get_attr(slot,tok)`（记分板链窗口读）、
  `vx_rt_cb_ret`（求交回调返回；分发器经 mtvec 注册、`mret` 退出）、
  `vx_rt_wtrace`（warp 级 trace 指令，固定 f0..f7 操作数）、`vx_rt_wait`；
  字节格式常量在 `sw/common/rtu_cfg.h`（CW-BVH 布局，主机转码器 ↔ 设备遍历器）。
- **`vx_tensor.h`（张量核）**：`vortex::tensor::wmma_context/wgmma_context`、
  `fragment_t`、`fill/load_matrix_sync/store_matrix_sync/mma_sync/wgmma_sync`、
  共享内存矩阵描述符、MX 微缩放与 2:4 结构化稀疏元数据加载；
  dtype/格式常量在 `sw/common/tensor_cfg.h`。

---

## 9. 链接与 vxbin 二进制格式

- **链接脚本** `sw/kernel/scripts/link{32,64}.ld`：`ENTRY(_start)`，`. = STARTUP_ADDR`，
  `KEEP(*(.vx_entry))`（配合 `--gc-sections`），`PROVIDE(__tls_block_size =
  __tbss_offset + __tbss_size)`，`_edata` 对齐 64 B（便于 OPAE 上传）；
- **vxbin 格式**（`sw/kernel/scripts/vxbin.py` 生成，`sw/runtime/common/module.cpp`
  解析）：

```
[min_vma : 8 字节][max_vma : 8 字节][平坦二进制映像]
可选尾表: [字符串池][N × 16B {name_off, name_len, pad, pc}][n_symbols : 4B][magic 'VXSYMTAB']
```

  无尾表时回退为"单个 `main` 位于 min_vma"；有尾表即为**多入口二进制**，
  `vx_module_get_kernel(name)` 按 PC 表取内核，RASTER 推送式片元启动与
  `vx_enqueue_commands` 多内核批量提交都依赖它；
- 运行时装载：`vx_module_load_*` 以 `vx_buffer_reserve` 在 `min_vma` 处预订 VMA
  区间 → `dev_write` 上传映像（代码 RX、全局 RW、BSS 清零）→ 缓存内核句柄。

---

## 10. 主机/设备共享层（sw/common）

`sw/common` 不安装，分四类内容：

1. **ABI 契约头**（主机与设备、RTL 共用）：`vx_gfx_abi.h`（图形在管线 ABI：
   定点类型、212 字节图元记录 `rast_prim_t`、tile 头 `rast_bin_header_t`）、
   `gfx_frontend_abi.h`（前端阶段编号 SETUP/SCAN/EMIT/BCOUNT/...）、
   `gfx_fs_desc_abi.h`（片元着色器参数块槽位）、`gfx_sw_abi.h`（纯 C 软件回退
   ABI，Mesa 驱动可直含）、`dxa_meta.h`（DXA 描述符位布局）、
   `vm_types.h`（SV32/SV39 页表项，模拟 MMU 与主机运行时共用）、
   `rtu_cfg.h`、`tensor_cfg.h`、`gfx_dcr.h`、`gfx_tex_const.h`；
2. **黄金模型（oracle，仅主机）**：`gfx_ff_model.h/.cpp`（Rasterizer/DepthTencil/
   Blender/TextureSampler 状态机模型，simx 使用）、`rvfloats.*` +
   `softfloat_ext.*`（Berkeley-SoftFloat 支持的 IEEE-754 运算，含 RISC-V 舍入
   模式与 fflags——是**模拟器** FPU，不是设备代码）；
3. **单一来源数学（主机+设备同编）**：`gfx_sw.h`（软件输出合并：深度/模板/
   逻辑操作/混合、MSAA resolve、全套软件纹理采样/取回/gather/阴影）、
   `gfx_frag_tex.h`、`gfx_frag_rast.h`（递归 tile→quad 边方程覆盖测试）、
   `gfx_setup.h`（近平面裁剪 + 三角形 setup）、`vx_tex_lod.h`；
4. **通用工具（主机 C++）**：`mem_alloc.h`（`MemoryAllocator` 页/块分配器，
   设备地址簿记）、`util.*`、`bitmanip.h`、`ringqueue.h`、`mempool.h`、
   `stringutil.h`（ByteStream）等。

---

## 11. 图形软件栈（sw/gfx）

Vortex 3.0 的图形栈是 **Vulkan 优先、全设备驻留**的，分属两个仓库
（设计文档 `docs/designs/graphics_software_stack.md`、
`docs/designs/vortexpipe_architecture.md`）：

- **mesa-vortex 仓库**（`vortex_3.x` 分支，工具链预装于 `$TOOLDIR/mesa-vortex`）：
  lavapipe（Mesa 的 Vulkan 前端）驱动 **vortexpipe** Gallium 驱动——NIR→LLVM
  代码生成、`.vxbin` 编译、驻留启动、绘制批发射器。它把本仓库纯当作 **SDK**
  消费（`$VORTEX_PATH` + pkg-config）；
- **本仓库 `sw/gfx/`** 只有 4 个文件，是设备内核的单一来源（驱动、simx 测试
  共同编译）：
  - `gfx_frontend_k.h` —— 设备端图形**前端**：`expand_k`（顶点装配）、
    `setup_k`（近平面裁剪/细剖 + 三角形 setup，扫描-计数-发射式写入
    `rast_prim_t` 图元缓冲）、`binning_k`（tile 网格分箱：计数/扫描/直方图/
    基址/散射，产出供硬件 RASTER 使用的 `rast_bin_header_t` tile 缓冲）——
    九次 CP 顺序 launch、两个入口完成；
  - `gfx_resolve_k.h` —— `msaa_resolve_k` 设备端 MSAA 盒滤波 resolve；
  - `gfx_sw_abi.cpp` —— 软件回退 C-ABI（深度/模板/混合/纹理采样等），编成
    LLVM bitcode，vortexpipe 把 NIR 生成的片元着色器与之 `llvm-link` 并内联——
    **固定功能 TEX/OM/RASTER 表达不了的状态走设备上 SIMT 软件路径，绝不回环
    主机**；
  - `libgfx_sw.mk` —— 加 `-mllvm -vortex-divergence-max-bbs=512`
    （内联 OM 合并是大型发散 CFG，需放宽结构化限制）。
- 主机侧图形支持在 `sw/runtime/common/graphics.cpp` + `include/graphics.h`：
  绘制命令构造器、RASTER/OM/TEX 的 DCR 编程发射器，主机 `graphics::Binning`
  保留作离线 oracle；`vx_enqueue_draw` 让整段 sort-middle 图形流水一次提交；
  RASTER 单元在设备上"推送式"启动片元着色器（入口 PC 来自 `vx_kernel_address`）。

---

## 12. 高层编程模型接入：OpenCL / Vulkan / HIP

三种高层 API 都是本 SDK 的**下游消费者**，共享同一编译→装载→启动管线：

### OpenCL 1.2（PoCL）

- 主机程序链接 `-lOpenCL`（ocl-icd），以 `OCL_ICD_VENDORS=$TOOLDIR/pocl/etc/OpenCL/vendors`
  指向预装的 pocl-vortex ICD；
- `clBuildProgram` 时 PoCL **在运行时**调 VOLT clang 把 OpenCL C / SPIR-V 编成
  RISC-V ELF，再经与 native 路径完全相同的 `link{32,64}.ld + vxbin.py` 打成
  vxbin。桥接由环境变量配置（`run-*` 目标自动设置）：
  `POCL_VORTEX_CFLAGS`（clang、`+xvortex`、march、sysroot）、
  `POCL_VORTEX_LDFLAGS`（链接脚本与库）、`POCL_VORTEX_BINTOOL`
  （OBJCOPY + `vxbin.py`）、`LLVM_PREFIX`；
- 详细设计文档：`docs/designs/opencl_on_vortex.md`（331 行，含时序图、
  设备内置函数库、启动/内存模型）。

### Vulkan（Mesa lavapipe + vortexpipe）

- 测试即普通主机程序（`cc -lvulkan`），运行时挂
  `VK_ICD_FILENAMES=lvp_icd.x86_64.json` + `GALLIUM_DRIVER=vortexpipe`；
  GLSL 着色器构建期由 glslc 编为 SPIR-V，RISC-V 内核在 vortexpipe 内部
  运行时生成（NIR→LLVM→vxbin），测试构建不需要 RISC-V 工具链；
- `MESA_VORTEX_STRICT=1`（默认）拒绝静默回退到 llvmpipe；每测试可设
  `NO_RASTER/NO_TEX/NO_OM` 让对应固定功能级走 SIMT 软件路径。

### HIP（chipStar）

- `tests/hip` 用 chipStar 的 `hipcc` 编译（`--offload-pointer-width=$(XLEN)`），
  产出内嵌 SPIR-V 的主机 ELF；运行时 chipStar→PoCL 用与 OpenCL 相同的
  `POCL_VORTEX_*` 变量 JIT。设计文档：
  `docs/designs/hip_on_vortex_chipstar.md`（原生 HIPVortex 工具链仅为设想）。

### 原生 C/C++

即第 8 节的 `vx_spawn2.h`（或 legacy `vx_spawn.h`）模型 + `vortex2.h` 主机 API，
是 regression / graphics / raytracing / mpi 等测试集与一切上层栈的基础。

---

## 13. 模拟器软件（sim/）

| 目录 | 说明 |
|---|---|
| `sim/simx/` | **C++ 功能级 GPU 模型**（"SimX v3"，TLM 风格）：processor/cluster/socket/core、调度器、CTA 分发器、译码/记分板/序列化、各 FU 单元（alu/fpu/lsu/sfu/wctl/csr/opc）、条件编译的扩展单元（amo/tcu/dxa/tex/raster/om/rtu/kmu/mem/dtm）、gem5/SST 耦合。产物两种：**独立 `simx` 二进制**（直接装 ELF/vxbin 跑 riscv-tests 与 tests/kernel，`-s` 统计、`-d` 调试、`-p` remote_bitbang 供 OpenOCD）与 **`libsimx.so`**（供运行时 simx 后端进程内链接） |
| `sim/rtlsim/` | **Verilator** 包装的 RTL（产物 `librtlsim.so` + 独立 `rtlsim`）；加 `-DSIMULATION -DSV_DPI`，配置经枚举伴随宏 + 生成的 `hw/VX_config.vh` 双通道进入 RTL |
| `sim/common/` | 共享基础设施：**`cmd_processor.cpp`（功能性 CP 模型，与 RTL 命令 ABI 一致）**、`elf_loader.*`（RISC-V ELF32/64 装载，支持 riscv-tests 的 HTIF tohost）、`host_monitor.cpp`（监视退出码）、`cout_drainer.cpp`（控制台环排空）、`dram_sim`、`mem.h` |
| `sim/opaesim/`, `sim/xrtsim/` | 把 RTL 核包进 FPGA AFU 外壳的仿真变体 |
| `sim/firesim/`, `sim/avedsim/` | FireSim 与 V80（AVED）对应模型 |

simx 与 rtlsim 之间的**模型等价性**由 CI 的 `model_parity` 检查保障（见下节）。

---

## 14. 测试与 CI 体系

### 14.1 tests/ 布局

| 目录 | 内容 |
|---|---|
| `tests/regression/` | 约 67 个主机+内核测试（vecadd、sgemm*、softmax、bfs、jacobi、stencil3d、sort、printf、fence、occupancy、multikernel、module_reload、vm_test、vm_fault、async_barrier、dxa_*、sgemm_tcu*、gfx_*_kernel、raycast 等） |
| `tests/opencl/` | 约 45 个 OpenCL 基准（blackscholes、kmeans、lavaMD、lud、streamcluster、image_* 等） |
| `tests/hip/` | 4 个 HIP 测试（vecadd、sgemm、histogram、atomicreduce） |
| `tests/vulkan/` | 约 90 个 Vulkan 测试（compute、draw3d、tex_*、msaa*、blend、depth、stencil、subgroup、raytrace、ubo、ssbo…） |
| `tests/graphics/` | 直接驱动固定功能图形单元（gfx_raster/tex/om/pipeline_*） |
| `tests/raytracing/` | 约 40 个 rt_* 测试（BVH、TLAS、SBT、anyhit/intersection 着色器） |
| `tests/kernel/` | 纯设备程序（无主机运行时），直接跑在 simx/rtlsim 二进制上 |
| `tests/riscv/` | 上游 riscv-tests（固定 commit）+ 标量基准，`tohost` 补丁到 Vortex MMIO 退出地址 |
| `tests/runtime/` | 运行时 API 自身的测试（vortex.h 与 vortex2.h 双 API、异步、句柄、时间线事件） |
| `tests/unittest/` | 主机单元测试（直链 sw/common：mem_alloc、软件光栅/纹理/OM/MSAA/分箱、DXA 乱序） |
| `tests/mpi/` | 多进程 MPI 测试（`OMPI_MCA_osc=pt2pt`） |

### 14.2 测试解剖与构建

每个 regression 测试是自包含目录：`common.h`（主机/设备共享参数结构体）、
`kernel.cpp`（设备代码）、`main.cpp`（主机代码，带本地 CHECK 宏）、`Makefile`
（设 `PROJECT/SRCS/VX_SRCS/KERNEL_LIB` 后 `include ../common.mk`）。
`tests/regression/common.mk` 完成：主机 g++ 编译链接 `libvortex.so`；设备 VOLT
clang 编译 + 两遍 crt0 + 链接脚本 + `vxbin.py` 打包；`config.stamp` 配置追踪；
`run-simx/run-rtlsim/run-opae/run-xrt/run-firesim/run-aved` 目标（自动重建对应
后端 `.so`，以 `VORTEX_DRIVER=<driver>` + `LD_LIBRARY_PATH` 运行主机程序）。

### 14.3 运行器与 CI

- **`ci/blackbox.sh`** —— 统一启动器：`--driver=simx|rtlsim|opae|xrt|aved|firesim|gpu`、
  `--app=<测试目录名>`、架构覆盖 `--clusters/--cores/--warps/--threads/--l2cache/--l3cache`、
  `--perf=<MPM 类别>`、`--debug/--vcd/--saif/--scope/--nohup/--rebuild`；
- **`tests/regression/run_parallel.sh`**（configure 生成）—— 并行回归
  （`-j N`、每测试日志、`--fail-fast`）；
- **CI v2（pytest + YAML 目录）**：`ci/testcases/*.yaml`（约 36 个类别文件）
  声明 (category × driver × configs × xlen) 矩阵，`ci/testcase.py` 提供
  lint/matrix/select 规划 CLI；`ci/conftest.py` + `ci/test_runner.py` 为 pytest
  入口；横向检查：
  - **`model_parity`**：simx 与 rtlsim 退休指令数**精确相等**、周期数在容差内
    （默认 5%）；
  - **`perf_gate`**：rtlsim 周期数对 `ci/baselines/perf/*.json` 黄金基线 ±2%
    （仅人工 `--update-baselines` 可更新）；
- **边界检查**：`check_config_boundary.sh`（`VX_config.h` 不得泄入 sw/tests）、
  `check_sw_sim_boundary.sh`、`check_simx_mt_boundary.sh`；
- GitHub Actions（`.github/workflows/ci.yml`）：configure 构建矩阵（xlen 32/64）
  → 每格 (category × driver × xlen) 跑 `pytest ci -m "<category> and <driver>"`；
  另有 `asic_gate.yml` / `fpga_gate.yml` / `apptainer-ci.yml`；
- 性能/调试工具：`ci/roofline.py`、`ci/perfetto.py` + `ci/trace_csv.py`
  （Perfetto 轨迹导出）、`ci/datagen.py`、`ci/gem5_run_app.py`、
  `ci/sst_run_hostless_app.py`。

---

## 15. 端到端流程示例：vecadd

以 `tests/regression/vecadd` 为例，从源码到结果：

1. **编译设备侧**：`kernel.cpp`（`__kernel void kernel_main(kernel_arg_t*
   __UNIFORM__ arg)`，索引 = `blockIdx.x * blockDim.x + threadIdx.x` 读 CTA CSR）
   由 VOLT clang 以 `riscv64-unknown-elf +xvortex` 编译；与 `libvortex2.a`
   （`vx_start.S` 的 KMU 序言、syscall 桩、printf、barrier 等）+ musl libc +
   compiler-rt 链接为 `kernel.elf`（入口 `STARTUP_ADDR`，crt0 已按
   `kernel_startup.sh` 探测结果二次编译）；
2. **打包**：`vxbin.py` 产出 `kernel.vxbin`（VMA 区间 + 平坦映像 + `VXSYMTAB`
   尾表，含 `main` 入口 PC）；
3. **编译主机侧**：`main.cpp`（`<vortex2.h>`）链接 `libvortex.so`；
4. **运行**：`make run-simx`（或 blackbox.sh 选任意后端）——主机进程启动，
   首次 `vx_device_open()` 时 `VORTEX_DRIVER` 触发 dlopen `libvortex-simx.so`，
   运行时 `cp_init()` 在 CP 可见主机内存建立 64 KiB 命令环并编程 CP 队列寄存器；
5. **装载模块**：`vx_module_load_file("kernel.vxbin")` → 预订 `[min_vma,max_vma]`
   设备区间 → `CMD_MEM_WRITE` 上传映像 → `vx_module_get_kernel(mod,"main")`
   查符号表得入口 PC；
6. **准备数据**：`vx_buffer_create` 分配 src0/src1/dst，`vx_enqueue_write` 把主机
   数据经 CP DMA 写入设备（参数块同时暂存进设备 scratch 槽）；
7. **启动**：`vx_enqueue_launch`（`vx_launch_info_t` 携带 grid/block 维度与
   args 指针；occupancy 不足时可用 `vx_device_max_occupancy_grid` 计算形状）→
   运行时编程 KMU DCR → 提交 `OP_LAUNCH` + 尾随 `OP_CACHE_FLUSH`，返回事件；
   CP 消费命令，KMU 按网格逐 CTA 在内核入口启动：`__vx_cta_entry` 序言 →
   `csrr VX_CSR_CTA_ENTRY` 分发进 `kernel_main`；
8. **执行与观测**：内核经 COUT 环打印（主机等待期间排空），完成写
   `VX_MEM_IO_EXIT_CODE`；队列工作线程轮询 `Q_SEQNUM` 观察命令退役，事件完成；
9. **取回**：`vx_enqueue_read` 经 CP DMA 把 dst 拷回主机暂存，主机校验；
   `vx_device_dump_perf` 打印 `PERF: instrs=..., cycles=..., IPC=...`。

---

## 16. 常用环境变量参考

| 变量 | 作用 |
|---|---|
| `VORTEX_DRIVER` | 选择后端（默认 `simx`）：simx/rtlsim/opae/xrt/aved/firesim/gem5-* |
| `CONFIGS` | 构建期 `-DVX_CFG_*/-DVX_*` 覆盖（透传到内核/主机/运行时/模拟器） |
| `VX_XLEN` | 供 CI 选择 32/64 构建树 |
| `XCLBIN_PATH` / `XILINX_XRT` | xrt 后端的 bitstream 与运行时 |
| `OPAE_DRV_PATHS` | opae 后端设备路径 |
| `OCL_ICD_VENDORS` | OpenCL ICD 指向 pocl-vortex |
| `POCL_VORTEX_CFLAGS/LDFLAGS/BINTOOL`, `LLVM_PREFIX` | PoCL 运行时 JIT 内核的编译桥接 |
| `VK_ICD_FILENAMES`, `GALLIUM_DRIVER`, `MESA_VORTEX_STRICT` | Vulkan (lavapipe+vortexpipe) 流程 |
| `VORTEX_CP_TRACE` | CP 命令环追踪 |
| `VORTEX_MEM_SELFTEST` | 启动时内存自检扫频 |
| `VORTEX_CP_POLL_TIMEOUT_S` | 完成轮询超时（秒） |
| `VORTEX_VM_PINNED_SIZE` | VM 固定页区尺寸 |
| `SCOPE_JSON_PATH` | 硬件 scope 采样输出 |

---

## 17. 延伸阅读

仓库内文档（相对仓库根）：

- `README.md` —— 总览与快速上手
- `docs/install_vortex.md`、`docs/building_toolchain.md`、`docs/environment_setup.md`
  —— 安装、源码构建工具链、环境布局
- `docs/simulation.md` —— 各驱动模式（simx/rtlsim/opae/xrt）与 blackbox 用法
- `docs/testing.md`、`docs/continuous_integration.md` —— 测试与 CI 流程
- `docs/debugging.md`、`docs/kernel_debugging.md`、`docs/perfetto_analysis.md`
  —— 调试、内核调试、轨迹分析
- `docs/designs/vortex_runtime_api.md` —— vortex2.h API 设计
- `docs/designs/command_processor.md` —— CP 命令 ABI 与微架构
- `docs/designs/kernel_entry_and_dispatch.md`、`docs/designs/cta_dispatch_architecture.md`
  —— 内核入口与 CTA 分发
- `docs/designs/opencl_on_vortex.md` —— OpenCL/PoCL 全流程
- `docs/designs/graphics_software_stack.md`、`docs/designs/vortexpipe_architecture.md`
  —— 图形软件栈与 Mesa 驱动
- `docs/designs/hip_on_vortex_chipstar.md` —— HIP 接入
- `docs/designs/simx_simulator_architecture.md` —— SimX 架构
- `AGENTS.md` —— 贡献/开发规则（含构建纪律与已知坑）

外部资源：

- Vortex 官网：<https://vortex.cc.gatech.edu/>
- MICRO-54 论文：*Vortex: Extending the RISC-V ISA for GPGPU and 3D-Graphics*（2021）
