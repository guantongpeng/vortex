# Vortex 深度学习全栈使能方案：HIP / PyTorch / Triton / 大模型 / 量化

> **文档定位**：本文是 Vortex GPGPU 面向深度学习 workload 的**全栈软件使能总体方案**（roadmap + 技术设计 + 工作分解）。
> 目标读者：需要规划/实施 Vortex DL 软件栈的编译器、运行时、算子库、框架与模型工程师，以及需要评估硬件改动的架构师。
>
> **基线**：本仓库 `VORTEX_VERSION=3.0`（git `v3.0-911-gbba80f8de`，2026-09-05），工具链 `TOOLCHAIN_REV=v3.0.1`（VOLT = LLVM 20.1.8、PoCL 7.0、chipStar、Mesa 25.1 均为 `vortex_3.x` 分支）。
> 姊妹文档：[vortex_software_system.md](vortex_software_system.md)（Vortex 3.0 软件系统逐层详解，本文不重复其细节，直接引用）。
>
> **结论先行（TL;DR）**：
> 1. Vortex 的**硬件底座已经具备**跑 DL 的关键部件（TCU/WGMMA、DXA 异步拷贝、硬件 CTA 调度、原子、异步屏障），但默认配置是"最小验证档"（1 核、4 warp × 4 线程、L2/L3 关闭、原子关闭、TCU 只开 FP16），且**没有 SFU 数学单元**（exp/log 靠软件 libm）——上 DL 栈之前必须先定"DL 推荐配置档"。
> 2. 现有 HIP 走 **chipStar→SPIR-V→PoCL JIT**，结构性**无法使用张量核/DXA 等 Vortex 内联指令**（`docs/designs/hip_on_vortex_chipstar.md` §5 已明确此差距），必须新建**原生 HIP 工具链**（HIPVortex：hipcc 直编 RISC-V ELF→vxbin + `libhip_vortex` 直接建在 `vortex2.h` 上）。
> 3. PyTorch 走 **PrivateUse1 自定义后端**（`torch-vortex`），不 fork PyTorch 的 rocm 后端；先用 eager + CPU-fallback 跑通，再接 torch.compile。
> 4. Triton 需要**自建 backend 插件**（`triton/third_party/vortex`），把 `tl.dot→WGMMA`、`async copy→DXA`、`shared memory→LMEM` 映射打通——这是全项目编译器侧最大的单项工程，也是性能上限的关键。
> 5. 量化路线：硬件 dtype（INT8/INT4/FP8/MX/NVFP4/2:4 稀疏）在 RTL 里**已实现、默认关闭**，软件侧从零开始；优先顺序建议 **W4A16 软件反量化 → W8A8 INT8 TCU → FP8 → MX/FP4 → 2:4 稀疏**。
> 6. Mamba/SSM 与 LLM decode 是**带宽型**负载，瓶颈在内存系统（多 bank HBM/DDR、L2/L3、DMA 并发），需要硬件侧配合（见 §9）。
> 7. 总量级：完整走到"ResNet/YOLO/SAM/DINOv3/Mamba/LLM/VLM + 量化"约 **35–55 人月**，建议 6–10 人团队 12–18 个月分 6 个里程碑推进（见 §11）。

---

## 目录

- [0. 摘要与总路线图](#0-摘要与总路线图)
- [1. Vortex 软硬件现状盘点](#1-vortex-软硬件现状盘点)
- [2. 目标定义与总体架构](#2-目标定义与总体架构)
- [3. M1：原生 HIP（HIPVortex）](#3-m1原生-hiphipvortex)
- [4. M2：基础算子库](#4-m2基础算子库)
- [5. M3：Triton 接入](#5-m3triton-接入)
- [6. M4：PyTorch 后端（torch-vortex）](#6-m4pytorch-后端torch-vortex)
- [7. M5：量化支持](#7-m5量化支持)
- [8. M6：模型使能](#8-m6模型使能)
- [9. 硬件侧配合工作](#9-硬件侧配合工作)
- [10. 测试、CI 与工具链工程](#10测试ci-与工具链工程)
- [11. 里程碑、工作量与团队规划](#11里程碑工作量与团队规划)
- [12. 风险登记册](#12-风险登记册)
- [13. 附录：映射表与资料索引](#13-附录映射表与资料索引)

---

## 0. 摘要与总路线图

### 0.1 一句话方案

以 **vortex2.h + VOLT 编译管线**为唯一地基，按"**原生 HIP 运行时 → 基础算子库 → Triton 后端 → PyTorch 后端 → 量化 → 模型**"六步，把 Vortex 从"OpenCL/Vulkan 功能验证平台"升级为"能原生运行 PyTorch 生态、覆盖 CNN/ViT/SSM/LLM/VLM、支持多档量化"的 GPGPU 软硬件栈。

### 0.2 总路线图（六里程碑）

```
M1 原生 HIP          M2 算子库            M3 Triton            M4 PyTorch
hipcc(VOLT)  ──►  vortex-prim/blas  ──►  triton backend  ──►  torch-vortex
libhip_vortex      conv/rand/attention    tl.dot→WGMMA        PrivateUse1
     │                  │                  DXA/AOTpipeline      │
     └────────┬─────────┴──────────┬───────────┬───────────────┘
              ▼                    ▼           ▼
        M5 量化（W4A16→W8A8→FP8→MX/FP4→稀疏；依赖 M2 GEMM + 硬件 dtype 开启）
              ▼
        M6 模型（ResNet/YOLO → SAM/DINOv3 → Mamba → LLM/VLM；依赖 M2/M4/M5）
```

依赖关系要点：

- M1 是所有上层的东西的地基（Triton 的 driver shim、PyTorch 的 allocator 最终都调它或调 vortex2.h）；
- M2 与 M3 可并行（算子库走 HIP C++，Triton 走 Python/MLIR，人技能不同）；
- M4 依赖 M1（内存/流/事件语义）+ M2（核心算子），M3 完成后 M4 可切换到 torch.compile 路线；
- M5 依赖 M2 的 GEMM 库（量化 kernel 是 GEMM 的 epilogue/变体）+ 硬件侧 dtype 使能（§9 P0）；
- M6 依赖 M4/M5，模型之间按"带宽敏感度"递进（CNN → ViT → SSM → LLM decode）。

### 0.3 与现状的三条可选路线（详见 §2.3）

| 路线 | 内容 | 结论 |
|---|---|---|
| A. 深化 chipStar+PoCL | 继续走 SPIR-V→PoCL JIT | 只能作为**过渡期兼容层**，结构性到不了 TCU/DXA |
| B. TheRock/ROCm fork | 把 ROCm 全家（ROCR、hipBLAS、RCCL、MIOpen…）整体移植到 RISC-V 目标 | 工作量与维护成本极高，且 ROCm 运行时深度绑定 AMDGPU/HSA，**不建议整体采用**；但其**库源码**（rocPRIM/hipBLASLt/Composable Kernel/RCCL）按 API 逐库选择性移植极有价值（用户已 clone `~/aichip/TheRock`，作为源码与构建参考） |
| C. **精简原生 HIP（推荐）** | 自建 `libhip_vortex`（HIP C API → vortex2.h）+ `hipcc` 壳（VOLT 后端）+ 自建算子库 + Triton 后端 + PyTorch PrivateUse1 | 完全掌控 ABI 与内联指令暴露面，工作量可控，与仓库现有 SDK 布局（`vortex-kernel.pc` 明确写着供 "custom HIP / OpenCL drivers" 使用）一致 |

---

## 1. Vortex 软硬件现状盘点

### 1.1 硬件现状

#### 1.1.1 ISA 与可编程模型

- **基础 ISA**：RV32IMAFDC / RV64IMAFDC（XLEN 可配，`EXT_D` 随 64 位自动开）+ Zicond（默认开）+ Zacas（可选）+ **RVA 硬件原子（`EXT_A_ENABLE`，默认关！）**。
- **自定义 SIMT 扩展**（RISC-V CUSTOM0/CUSTOM1，完整映射见 `sw/kernel/include/vx_intrinsics.h`，编码见 `hw/rtl/core/VX_decode.sv`、`hw/rtl/VX_gpu_pkg.sv`）：
  - warp 控制：`tmc`（线程掩码）、`wspawn`、`split/join`（IPDOM 发散栈重汇合）、`pred`、`bar`、`wsync`；
  - **异步屏障**：`arrive/wait` 分离 + `expect_tx`（事务到达计数，DXA 完成即放行——等价 CUDA `cp.async` 的 barrier 语义）；
  - 协作原语：vote（all/any/uni/ballot）、shuffle（up/down/bfly/idx）、`wgather`（按 lane 收集）、quad 导数；
  - 打包加载：`packlb/packlh`（4×byte / 2×half 单指令跨步装载，TCU tile 供数用）；
  - 加速器内联指令：TCU（WMMA/WGMMA/WMMA_SP/WGMMA_SP/TCU_LD）、DXA issue、TEX 采样、OM export、RTU trace；
  - CTA 编程模型：`VX_CSR_CTA_*`（thread/block/grid id/dim、CTA id/rank、cluster、入口 PC、LMEM 基址）。
- **注意（对 DL 栈有直接影响的事实）**：
  - **没有 SFU 数学指令**（无 exp/log/rsqrt/sin/cos 硬件单元；`VX_sfu_unit.sv` 只挂 wctl/CSR/DXA/TEX/RTU PE）。softmax/GELU/SiLU/sigmoid 走 musl libm 软件 → 这是每层都吃的税，见 §9 P1。
  - 没有 prefetch / cache-hint 内联指令；`vx_fence()` = `fence iorw,iorw`。
  - 设备侧**没有堆**（`_sbrk` 触发 ebreak），需要动态内存的 kernel 自带分配器。
  - warp 大小 = `NUM_THREADS`（4..32 可配），**不是天然 32**——CUDA/HIP 生态大量假设 warpSize=32，见 §2.5。

#### 1.1.2 微架构与规模（`VX_config.toml` 默认值 vs DL 推荐档）

| 维度 | 默认（最小验证档） | DL 推荐档（仿真/FPGA） | 说明 |
|---|---|---|---|
| 拓扑 | 1 cluster × 1 core，SOCKET_SIZE=1 | ≥1 cluster × 8–32 cores，SOCKET_SIZE=2–4，共享 L1 | 多核必须开 L2；多 cluster 必须开 L3（单一 LLC 一致性点） |
| 每 core warp 数 | 4 | 8–16 | 决定每 core 驻留 CTA 数（占用率） |
| warp 大小 NT | 4 | **32**（或 16） | CUDA 生态兼容 & TCU tile 几何更大 |
| 每线程寄存器 | 32 int + 32 fp（64 项统一计分牌命名空间） | 同 | 寄存器压力是 Triton/CUTLASS tile 大小的硬约束 |
| issue 宽度 | 1（`up(NW/16)`） | ≥2–4 | WGMMA warpgroup = ISSUE_WIDTH 锁步 |
| I/D-cache | 各 16KB/4 路 | D 32–64KB | L1 为 socket 级共享（每 4 core 一份） |
| LMEM（共享内存） | **16KB/core** | **64–128KB** | 见 §9 P0：16KB 放不下 Triton 双缓冲 tile；smem 描述符偏移字段 16bit 上限 64KB |
| L2 / L3 | 1MB/8 路（关）、2MB/8 路（关） | 1–4MB 开 / 2–8MB 开 | L2 多核必需；LLC 写回、上层写穿 |
| TCU | TFR 后端，**仅 FP16 开**；WGMMA/MX/INT8/FP8/FP4/稀疏全默认关 | 全开（分阶段验证） | dtype 使能见 §7.1、§9 P0 |
| DXA | 每 8 core 1 个引擎，16 描述符 | 开 | GEMM/Conv 的 tile 预取 |
| 原子 | **EXT_A 默认关** | **开** | PyTorch 的 scatter/index 系、HIP atomicAdd 都要 |
| 屏障槽 | NUM_BARRIERS=8/核，MAX_BAR_EVENTS=32 | 16+ | 每驻留 CTA 占槽 |
| 内存平台 | 2 bank × 64B，峰值带宽默认填 460GB/s@400MHz（V80 级） | 多 bank/HBM 交织 | `docs/proposals/hbm_bank_interleaving_proposal.md` |

**已实测性能锚点**（`ci/baselines/perf/*.json`，rtlsim 周期数）：

| 负载 | 配置 | 周期数 | 备注 |
|---|---|---|---|
| 标量 sgemm 128³ | NT=16 | 1.82M | 纯 FPU 路径 |
| sgemm_tcu fp16 128³ | NT=8, 1 warp | 260K | TCU 加速 ≈7×（vs 标量） |
| sgemm_tcu fp16 128³ | 2 core, NT=16 | 59K | 多核扩展有效 |
| WGMMA SS NRC=32 128³ | — | 759K | |
| WGMMA **2:4 稀疏** 128³ | — | 408K | 稀疏 ≈1.86× 加速实测 |
| README（FPGA 实测） | 32 core @ Stratix10 200MHz | 25.6 GFLOPS | 上游 v1 数据，量级参考 |
| ASIC 综合（ASAP7 门禁） | core(NT=NW=16) 500MHz、cache 800MHz、TCU 500MHz、DXA 800MHz、RTU 400MHz | — | V80 实际 300MHz 运行 |

#### 1.1.3 TCU（张量核）能力明细

（权威文档 `docs/designs/tensor_core_wgmma_engine.md`；软件面 `sw/kernel/include/vx_tensor.h` + `sw/common/tensor_cfg.h`）

- **dtype 全集**：fp32 / tf32 / fp16 / bf16 / fp8(e4m3) / bf8(e5m2) / int32 / int8 / uint8 / int4 / uint4 / **mxfp8 / mxbf8 / mxfp4（32 元素块缩放）/ nvfp4（16 元素块缩放）** + **2:4 结构化稀疏**（独立 opcode WMMA_SP/WGMMA_SP，元数据经 `TCU_LD` 预载到 per-warp SRAM）。
- **tile 几何随 NT 缩放**：`tcM = 2^ceil(lgNT/2)`，`tcN = tcK = 2^floor(lgNT/2)`；WGMMA per-warp tile：`xtileM = 2·tcM`、`xtileK = 2·tcK`（FEDP2K 再翻倍）、`xtileN = NRC·NT/xtileM`，NRA=4、NRC∈{8,16,32}。例：NT=16 → tcM=tcN=tcK=4，per-warp WGMMA M=8、K=8、NRC=32 时 N=64。
- **操作数来源**：RS（A 在寄存器 f24..f27）/ SS（A、B 共享内存描述符，`smem_matrix_desc` = 16bit LMEM 偏移 + 16bit 行步长；stride=0 表示 block-major）；tile buffer（`VX_tcu_abuf` per-block + 共享 `VX_tcu_bbuf`）直接从 LMEM DMA 口取数，支持 block-major 与 k-major（DXA 转置写入）两种布局。
- **warpgroup 锁步**：`VX_tcu_lockstep` 保证单 CTA 独占共享 B buffer；跨 CTA 冲突时 defer。
- **已知开放 bug**：XLEN=64 `sgemm_tcu_wg` fp16 rtlsim 数值失败、tf32 rtlsim 周期计数污染（`tensor_core_wgmma_engine.md` §7）——**上量化/dtype 前必须先清掉**。

#### 1.1.4 内存系统与同步

- 层级：core → socket 级 L1（icache/dcache，每 4 core 一份）→ cluster L2 → 全局 L3（唯一 LLC，写回）→ AXI 多 bank。无目录/侦听——LLC 之上写穿，**多核必须配共享级**。
- **AMO 在 LLC 执行**（LR/SC 预约表 + `amo{add,swap,xor,or,and,min,max}`，可选 amocas）；LLC 顺序一致。
- 异步屏障 + `expect_tx`：`VX_bar_unit` 每核 warp×barrier 槽表；DXA 完成（LMEM bank 写窥探 → txbar 总线）递减事务计数放行屏障——**这就是 CUDA `cp.async.wait_group` + `bar.arrive` 的 Vortex 等价物，GEMM 双缓冲的同步骨架**。
- 全局屏障 `VX_gbar_unit`（cluster 级）、cluster 内 group barrier（DXA 多播前会合）。
- 设备地址空间：`VX_MEM_USER_BASE_ADDR=0x10000` 起，全局 4GB(32bit)/8GB(64bit)；每 hart 栈 8KB；printf 为 64 槽×512B 有损环。

#### 1.1.5 主机↔设备通道（CP + KMU）

- **CP（命令处理器）**：主机 pinned 内存中 64KB 命令环（64B/行），doorbell=写 `Q_TAIL`，完成=轮询 `Q_SEQNUM`；命令集 `MEM_WRITE/READ/COPY`（DMA）、`DCR_WRITE/READ`、`LAUNCH`（≈18–22 条 KMU DCR 写 + 触发）、`FENCE`、`EVENT_SIGNAL/WAIT`、`CACHE_FLUSH`（每次 launch 后自动追加，类似 AMD ACQUIRE_MEM）；批量提交 `cp_batch_begin/end` 单 doorbell。
- **CP 是唯一 DMA 引擎**、默认单队列（`VX_CP_NUM_QUEUES=1`）→ 当前**所有流在设备端实际串行**（多流语义缺口，见 §9 P1）。
- **KMU（硬件 CTA 调度）**：grid/block/cluster 维度 + 入口 PC + 参数指针 + lmem_size 一次编程，逐 CTA 硬件发射；设备侧入口 `__vx_cta_entry` 20 字节分发窗（`csrr VX_CSR_CTA_ENTRY; jalr`），CTA 退役后 PC 回卷复用 warp 槽——**无软件调度器开销**。
- 后端：simx（C++ 功能模型，最快）/ rtlsim（Verilator，周期精确）/ opae / xrt（Xilinx）/ **aved（V80 ASIC，SLASH/VRT）** / gem5 / firesim。

### 1.2 软件现状

（详见姊妹文档 [vortex_software_system.md](vortex_software_system.md)，此处只列对 DL 使能关键的点。）

| 层 | 现状 | 对 DL 栈的意义 |
|---|---|---|
| **编译器 VOLT** | LLVM 20.1.8 + `+xvortex +zicond`，`riscv{32,64}-unknown-elf`，`annotate("vortex.kernel")` + `__UNIFORM__` 发散分析；lld + `link{32,64}.ld` + `vxbin.py`（多入口 VXSYMTAB） | **设备编译管线已被 PoCL 与 vortexpipe 双消费者验证**，HIP/Triton 是第三个消费者；VOLT 仓库需要小改（HIP header 内建、Triton 用的 LLVM intrinsic/inline-asm 路径） |
| **主机运行时 `vortex2.h`** | CUDA Driver API 风格异步 API：device/buffer/queue/event/module/kernel + `vx_enqueue_launch/copy/commands/fill/...`；`vx_launch_info_t`（grid/block/cluster/lmem/args blob）；7 个后端 HAL | **上层翻译层的规范接口**（头文件注释原话）；缺：per-arg setter、事件回调、内存池、多硬件流并发 |
| **设备运行库 `libvortex2.a`** | KMU 模型启动序言、newlib 桩（无堆）、printf 环、`vx_intrinsics.h`/`vx_tensor.h`/`vx_dxa.h`/`vx_barrier.h` | HIP 设备侧 runtime 的直接原料；**无 device malloc、无 malloc 上层包装** |
| **OpenCL** | PoCL 7.0 fork（`vortexgpgpu/pocl @ vortex_3.x`），CL 1.2 全特性（含 image，走 TEX 快路径）；44 个 Rodinia/Parboil 级应用 | HIP 过渡期兼容层；PoCL 的 kernel arg 缓冲、异步 worker 模型可参考 |
| **HIP（现状）** | chipStar→SPIR-V→PoCL JIT；rv32/rv64 vecadd/sgemm 过；rv32 conformance ~36%；4 个测试；**到不了 TCU/DXA 内联指令** | M1 要替换的对象；chipStar 的 HIP API 语义测试集可复用为 `libhip_vortex` 的验收 |
| **Vulkan** | Mesa 25.1 fork，lavapipe + vortexpipe（NIR→LLVM→vxbin，绕过 PoCL） | 证明"直连 vortex2.h 的驱动"工程上可行——HIPVortex 是同一模式的第二个实例 |
| **测试资产** | regression 66 个（含 sgemm_tcu 全家 11 个、softmax/relu/dropout、dxa、amo、vm）；opencl 44；vulkan 78；rt 34；runtime/unittest/riscv/mpi；CI v2（pytest+YAML 目录，model_parity + perf_gate ±2%） | DL 栈每层的回归底座；perf 基线机制直接沿用 |
| **DL 相关** | **无 PyTorch、无 Triton、无 BLAS/PRIM 库、无量化**（PyTorch 仅在 `hw/unittest/tcu_fedp/fedp.py` 里当宿主机 golden 参考用；cupbop 只是预编译二进制夜间流程） | 全部从零开始，即本文 M1–M6 |

### 1.3 能力 × DL 需求差距矩阵

✅=已具备且验证 ⚠️=已具备但默认关/规模不足/未验证 ❌=缺失

| 能力 | 状态 | 差距说明 |
|---|---|---|
| 设备编译器（C/C++ → vxbin，AOT） | ✅ | VOLT，LLVM20 |
| 硬件 CTA 调度 / grid-block-cluster | ✅ | KMU |
| warp 原语（vote/shfl/split-join/wgather） | ✅ | |
| CTA 屏障 + 异步屏障 + expect_tx | ✅ | |
| 硬件原子（AMO） | ⚠️ | RTL 有，`EXT_A` 默认关；PyTorch 必需 → 默认开 |
| 张量核 WMMA/WGMMA（FP16） | ⚠️ | RTL+测试有，默认关（`EXT_TCU_ENABLE=false`）；需常开并固化配置 |
| 张量核 INT8/FP8/FP4/MX/稀疏 | ⚠️ | RTL 已实现、**全默认关、无 DL 级验证**（数值开放 bug 见 §1.1.3） |
| 异步拷贝引擎（cp.async/TMA 等价） | ✅ | DXA + k-major + 多播 |
| 共享内存容量 | ⚠️ | 16KB/core 太小（Triton/CUTLASS 双缓冲 tile 需 64KB+；smem 描述符 16bit 偏移上限 64KB） |
| 超越函数（exp/log/rsqrt） | ❌ | 无 SFU 数学单元，走 libm 软件 |
| 设备堆 / device malloc | ❌ | `_sbrk` 即 ebreak |
| 多流并发（异步 memcpy 与 kernel 重叠） | ⚠️ | 运行时多 queue 有，设备端单 CP 队列串行 |
| HIP 原生（直编 + 直连运行时 + 内联指令暴露） | ❌ | M1 |
| BLAS / PRIM / 卷积 / attention 算子库 | ❌ | M2（仅测试级 sgemm 种子） |
| Triton 后端 | ❌ | M3 |
| PyTorch 后端 | ❌ | M4 |
| 量化（kernel + 框架接入） | ❌ | M5 |
| LLM 推理引擎（KV cache/paged attn/sampling） | ❌ | M6 |
| 多卡 / 集合通信 | ❌ | MPI 测试有雏形；RCCL 级远期 |
| 性能观测 | ✅ | MPM 计数器（含 TCU/DXA 类）+ Perfetto + roofline + perf_gate |

### 1.4 差距清单（按层汇总）

1. **配置层**：没有"DL 档"硬件配置（cores/warps/NT=32/LMEM/L2/L3/TCU dtype/EXT_A 组合）的固化与门禁——先定档、再让 CI 全量跑该档。
2. **运行时层**：多流并发、host 注册内存/统一内存语义、device malloc、内核参数 >4KB、事件回调；这些决定 HIP API 兼容度与 PyTorch allocator 行为。
3. **编译器层**：HIP 头的内联指令暴露面（wmma/wgmma/DXA/异步屏障）；Triton 后端；libdevice（数学函数表）。
4. **算子层**：从零建 GEMM/conv/attention/prim/rand 库；无 autotuner。
5. **框架层**：PyTorch PrivateUse1 全套（allocator/dispatch/RNG/compile）。
6. **模型层**：量化 kernel、KV cache 管理、selective_scan 等模型特有算子。
7. **硬件层**：LMEM 扩容、SFU 数学、多队列 DMA、HBM 交织（§9）。

---

## 2. 目标定义与总体架构

### 2.1 分层目标（G0–G3）与量化验收指标

| 目标档 | 载体 | 验收指标（建议） |
|---|---|---|
| **G0 功能全通** | simx（+`SIMX_FUNCTIONAL` 快速档） | 所有 M1–M6 测试在"DL 档"配置下数值正确（fp32 atol 1e-5 / fp16 1e-2 相对）；PyTorch resnet50/yolo/sam/dinov3/mamba/7B-LLM 推理端到端跑通（不限时延） |
| **G1 周期可析** | rtlsim + MPM + Perfetto | 关键算子（GEMM fp16/int8、attention、conv）达到 roofline 带宽/算力利用率的既定百分比（建议首版 ≥30% 峰值算力 / ≥40% 峰值带宽）；perf_gate 基线入库 |
| **G2 FPGA 实测** | U55C / V80 | resnet50 推理、LLM decode 实测延迟数据；HBM 带宽利用率报告 |
| **G3 ASIC 定档** | ASIC 综合门禁（ASAP7/Synopsys） | TCU dtype 全开后的面积/频率代价量化（tcu@500MHz 基线不回退 >x%）；DL 档配置进 ASIC 门禁目录 |

### 2.2 目标软件栈架构（对标 CUDA/ROCm）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 应用层    HF transformers / torchvision / ultralytics / vLLM(移植) / 自研     │
├─────────────────────────────────────────────────────────────────────────────┤
│ 框架层    PyTorch (PrivateUse1→"vortex")                                    │
│            torch-vortex: c10 Allocator/DeviceGuard/Stream/Event/Generator   │
│            ATen 算子（手写 kernel + Composite 复用）  torch.compile(可选)     │
├──────────────────────────────┬──────────────────────────────────────────────┤
│ 图编译层   torch.compile/Inductor ──► Triton ──┐                            │
│            （可选：torch.export→AOT）            │ tl.dot→WGMMA              │
├──────────────────────────────────────────────┼──────────────────────────────┤
│ 算子库层   vortex-prim（reduce/scan/sort）  vortex-blasLt（GEMM 模板库）       │
│            vortex-dnn（conv/bn/pool）      vortex-rand（Philox）             │
│            vortex-attention（flash/paged）  vortex-quant（§7）               │
├──────────────────────────────────────────────┼──────────────────────────────┤
│ 语言层     HIP C/C++（hipcc）                │ Triton（vortex backend）      │
│            设备头：warp 原语 / mma / DXA       │  M3                          │
├──────────────────────────────────────────────┴──────────────────────────────┤
│ 运行时层   libhip_vortex（HIP C API）──► libvortex.so（vortex2.h 规范 API）   │
│            Triton driver shim 同样落在这 ──┘    （stub 分发 + 7 后端 HAL）     │
├─────────────────────────────────────────────────────────────────────────────┤
│ 硬件接口   CP 命令环（DMA/LAUNCH/DCR/EVENT）→ KMU CTA 调度 → Vortex GPU       │
│            TCU(WGMMA) DXA L1/L2/L3 LMEM AMO 异步屏障                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

设计原则（继承仓库现有约定）：

1. **vortex2.h 是唯一规范接口**——`libhip_vortex` 与 Triton shim 都只是它的翻译层，不绕过运行时直敲后端。
2. **vxbin 是唯一设备映像格式**——hipcc 与 Triton 后端最终都产 vxbin（多入口 VXSYMTAB），复用 `vx_module_load_bytes` + `vx_module_get_kernel`。
3. **VOLT 是唯一设备编译器**——任何前端（HIP、Triton、未来 MLIR）最终落到"LLVM IR（riscv triple + `+xvortex`）→ lld + link 脚本 + vxbin.py"。
4. **一切配置经 `VX_config.toml`/CONFIGS**——"DL 档"是配置组合，不是 fork。

### 2.3 路线决策（A/B/C 详评）

**路线 A：深化 chipStar + PoCL**
- 优点：零编译器投入，今天就能跑更多 HIP 源码。
- 致命缺点：SPIR-V 中间层**结构性隔离**了 Vortex 内联指令（wmma/wgmma/DXA/expect_tx 无法从 HIP 源码触达）；PoCL JIT 每次启动重编；rv32 conformance 36%。
- 定位：M1 落地前的**过渡兼容层**，保留 `tests/hip` 作为 API 语义对照。

**路线 B：TheRock/ROCm 整体移植**
- 内容：在 TheRock（ROCm 官方模块化构建系统，用户已 clone 至 `~/aichip/TheRock`）中新增 `vortex` target family：换 `amd-llvm`→VOLT、ROCR Runtime→vortex 后端、把 hipBLAS/Tensile、rocPRIM、rocRAND、RCCL、MIOpen、hipFFT 全链在 RISC-V 目标上重建。
- 现实评估：ROCR 深度绑定 HSA/HSAKMT（KFD 内核驱动、AQL 队列、代码对象 V3）；AMDGCN 假设遍布 comgr/Tensile/MIOpen（汇编 intrinsics、`__builtin_amdgcn_*`、LDS 布局）；RISC-V 目标对上游不可上游化，维护面 = fork 整个 ROCm。**不建议整体采用**。
- 保留价值（重要）：TheRock 中的**纯 HIP-C++ 库源码**是高质量的移植原料——rocPRIM（block/device 原语）、hipBLASLt（GEMM 模板 + epilogue 语义）、Composable Kernel（tile 编程范式）、rocRAND（Philox/Threefry）、RCCL（集合通信算法层）。M2 的策略是"**API 兼容、实现换底**"：保留其公共头与语义测试，把设备代码换写到 vx_intrinsics/vx_tensor 上。TheRock 同时是后续**构建/发布编排**的参考（其 artifact 分包、CI 矩阵、PyTorch wheel 夜间构建流程值得抄）。

**路线 C：精简原生 HIP（本文主线）**
- `libhip_vortex`：直接实现 HIP Runtime C API 子集（hipDevice/Memory/Stream/Event/Module/Kernel/printf/occupancy），落 vortex2.h。
- `hipcc-vortex`：hipcc 壳（可基于 ROCm hipcc 的 Python driver 改造，或自写薄壳）调 VOLT clang 产 vxbin，**不走 SPIR-V**。
- `libhip_device`（设备头）：HIP 设备内建（warp 原语、原子、math）→ vx_intrinsics.h 映射；`hip::wmma`-风格头 → vx_tensor.h。
- 上层：自建算子库（复用 TheRock 源码逻辑）、Triton 后端、PyTorch PrivateUse1。

### 2.4 版本锚定策略

| 组件 | 锚定 | 说明 |
|---|---|---|
| Vortex 本体 | 本仓库 master（v3.0-911 基线） | DL 改动全部走 PR 进主干，避免长期 fork |
| VOLT | `vortexgpgpu/Volt` / llvm `vortex_3.x`（LLVM 20.1.8） | M1/M3 需要 VOLT 侧改动（HIP 内建、intrinsic 暴露），单开 PR 上游化 |
| PoCL / chipStar / Mesa | `vortex_3.x` | 过渡期维护；M1 完成后 chipStar 降级为可选 |
| Triton | **pin 一个 minor（建议 3.3/3.4 系）** | Triton backend API 在 3.x 间漂移大，锁版本 + vendor 补丁目录 |
| PyTorch | pin 2.x（建议 2.5–2.7 之一，全测试通过后升） | PrivateUse1 与 inductor 接口随版本演进 |
| Python / numpy | 3.10+ / 1.26+ | 与 PyTorch pin 匹配 |

### 2.5 一个贯穿全项目的架构决策：warpSize

CUDA/HIP/Triton/CUB 生态默认 warpSize=32（lane id 语义、ballot 位宽、`__shfl_sync` 掩码、Triton 的 layout 计算）。Vortex NT 可配（4..32）。**决策建议**：

- **DL 档统一 NT=32**（TCU 几何也最大：tcM=8, tcN=tcK=4→per-warp WGMMA M=16、K=8、NRC=32 时 N=64）；`warpSize` 通过 `hipDeviceProp.warpSize=32` 与 `VX_CAPS_NUM_THREADS` 一致暴露。
- 算子库所有 block 级原语**模板参数化 WarpSize**（rocPRIM 移植时本来就要改的点），对 NT=16 提供编译分支，便于小配置跑功能测试。
- CI 双档：`NT=32`（生态兼容档，G0–G2 全跑）+ `NT=16`（ASIC 门禁已有的核档，只跑功能子集）。

---

## 3. M1：原生 HIP（HIPVortex）

**目标**：`hipcc` 直接把 HIP C++ 编成 Vortex vxbin（AOT，不经 SPIR-V/PoCL）；主机侧 `libhip_vortex` 直接落 `vortex2.h`；HIP 设备头暴露 Vortex 全部加速内联指令。完成标志：`tests/hip` 四例 + 新增 HIP smoke 套件全绿（simx + rtlsim），且 `hip::wmma` 能从 HIP 源码跑通 sgemm_tcu_wg 等价 GEMM。

### 3.1 `libhip_vortex`：HIP Runtime API → vortex2.h 映射

| HIP API | Vortex 实现 | 备注 |
|---|---|---|
| `hipInit` / `hipGetDeviceCount` / `hipSetDevice` / `hipDeviceGet` | `vx_device_count` / `vx_device_open` / 引用计数 | 惰性初始化；进程内单实例 + 设备表 |
| `hipGetDeviceProperties` | `vx_device_query(VX_CAPS_*)` 逐字段填 | warpSize=`NUM_THREADS`、multiProcessorCount=`NUM_CORES`、`regsPerBlock`=32×NT×NW、sharedMemPerBlock=`LOCAL_MEM_SIZE`、clockRate、ISA flags → `arch` 字符串（建议 `vortex-rv64-tcu` 风格自定 id） |
| `hipMalloc` / `hipFree` / `hipMemset` | `vx_buffer_create/release` + `vx_enqueue_fill_buffer` | 内存地址即 `vx_buffer_address`（UVA 模型：设备地址数值直接可放进 kernel 参数，与现有 launch ABI 一致） |
| `hipMemcpy` / `hipMemcpyAsync` / `hipMemcpy2D/3D` | `vx_enqueue_copy/read/write`（含 `_rect` 变体） | H2D/D2H/D2H 全走 CP DMA；kind 判定：指针落在已登记区间 |
| `hipHostMalloc` / `hipHostRegister` / `hipHostGetDevicePointer` | `VX_MEM_PIN_MEMORY` / `VX_MEM_HOST` 分配 + 地址登记表 | `VX_MEM_HOST`（CP 可见主机孔径）语义上最接近 zero-copy；`hipHostRegister` 首版可返回不支持 |
| `hipStreamCreate(WithPriority)` / `hipStreamDestroy` / `hipStreamSynchronize` / `hipStreamQuery` / `hipStreamWaitEvent` | `vx_queue_create` / `vx_queue_finish` + 事件 wait-list | **注意**：设备端单 CP 队列 → 多流当前实际串行；优先级透传 `vx_queue_info_t.priority`；语义先对齐（正确性），并发留 §9 P1 |
| `hipEventCreate(WithFlags)` / `hipEventRecord` / `hipEventSynchronize` / `hipEventElapsedTime` / `hipEventQuery` | `vx_event_*`（时间线信号量模型）+ `vx_event_get_profiling` | binary event = signal 1/wait 1；elapsed 用 profiling 时间戳 |
| `hipModuleLoad(Data)` / `hipModuleGetFunction` / `hipModuleLaunchKernel` / `hipLaunchKernelGGL` / `<<<>>>` | `vx_module_load_bytes(file)` / `vx_module_get_kernel(name)` / `vx_enqueue_launch` | fatbin 里嵌的就是 vxbin（见 3.2）；args 打包成 blob（`vx_launch_info_t.args_host`，4KB 上限首版声明为 `HIP` 限制并文档化） |
| `hipFuncGetAttributes` / `hipOccupancyMaxActiveBlocksPerMultiprocessor` | `vx_kernel_get_max_block_size` / `vx_device_max_occupancy_grid` | |
| `hipDeviceSynchronize` | 所有队列 `vx_queue_finish` + 默认 device 同步 | |
| `hipMallocManaged` | 首版不支持（返回 `hipErrorNotSupported`） | 统一内存远期（§9 P2，MMU 已有 SV32/SV39 基础） |
| `hipMallocAsync` / memory pool | 首版不支持 | M4 PyTorch caching allocator 不依赖它 |
| `hipError_t` / `hipPeekAtLastError` / `hipGetLastError` | thread_local 错误槽 | |
| `hipPrintf` / `hipassert` | COUT 环（`VX_MEM_IO_COUT_ADDR`，有损语义与 CUDA 一致） | 设备侧直接用 `libvortex2.a` 的实现 |
| `hipDeviceCanAccessPeer` / P2P | 不支持（单卡） | 多卡远期 |
| Cooperative launch / `hipLaunchCooperativeKernel` | 可映射到 KMU CTA cluster（`cluster_dim`）+ `gbarrier` | 首版可只支持"grid 同步 = 全局屏障"的受限形式 |

工程形态：

- **仓库**：新建 `~/vortex/hip/`（或独立仓 `vortexgpgpu/hipvortex`，仓库布局对齐 SDK：`pkg-config vortex-runtime.pc` 消费 `$VORTEX_PATH`）。产物 `libhip_vortex.so` + 头 `hip/hip_runtime.h`（API 面向 chipStar/ROCm 的 HIP 头对齐，保证应用源码级兼容）。
- **线程模型**：沿用 `vortex2.h` 的每队列 worker 线程；HIP 层只做句柄封装与错误翻译，不引入自己的线程。
- **多设备**：`vx_device_open(index)` 支持多实例；`hipSetDevice` 线程局部当前设备（对齐 CUDA 语义）。

### 3.2 `hipcc-vortex`：编译驱动与 fatbin

编译管线（全部复用现有脚本，只是从 PoCL 的环境变量桥接改为 hipcc 内置）：

```
app.cpp (HIP C++)
  │ hipcc-vortex  --offload-arch=vortex-rv64
  ├─ host:   系统 clang/g++ 编 host ELF（内嵌 fatbin 节）
  └─ device: VOLT clang
        --target=riscv64-unknown-elf --sysroot=$VORTEX_PATH/...
        -march=rv64imafdc -mabi=lp64d  -Xclang -target-feature -Xclang +xvortex +zicond
        -mcmodel=medany -O3  -D__HIP_ARCH_VORTEX__=1
        → .o → ld.lld -T sw/kernel/scripts/link64.ld + libvortex2.a + libhip_device.a + musl libc/libm + libclang_rt.builtins
        → vxbin.py → kernel.vxbin（多入口 VXSYMTAB：__vx_kentry_<name>）
  fatbin 容器：自定义 `VXFATBIN`（magic + [name, offset, size, pc] 表）或直接把 vxbin 塞进 ELF 自定义节 `__vortex_fatbin`
  hipModuleLoadData → 从内存解包 → vx_module_load_bytes
```

要点与坑（来自现有管线的既有约定，`docs/building_toolchain.md`）：

- **feature 名是 `+xvortex` 不是 `+vortex`**；不要给 clang 设 RISC-V 默认 triple（会破坏 host 编译）。
- crt0 两遍编译（`kernel_startup.sh` 按 `.tdata/.init_array` 决定 `-DNEED_TLS/-DNEED_INITFINI`）必须保留，hipcc 里集成该逻辑或直接复用脚本。
- `kernel_main` 会被 vxbin.py 重命名为 `main`——HIP 多 kernel 场景依赖 VXSYMTAB 多入口，hipcc 侧要保证每个 `__global__` 都产 `annotate("vortex.kernel")`（改头文件宏：`#define __global__ __attribute__((annotate("vortex.kernel"), used)) extern "C"`——`vx_spawn2.h` 已有先例）。
- hipcc 壳本身建议从 ROCm 的 hipcc.py 抽骨架（它已处理 `--offload-arch`、`-std`、链接参数分流），把 target 检测改为 vortex 分支；这比自写更能跟随 HIP 生态的编译参数演进。

### 3.3 `libhip_device`：设备头与内联指令暴露面

| HIP 设备 API | Vortex 映射（`vx_intrinsics.h` 等已有） | 备注 |
|---|---|---|
| `__syncthreads()` | `vx_barrier(get_local_group_id(), get_num_sub_groups())` | `vx_spawn2.h` 同款 |
| `__syncwarp()` | `vx_wsync()` | |
| `__shfl_{up,down,bfly,idx}_sync` | `vx_shfl_{up,down,bfly,idx}` | 掩码参数在 Vortex 无意义（全 warp 参与）→ 忽略或断言 full mask |
| `__ballot/__all/__any/__unary` (`__all_sync`...) | `vx_vote_ballot/all/any/uni` | 返回类型：NT<32 时 ballot 用 `uint32_t` 高位补零 |
| `__threadfence()` / `__threadfence_block()` | `vx_fence()`（`fence iorw,iorw`） | |
| `atomicAdd/Sub/Min/Max/And/Or/Xor/Xchg/CAS` | RVA `amo*`（经 clang 内建或内联 asm 封装） | **依赖 `EXT_A_ENABLE`**（DL 档默认开）；LLC 顺序一致 → 语义强于 CUDA，安全 |
| `__expf/__logf/__exp2f/...` | libm（musl）标量软件实现 | 无 SFU；提供 `__expf` → `expf` 直通（后续可加多项式快速版，见 §9 P1） |
| `__hmul/__hadd/half2` 等半精度 vector | 首版走 fp16↔fp32 转换 + FPU | TCU 才有原生 fp16 数据通路；给 `half`/`__nv_bfloat16` 类型定义（存取/转换内建即可） |
| `hip::wmma`（fragment/mma_sync） | `vortex::tensor::wmma_context`（API 形状已刻意对齐 CUDA wmma） | 直接包一层命名空间适配 |
| `hip::wmma` 的 wgmma / 异步（无 CUDA 标准 API） | 自定义扩展头 `hip/vortex_mma.h`：`wgmma_context`、`vx_dxa_issue_*_wg`、`vx_barrier_expect_tx` | **这是 chipStar 路线做不到的核心增量**：让 HIP 源码能写 Hopper 风格的 smem-descriptor WGMMA + DXA 双缓冲 GEMM |
| `hipTensorMap`（TMA 等价） | DXA 描述符（主机 `dxa.h` DCR 编程 + 设备 `vx_dxa_issue_*`） | |
| `__shared__` | LMEM：`__local_mem()` 基址 + 静态分配（链接脚本 `.lmem` 段或编译器 annotate） | **缺口**：目前共享内存没有语言级 `__shared__`；首版用显式 API（`vortex::smem<T>` 从 CTA LMEM 划分），中期在 VOLT 加 address space 或 annotate 支持（见 3.6） |
| `__constant__` | vxbin 数据段（RO），经 kernel 参数传指针 | PoCL 已有等价机制可参考 |
| `printf` | `vx_printf`（有损环，语义对齐 CUDA） | 已有 |
| cooperative groups | `cta_cluster` CSR + `group_barrier` | 子集：thread_block / thread_block_tile<WarpSize> / cluster |

### 3.4 需要在 vortex2.h / 运行时补的缺口（上游 PR 清单）

1. `vx_kernel_set_arg`（逐参数设置，替代一次性 blob）与 `vx_event_set_callback`（PoCL 已提过需求，`docs/designs/vortex_runtime_api.md` §3 明确 deferred）——HIP `hipSetupArgument`/PyTorch lazy loader 需要。
2. **内核参数 >4KB**（当前 scratch 槽 4KB）：PyTorch 风格大参数（多指针 + strides）通常 <1KB，够用；但要把"超限"变成显式错误码而非静默截断。
3. `vx_buffer_export_to_host` / host-mapped pointer 查询（`hipPointerGetAttributes` 需要）。
4. 内存池 API（可选；PyTorch 自带 caching allocator，可不急）。
5. 多队列并发（§9 P1 硬件项的软件前置：运行时已支持多 `vx_queue`，落硬件后即多流）。

### 3.5 测试与验收

- **API 单测**（新增 `tests/hipvortex/` 或 `tests/hip/native/`）：每 API 家族一个 smoke（malloc/copy/memset/stream/event/module/launch/printf/occupancy/error）。
- **chipStar conformance 复用**：把 `vortexgpgpu/chipStar` 的 `known-failures-vortex32.txt` 逐条转成 `libhip_vortex` 的过/挂记录——目标是把 rv32 ~36% 提到 >90%（原生路径绕开 SPIR-V 尺寸假设后大部分应转绿）。
- **内联指令暴露验证**：`sgemm_tcu_wg{,_mx,_sp,_dxa}` 家族用 HIP 源码重写一遍（`hip/vortex_mma.h`），与 `tests/regression` 原生版对拍周期数。
- CI：新增 `ci/testcases/hip_native.yaml`（tier=smoke，drivers=simx,rtlsim，xlen 32/64）。

### 3.6 工作量、风险与上游协同

- 工作量：**4–6 人月**（runtime 1.5–2，hipcc 壳 + fatbin 1–1.5，设备头/文档/CI 1–1.5）。
- 风险：HIP 头版本选择（建议直接对齐 chipStar 的 hip_runtime.h 子集，避免追 ROCm 全量）；`__shared__` 语义缺语言级支持 → 显式 API 过渡要写进用户文档，避免用户拿 CUDA 源码直接编译预期无缝。
- VOLT 上游协同：`__global__→annotate` 映射、（中期）`__shared__` address space 或 `annotate("vortex.smem")` + LMEM 分配 pass；这两项是 HIP 源码级兼容度的决定因素。

---

## 4. M2：基础算子库

**目标**：建立 PyTorch/模型层之下的"算力供给层"——并行原语、GEMM、卷积、随机数、attention。完成标志：fp16 GEMM 在 DL 档达到 G1 的利用率指标；conv2d/bn/pool/softmax/reduce/scan/sort/topk 数值全绿；Philox 与 torch 对拍一致。

### 4.1 `vortex-prim`（对标 rocPRIM/hipCUB）

- **API**：block 级（`block_reduce`/`block_scan`/`block_radix_sort`/`block_exchange`——全部模板参数化 `WarpSize`）与 device 级（`device_reduce/scan/segmented_reduce/radix_sort/merge_sort/merge/binary_search/transform`）。
- **移植策略**：拿 TheRock 里 rocPRIM 源码作语义参考（其测试用例直接复用为验收），实现换底到 `vx_intrinsics.h`（vote/shfl/wgather）。重点改造点：
  - warpSize 常量 → 模板参数（rocPRIM 硬编码 32 处不少）；
  - warp 同步原语 → `vx_wsync`/`vx_vote`/`vx_shfl`；
  - 共享内存 → 显式 LMEM 分配 API；
  - `__syncthreads_or` 等 CUB 扩展 → `vx_vote_any` + barrier 组合。
- **排序种子**：`tests/regression/sort` 已有基础实现，作为 radix sort 的起点。
- **关键实现注意**：LMEM bank 数 = `NUM_LSU_LANES`（非 CUDA 32 bank 模型）——bank 冲突避免的 swizzle 策略要按 Vortex 几何重新推导，写成 `vortex::swizzle` 工具头供全栈复用。

### 4.2 `vortex-blas`（GEMM 模板库，对标 CUTLASS/hipBLASLt）

这是整个 DL 栈性能的**心脏**，也是最应该投入的地方。

**架构**（CUTLASS 三段式，全部 C++ 模板 + HIP 设备头）：

```
TilePolicy<M_,N_,K_, stages>            // tile 形状与双缓冲级数（受 LMEM 约束）
Mainloop<Policy, LayoutA/B, dtype>      // DXA k-major 预取 → WGMMA(SS) 流水
  ├─ dxa_issue_2d/3d(barrier_id, desc)  // 异步搬运 + expect_tx 放行
  ├─ wgmma_sync(D, desc_a, desc_b, C)   // smem 描述符路径（AM DMA 口直供 tile buffer）
  └─ producer/consumer 双缓冲（warpgroup 分工或 warp specialization）
Epilogue<Bias, ActFn, Residual, Quant>  // 偏置/GELU/SiLU/residual/量化 scale 融合
```

- **现有种子**：`tests/regression/sgemm_tcu_wg*`（wgmma + MX + SP + DXA + 多播全家桶）、`sgemm2_tcu`——这些就是 TilePolicy 的原型，抽出来库化。
- **必做形态**：NN/NT/TN/TT、split-K、batched（strided）、grouped（MoE 用）、GEMV/GEMM 混合 dispatch（LLM decode 是 skinny GEMM）、NT=32 与 NT=16 双档 tile 表。
- **API**：hipBLASLt 风格（`vortexLtMatmul(descA, descB, descC, algo, workspace...)` + epilogue 枚举）——这是 PyTorch `aten::mm/bmm/addmm` 与量化 kernel 的直接落点；hipBLAS 兼容薄层可后置。
- **Autotuner**：tile/stages/split-k 枚举 + 运行时实测选优（simx 周期数或 `vx_rdcycle_sync` 设备侧计时）；结果按 (shape, dtype, arch) 缓存。首版离线表即可，在线版挂到 Triton autotune 一起做。
- **数值注意**：TFR 是定点归约树——与 IEEE 逐加参考存在可容许差异，验收标准用相对误差（fp16 累加 fp32：rtol 1e-2 / atol 1e-3 量级，具体以 golden 对拍定档并写死进 CI）。

### 4.3 `vortex-dnn`（卷积与常规 DL 算子）

| 算子族 | 方案 | 优先级 |
|---|---|---|
| conv2d/conv3d fwd | L1: im2col + GEMM（复用 blasLt，NHWC 布局优先——TCU 喂数友好）；L2: direct conv（winograd 3×3 可选后置） | P0 |
| conv1d（causal，Mamba 用） | 直接滑窗 kernel（专门写，见 §8.5） | P1 |
| conv 反传（dgrad/wgrad） | im2col 反向 + GEMM（训练才需要；推理栈 P2） | P2 |
| batchnorm（inference fold）/ layernorm / rmsnorm | 行归约 + scale-shift 融合；rmsnorm 是 LLM 必需（P0） | P0 |
| pooling（max/avg/adaptive） | 简单 kernel | P0 |
| softmax / logsoftmax | 已有 `tests/regression/softmax` 种子；行 max→exp→归约，exp 走 libm（软肋，见 §9 P1） | P0 |
| activation：relu/gelu/silu/sigmoid/tanh | 元素级；gelu 用 tanh 近似或 erf 表 | P0 |
| reduction：sum/mean/max/min/norm | `vortex-prim` device_reduce 封装 | P0 |
| interpolate：nearest/bilinear/bicubic/upsample | YOLO/SAM 必需 | P0 |
- embedding（gather）、scatter/index_add（需 AMO）、one-hot | gather/scatter 原语 | P0 |
| topk / sort / argsort | radix topk（prim 之上） | P1 |
| NMS（iterative/soft） | YOLO 必需；block 并行 IoU + 状态机迭代 | P1 |
| ROI align / grid_sample | SAM（`grid_sample`）/检测头 | P1 |
| dropout / RNG kernel | `vortex-rand` Philox | P1 |
| cumsum / prefix scan | prim 的 scan 封装 | P1 |

（P0 = resnet50/yolo/llm 推理跑通所需最小集；P1 = SAM/DINOv3/Mamba/训练补齐；P2 = 训练全量。）

### 4.4 `vortex-rand`（对标 rocRAND）

- Philox4x32-10 为主（与 PyTorch CUDA 生成器算法一致，便于 checkpoint 复现），Threefry 备选；curandApi 风格 device API（`philox_state`、`rand_uniform/normal`）+ host API（seed/offset 管理，供 PyTorch Generator 对接）。
- 种子：`tests/regression/dropout` 的 WangHash 只作占位，替换为 Philox。

### 4.5 `vortex-attention`

- **flash attention（前向）**：分块 Q×K^T → 在线 softmax（两遍 max/rescale 流水）→ ×V；tile 大小受 LMEM（K/V tile）与寄存器（Q×O tile）双约束；exp 软肋同上。
- **paged attention（decode）**：KV cache 块表（page size 16/64）+ GEMV 形 attention；LLM 专用，见 §8.4。
- 实现语言直接用 HIP C++（M2 期），Triton 就绪后重写为参考对照。

### 4.6 通信（现状：单卡）

- 卡内：跨核 `gbarrier` 已有；DXA 多播可当"共享输入广播"用。
- 多卡：`tests/mpi` 已验证多 FPGA/主机 MPI 流程；RCCL 级集合通信为远期（§9 P3）。DDP 训练首版走 CPU gloo（§6.5）。

### 4.7 验收与基线

- 数值：CPU 参考实现（同一份 C++ 源码 host 编译）或 PyTorch CPU golden；容差策略统一（fp32 1e-5、fp16 1e-2 相对，量化另计）。
- 性能：`ci/roofline.py` + MPM（`STALL_TCU`、`TBUF_STALLS`、`LMEM_READS`、mem 带宽计数）自动出利用率报告；每算子 perf_gate 基线入库（沿用 ±2% 机制）。
- 新增 CI 目录：`ci/testcases/{prim,blas,dnn,rand,attention}.yaml`。

---

## 5. M3：Triton 接入

**目标**：`triton.jit` 内核编译到 Vortex 并经 driver shim 启动；`tl.dot` 走 WGMMA、`num_stages` 双缓冲走 DXA + expect_tx。完成标志：Triton 官方 tutorials 的 vector-add / fused-softmax / matmul 全绿，matmul 性能与手写 `vortex-blas` 同档（±20%），PyTorch inductor 能用该后端编译出可跑 kernel（衔接 M4）。

### 5.1 为什么 Triton 值得做（以及什么时候做）

- 价值：**PyTorch 2.x 的性能层事实标准**。inductor 自动生成的融合 kernel、FlashAttention/Triton 版 GEMM、torchao 的量化 kernel 都是 Triton。没有 Triton 后端，PyTorch 后端只能停留在"手写算子库"的 eager 形态。
- 时机：M1（driver API 落定）之后即可启动，与 M2 并行；M4 的 torch.compile 路线以它为前提。

### 5.2 Triton 后端插件结构（以 Triton 3.3/3.4 为基线）

Triton 通过 `triton/backends/<name>/{compiler.py,driver.py}` + CMake 源码目录组织后端（nvidia/amd 为参考实现）。新增 `third_party/vortex/`：

```
triton/third_party/vortex/
├── backend/
│   ├── compiler.py     # VortexTarget(GPUTarget('vortex', arch, warp_size))
│   │                   # VortexBackend(BaseBackend):
│   │                   #   parse_options（num_warps/num_stages/共享内存上限→lmem_size）
│   │                   #   compile: ttir→ttgir→llir→llvm→obj→vxbin（见 5.3）
│   │                   #   get_attrs/JSON 化 cache key
│   └── driver.py       # VortexDriver: active driver 探测、VortexUtils（≈CudaUtils）
│                       #   launcher：参数打包→libhip_vortex hipModuleLaunchKernel
├── python/…/vortex/    # tl 扩展（如 vortex.extra::wgmma descriptor 实验 API）
└── CMakeLists.txt
```

`GPUTarget` 关键参数从 `vx_device_query` 来：`warp_size=NUM_THREADS`、共享内存上限=`LOCAL_MEM_SIZE`、`arch` 字符串带 dtype 能力位（供 `tl.dot` 输入 dtype 合法性检查）。

### 5.3 编译管线与 lowering 映射（核心工程）

| Triton 层 | 现状（NVIDIA/AMD 路径） | Vortex 映射方案 |
|---|---|---|
| TTIR→TTGIR（layout 推导） | `blocked` / `mma` / `dot_operand` / `shared` / `slice` 等布局 | `blocked`：lane = tid 映射不变；`mma`：映射到 WGMMA fragment 布局（NRA=4/NRC 变量寄存器窗，由 `wgmma_context` 的布局公式给出，需在 `make_ttgir` 里写 Vortex 版 `mmaVersion`/`getMMAVersion` 等价物）；`dot_operand`：A 寄存器路径（RS）或 smem 描述符路径（SS）；`shared`：LMEM 地址 + `smem_matrix_desc`（含 stride=0 block-major 判断） |
| `tl.dot` | → `mma.sync` / `mfma` | → **WGMMA（SS 优先，RS 用于 A 常驻场景）**；K 维按 `xtileK` 步进（FEDP2K 开启时翻倍）；累加 fp32；输入 dtype 检查 = TCU 使能位 |
| `num_stages` 流水（async copy） | `cp.async` / TMA | → **DXA issue（2D/3D 描述符）+ `vx_barrier_expect_tx` + arrive/wait**——Vortex 的异步屏障事务计数语义与 `cp.async.commit_group/wait_group` 一一对应，映射干净 |
| shared memory 分配 | `.shared` 地址空间静态/动态分配 | → LMEM：kernel 编译期算出总需求 → 装进 `vx_launch_info_t.lmem_size`；LMEM≤64KB 硬上限写进 `parse_options`（超出直接报错并建议减小 BLOCK/stages，**不静默**） |
| warp 原语 | `llvm.nvvm.shfl` 等 | → inline-asm/C 内建调用 `vx_shfl_*`/`vx_vote_*`/`vx_wsync`（Triton 的 LLIR 层允许发射任意 LLVM IR，inline asm 最稳，不需要等 VOLT 加 intrinsic） |
| barrier | `barrier` PTX / `s_barrier` | → `vx_barrier`（CTA id 嵌槽号语义由运行时保证） |
| 原子 | `atom.global.add` | → RVA `amo*`（inline-asm） |
| math (`libdevice`：`__nv_expf`...) | NVVM libdevice bc | → **`vortex_libdevice`**：把 `exp/log/tanh/erf/pow/rsqrt...` 映射到 musl libm（或自写快速多项式）；以 LLVM IR/bitcode 形式提供，接口签名对齐 Triton 的 `math` dialect 调用约定 |
| `tl.load/store` | global ld/st + cache hint | → 普通 LSU 路径（无 cache hint 内联指令，先忽略 hint 参数）；coalescing 由编译器现有合并 + dcache bank 处理 |
| launch 元数据 | cubin/hsaco + metadata json | → **vxbin（VXSYMTAB）+ Triton metadata json**（name、shared_mem=等价 lmem_size、num_warps） |

**对象文件与链接**：Triton 在内存里拿到 LLVM IR module 后，调 VOLT 的 `llc/clang`（triple=`riscv64-unknown-elf`，features=`+xvortex`）→ `.o` → `ld.lld -T link64.ld`（带 `-e` 入口 stub）→ `vxbin.py` → bytes。可执行形态直接复用 `miscs`/`sw/kernel/scripts` 现有脚本，做法与 vortexpipe 完全同构。

**关键难点（提前排雷）**：

1. **WGMMA 是 warp 级/描述符驱动的**，不是 NVIDIA 那种 warp-collective 内联；Triton 的 mma layout → fragment 寄存器映射要按 `wgmma_context` 的 per-warp 几何精确生成（`xtileM=2·tcM` 等），错一位就是静默数值错。对策：先做 **RS 路径**（A/B 都从寄存器 fragment 进），再上 SS（描述符）路径；每一步用 Triton matmul vs `vortex-blas` 对拍。
2. **单 CTA 锁步门**（`VX_tcu_lockstep`）：同一时刻 tile buffer 只归属一个 CTA 的 warpgroup——Triton 的 persistent kernel / 多 warpgroup 模式要保证 CTA 内串行 WGMMA，mapping 时不要跨 CTA 交错发射。
3. **warpSize≠32 的地方**：Triton 3.x 大多已参数化 `WARP_SIZE`，但 layout 计算与某些 reduction（`tl.reduce` 的跨 warp 树）假设偶发；锁 NT=32 后此类问题退化为常数，另一个 NT=16 档做编译期断言。
4. **LLVM 版本对齐**：Triton 自带 LLVM（3.3 ≈ LLVM 20 系），VOLT 也是 20.1.8——版本接近，直接把 VOLT 当目标编译器用是本方案可行性的运气所在；若 pin 的 Triton 用了更新 LLVM 语法，降级路径是"Triton 停在 LLVM IR → 由 VOLT 的 llc 编"（IR 版本兼容性需在 CI 里锁死）。
5. **num_warps 与 CTA 形状**：Triton block = `num_warps × warp_size` 线程 = 1 个 CTA（`BLOCK_SIZE = num_warps` 个 Vortex warp）；DL 档 NW=NT=32 时单 CTA 最大 1024 线程，覆盖 Triton 常用 4–8 warps。

### 5.4 Driver shim 与 Python 侧

- `driver.py` 的 launcher：把 grid 计算、kernel 参数（含 tensor data_ptr）打包成 blob → `hipModuleLaunchKernel`（走 M1 的 `libhip_vortex`，从而复用其队列/事件语义）；`torch.cuda` 风格的 stream 从 PyTorch 侧透传（M4 衔接点）。
- `VortexUtils`：`getDeviceProperties`、`loadBinary`（vxbin bytes + metadata → module/kernel 句柄缓存）、时钟/SM 数查询。
- autotune：`@triton.autotune` 直接可用（其计时走 stream event）；autotuner cache 目录按 arch+shape 命名。

### 5.5 与 PyTorch Inductor 的衔接（详细留给 M4）

inductor 的 Triton codegen 输出标准 Triton 源 → 经本后端编译 → launcher 由 inductor 运行时调用。需要适配的点集中在 `torch._inductor.runtime` 的设备探测与导入逻辑（`== "cuda"` 的分支），以及 pinned memory / stream 语义对接——M4 一并处理。

### 5.6 验收、工作量与风险

- 验收：tutorials 01/02/03/05（vecadd/softmax/matmul/低配 fused-attention）+ `tests/triton/` 新目录（每个 tutorial 一个 CI 用例，simx 数值 + rtlsim 抽样周期）+ matmul autotune 曲线报告。
- 工作量：**6–10 人月**（TTGIR→LLVM lowering 3–4，libdevice/运行时/launcher 1–2，调试与数值对齐 2–3）。全项目最大编译器单项，建议由有 MLIR/Triton 经验的人牵头。
- 风险：Triton 上游 API 漂移（pin 版本 + 定期重定基）；WGMMA layout 对齐的静默数值错（用"先 RS 后 SS、逐步对拍"缓解）；LLVM IR 兼容（CI 锁死）。

---

## 6. M4：PyTorch 后端（torch-vortex）

**目标**：`torch.randn(..., device="vortex")` 可用；resnet50/yolo/sam/dinov3/mamba/7B-LLM 在 eager 模式端到端推理通过；`torch.compile` 分阶段接入。不 fork PyTorch 主干（或只保留最小 patch 集）。

### 6.1 路线：PrivateUse1（确定不走 fork-rocm）

PyTorch 官方自定义设备通道 `PrivateUse1`（昇腾/寒武纪/摩尔线程早期同款路线）：

- `torch.utils.rename_privateuse1_backend("vortex")` → `device="vortex"`；
- C++ 侧注册 dispatch key `PrivateUse1` 的 kernel；
- Python 侧 `torch.utils.generate_module_for_privateuse1_backend("vortex")` 自动生成 `torch.vortex.*` 模块（empty/rand/stream/event/save/load...）。

对比 fork rocm 后端：需要伪造一整套 ROCm 运行时符号、跟死 PyTorch release 节奏、连 RCCL/MIOpen 依赖一起扛——否决。PrivateUse1 的代价是 autograd/compile 的部分机制要自己接线，但都在公开 API 面上，可控。

### 6.2 组件清单（工程顺序）

| 组件 | 内容 | 依赖 |
|---|---|---|
| **Allocator** | `c10::GetAllocator(kPrivateUse1)`：进程级 caching allocator——按 2MB 段向 `vx_buffer_create` 批量要内存，内部 size-bucket 空闲链；**stream-aware 延迟释放**（块标 last-use stream，释放前等该 stream 事件，避免用后释放）；OOM → `empty_cache()` 重试；`MemoryStats` 填全（torch.cuda.memory_summary 可用） | M1 |
| **Device/Stream/Event** | `c10::impl::DeviceGuardImplInterface`；`c10::Stream` → `hipStream_t` → `vx_queue`；Event → `vx_event`；`record_stream()` 语义接 allocator | M1 |
| **基础 Tensor 事实** | `at::empty/zeros/ones/full/copy_`（D2D/D2H/H2D）；打印/类型提升默认继承 ATen 通用逻辑 | Allocator |
| **Dispatch 骨架** | 注册 `PrivateUse1` 的 `copy_`/`item`/`to`；**CPU-fallback 通用包装**：未实现算子自动搬运到 CPU 执行再搬回（`torch.library` 的 fallback 机制）——保证"任何模型都能跑（慢）"，把功能可用日从算子全绿日大幅提前 | 基础 Tensor |
| **核心算子 kernel** | 从 §4.3 P0/P1 表逐个把 ATen 算子接到 `vortex-*` 库（kernel 用 hipcc 编 vxbin，多入口模块静态注册，进程启动时 `vx_module_load_bytes` 一次加载全部符号） | M2 |
| **Autograd** | 大部分算子注册 `CompositeImplicitAutograd`（ATen 自动微分公式分解到前向原语）；少量手写 backward（conv/softmax 类的稳定版本） | 核心算子 |
| **RNG** | `c10::GeneratorImpl`（vortex 版）：Philox（`vortex-rand`）算法 + seed/offset 状态；`torch.manual_seed` 设备语义；与 CUDA 版同算法保证数值可复现 | M2 rand |
| **序列化** | `torch.save/load`：tensor 存储 blob + device 标记（PrivateUse1 官方路径）；可选 DLPack 导出 | 基础 Tensor |
| **pinned memory** | `torch.Tensor.pin_memory()` → `VX_MEM_PIN_MEMORY` 分配；dataloader `pin_memory=True` 生效 | M1 |
| **NN 模块** | `torch.nn.functional` 大多自动走 ATen；conv/bn 的 cudnn 等价路径不需要（`vortex-dnn` 已在 ATen 之下） | 核心算子 |

### 6.3 算子接入的三级火箭

1. **L0（第 1 周起可用）**：CPU-fallback——全模型可跑，速度仅够功能调试与 golden 对拍。
2. **L1（P0 算子 ~100 个）**：conv/gemm/bn/act/norm/pool/embedding/gather/softmax/reduce/interpolate 接原生 kernel；此时 resnet50、yolo、llm eager 推理达到"可用"。
3. **L2（torch.compile）**：
   - 阶段 a：`torch.compile(backend="aot_eager")`（纯 ATen 图执行，无 inductor）——免费拿到图消除收益；
   - 阶段 b：**inductor + Triton（M3 就绪后）**：需要小 patch 集——`torch._inductor.runtime` 的设备探测/导入分支、`VortexConfig`（inductor 的 backend config 对象）、wrapper 代码生成中的 stream/pin 语义。上游有非 cuda 后端接入 inductor 的先例（xpu 等），patch 面可控但要锁 PyTorch 版本；
   - 阶段 c（可选）：`torch.export` + AOTInductor 产离线制品（部署形态，服务 ASIC 卡发行）。

### 6.4 多进程与分布式

- 单卡起步；DataLoader 多进程（CPU 侧）天然可用。
- DDP：gloo 后端 + gradient allreduce 在 CPU（NHWC 参数搬回主机）——只服务"能训"的演示级；真正的多卡集合通信见 §9 P3。

### 6.5 构建与发布形态

- 仓库 `vortexgpgpu/torch-vortex`（setup.py 扩展包，不进 PyTorch 源码树）；CI 产出 wheel（Python 3.10/3.11 × torch pin × xlen 64）；依赖 `$VORTEX_PATH`（pkg-config）+ `libhip_vortex`。
- 环境变量：`VORTEX_DRIVER` 透传（simx 调试 / rtlsim 周期 / aved 实卡同一份 wheel——**这是 Vortex 多后端架构对 PyTorch 层的免费馈赠**：开发者本地 simx 跑 CI，CI 上 rtlsim，板上 aved）。

### 6.6 验收

- 功能：`tests/pytorch/`——每模型一个 harness（torchvision resnet50、ultralytics yolov8、segment_anything、dinov3、mamba-ssm、HF llama-7b 伪码小模型 + 真权重小模型 tinyllama），CPU 对拍（atol/rtol 表格化）。
- 性能：resnet50 推理延迟、LLM decode tokens/s，G1/G2 指标；`torch.profiler` 时间线（用 `vx_event_get_profiling` 支撑）。
- CI：`ci/testcases/pytorch.yaml`（simx 全量 / rtlsim 抽样 / 板上 nightly）。

---

## 7. M5：量化支持

**目标**：覆盖 W4A16 / W8A8(INT8) / W8A8(FP8) / MX-FP8/FP4 / NVFP4 / 2:4 稀疏的推理路径；量化模型（LLaMA 级 LLM 与 ViT）端到端精度达标（PPL 损失阈值见 §7.5）。

### 7.1 硬件能力矩阵与使能顺序

RTL 已实现（`VX_tcu_pkg`/`tensor_cfg.h`）、默认全关（`VX_config.toml [tcu]`）：

| dtype | TCU 支持 | 建议使能波次 | 前置验证 |
|---|---|---|---|
| FP16（含 BF16） | ✅ 默认开 | 0（现状） | 清 XLEN=64 wgmma fp16 rtlsim 数值 bug、tf32 周期计数 bug |
| **INT8（i8/u8→i32）** | ✅ 关 | **1** | `sgemm_tcu` int8 变体 + golden；对齐 `fedp.py` 的宿主 PyTorch 对拍方法 |
| FP8 e4m3 / BF8 e5m2 | ✅ 关 | 2 | 同上 + 缩放因子策略 |
| INT4 / FP4（纯 i4/u4） | ✅ 关 | 3（多数场景被 MX/NVFP4 取代） | |
| MXFP8 / MXBF8（32 元素块缩放） | ✅ 关 | 3 | `sgemm_tcu_mx` 已有测试底座；TCU_LD 预载 scale SRAM 路径 |
| MXFP4 / NVFP4（16/32 块缩放） | ✅ 关 | 3 | `sgemm_tcu_sp_mx` 底座 |
| 2:4 结构化稀疏 | ✅ 关 | 4 | 已实测 1.86× 周期加速（§1.1.2）；host 侧 `compress_2to4_matrix` 已有 |
| TF32 | ✅ 关 | 可选 | |

**每波次流程**：`CONFIGS` 打开 → simx 数值 → rtlsim 数值+周期 → `ci/baselines/perf` 基线 → FPGA/ASIC 门禁（面积/频率代价记录进 asic_gate）→ 才允许进入算子库默认档。

### 7.2 量化方案分层（软件侧工作）

| 层级 | 方案 | kernel 需求 | 硬件依赖 | 适用 |
|---|---|---|---|---|
| L1 | **W4A16 / W8A16 权重量化**：int4/int8（或 fp4）权重 + 计算前反量化到 fp16 | ①dequant-in-register GEMM：`packlb`（4×byte 单指令跨步装载已有）解包 → fp16 转换 → WGMMA；②按组（group=64/128）scale 融合 | 仅 FP16 TCU（**零硬件前置**） | LLM 推理首选起步（显存减半以上、带宽降） |
| L2 | **W8A8 INT8 动态/静态**：per-channel 权重 + per-tensor/per-token 激活 scale，SmoothQuant 式 | fused quant GEMM：int8×int8→i32 累加 → requant（scale+zero-point+clamp）epilogue；动态激活量化 kernel（absmax 归约） | INT8 TCU | 吞吐敏感场景（YOLO/CNN） |
| L3 | **W8A8 FP8**（e4m3 权重+激活） | 同上，scale 用 per-tensor 常量 | FP8 TCU | 训练感知量化/高吞吐推理 |
| L4 | **块缩放 MX/NVFP4**：32/16 元素块 E8M0 scale | TCU_LD 预载 scale（`load_mx_metadata` 已有设备 API）+ MX WGMMA；权重打包工具 | MX 使能 | 极致带宽/容量场景 |
| L5 | **2:4 稀疏**：离线剪枝（magnitude/OWL）+ 压缩 | 压缩工具（host 已有 `compress_2to4`）+ SP WGMMA；运行时零额外 kernel | SPARSE 使能 | 大 GEMM 加速 ~1.8× |

### 7.3 框架接入

- **torchao 风标 API**：在 torch-vortex 里提供 `torch.vortex.quant`（`quantize_(model, W4A16Config(...))`），内部把 `nn.Linear` 换成 `VortexQuantLinear`（预打包权重 + fused kernel 引用）。
- **离线权重复理**（AWQ/GPTQ 校准、SmoothQuant scale 求解）：**全部在 CPU/GPU 宿主机做**，Vortex 只需运行时 kernel——这是量化工作量的关键减负设计；产出标准权重包格式（`.vxq`：packed weights + scales + metadata）。
- **LLM 引擎集成**：M6 的推理引擎读 `.vxq`（见 §8.4）。

### 7.4 量化 kernel 清单（并入 `vortex-quant` 库）

quant/dequant 原语（per-tensor/channel/token/group）、absmax/percentile 校准统计、fused dequant-GEMM（W4A16/W8A16）、fused int8-GEMM-requant（W8A8）、FP8 GEMM、MX 打包/装载、稀疏压缩验证器。

### 7.5 精度验证方法学

- 算子级：量化 GEMM vs fp32 参考，PSNR/相对误差分布（不只 max error）。
- 模型级：LLM 用 WikiText-2 PPL（阈值：INT8 < +0.05、W4A16-g128 < +0.15、FP8 < +0.03 经验值，写进 CI）；ViT/检测用 top-1/mAP 损失（<0.5%）；逐层输出 SNR 报告定位劣化层。
- 过程资产：golden 对拍用宿主机 PyTorch（`fedp.py` 模式），逐 dtype 建参考表。

---

## 8. M6：模型使能

**目标**：ResNet50 → YOLO(v8/11) → SAM → DINOv3 → Mamba(1/2) → LLM decode/prefill → VLM 依次点亮，每模型有 CI 化的数值验收与性能报告。

### 8.1 模型 → 算子需求矩阵

| 模型 | 特有/重度算子 | 依赖库 | 难点 |
|---|---|---|---|
| ResNet50 | conv2d、BN(fold)、maxpool/avgpool、GEMM(fc)、relu | dnn+blas | conv 算法选择与 NHWC 布局 |
| YOLOv8/11 | conv+SiLU、C2f/注意力头、**upsample/interpolate、NMS**、letterbox 预处理（CPU） | dnn+blas+prim | NMS 迭代性（并行化策略）；端到端流水 |
| SAM | ViT encoder（**window attention 14×14 + 少量全局层**）、**相对位置编码（window 索引 gather）**、prompt encoder（小 GEMM）、mask decoder（**双向 attention + `grid_sample`/点采样 + topk**） | dnn+attention+prim | window attention 的 gather/broadcast；decoder 小 kernel 多、launch 开销占比高（用 CP 批量提交摊薄） |
| DINOv3 | ViT（**RoPE**、SwiGLU、register tokens、LayerScale）、大 MLP 比例高、DINO 头 | dnn+attention+blas | RoPE 的 sin/cos 表（预计算放 host 或常量段）；长序列 attention（flash） |
| Mamba(1/2) | **selective_scan（SSM）**、**causal_conv1d**、门控融合、embedding/norm | 新 kernel + blas | 见 §8.5，全项目最"算子研究"的单项 |
| LLM（LLaMA 系） | embedding、rmsnorm、RoPE、GEMV/GEMM(skinny)、**prefill attention（flash）+ decode paged attention**、SwiGLU、sampling（top-k/top-p/temperature） | attention+blas+quant+prim | KV cache 布局与带宽；decode 是访存瓶颈；sampler 的 sort/topk |
| VLM（如 LLaVA/Qwen-VL 系） | 视觉 encoder（ViT/CLIP）+ projector（GEMM+act）+ LLM | 以上全部 | 三段拼装与中间张量驻留（显存预算） |

### 8.2 CNN 路线（ResNet/YOLO）

- 推理路径：BN fold 进 conv 权重（host 离线）；conv2d NHWC + im2col-GEMM 起步，direct conv 后置；NCHW 输入在首层前转换（`nchw2nhwc` kernel）。
- YOLO 后处理：box 解码 + confidence 阈值（元素级）→ 候选装箱（prefix-sum + scatter）→ block 并行 NMS（每 warp 一个类、IoU 矩阵分块）→ topk。
- 验收：torchvision 权重 top-1 差 ≤0.3%；COCO val 子集 mAP 差 ≤0.5%。

### 8.3 ViT 路线（SAM/DINOv3）

- patch embed = stride conv（走 conv 库）；attention 用 flash 前向（§4.5）——SAM 的 window attention 本质是 batch 维爆炸的 batched GEMM + softmax（`bmm` 路径优先）。
- RoPE：sin/cos 表预计算为常量张量（host 一次生成）；apply_rope 为元素级复数旋转 kernel（可与 QKV 投影 epilogue 融合，inductor 阶段自动完成）。
- SAM decoder 小 kernel 密集——eager 下 launch 开销是主要风险，两个缓解：CP `vx_enqueue_commands` 批量提交（一次 doorbell 多 kernel）；torch.compile 图融合。
- 验收：SAM IoU（与参考掩码）≥0.98 比例 >95%；DINOv3 线性探针/特征余弦相似度 ≥0.999。

### 8.4 LLM 路线

**执行引擎选型（按移植成本递增）**：

1. **HF transformers eager on torch-vortex**（首选第一步）：模型代码零改动，逐算子跑通；性能差但作为 M4 的终极验收与功能基线。
2. **llama.cpp HIP backend 移植**（中期，性价比最高的推理引擎）：backend 面小（ggml 的 ~150 个算子 → 我们的库映射），自带 W4A16/W8A0 量化格式与 KV cache 管理；改动集中在 `ggml/src/ggml-hip/` 换 kernel 实现。
3. **vLLM/SGLang 移植**（远期，吞吐引擎）：依赖 CUDA graph（用 CP 批量提交近似替代）、PagedAttention v2、custom allreduce（多卡）、Triton kernel（M3 就绪后复用）。工作量最大，放在量化与 M3 都稳定后评估。

**KV cache 与 decode 带宽模型**（决定硬件配置论证）：

- decode 每 token 的显存流量 ≈ 2×params（权重）+ 2×layers×ctx×d_model（KV 读）；7B fp16 ≈ 14GB/token 周期 → **W4A16（3.5GB）+ paged KV 是 Vortex 规模下唯一现实路线**；INT4 权重带宽需求正好落在 `packlb`+反量化 kernel 的设计点上。
- KV 布局：块表（page 16 token）+ 每层独立 buffer（避免层间跳跃的 TLB/缓存抖动）；decode attention 用 GEMV-per-head + 在线 softmax（flash-decoding 分割归约）。
- prefill 是 compute-bound：走 flash attention + batched GEMM，与 CNN 共用库。
- 采样：temperature/top-k（radix select）/top-p（sort + 前缀和）走 prim 库；小 batch 下全在单 kernel 融合。

### 8.5 Mamba/SSM：selective_scan kernel 设计（本项目最特别的算子）

状态空间层：`h_t = exp(Δ_t·A) ⊙ h_{t-1} + Δ_t·B_t·x_t`，`y_t = C_t·h_t (+ D·x_t)`。难点：序列维串行依赖 + (d_inner×d_state) 大状态。

**推荐算法：chunked scan（flash-linear-attention 风格）**，把串行拆成三段可并行：

1. **块内并行**：把 chunk 内 decay `exp(ΔA)` 做成下三角"线性注意力"形式——`Y_intra = (Q ⊙ decay_mask) @ (K^T ⊙ ...)`，本质 batched GEMM（**复用 TCU**），chunk 长度 L=64/128；
2. **块间状态递推**：`S_chunk = ⊙-cumprod 累乘`——沿 chunk 维串行，但每步是 `(n, d_state) ⊙ (n,) + GEMM(B, X_chunk)`，块间步数 = seq/L（可接受）；该 phase 用单 CTA 常驻（persistent kernel）+ LMEM 缓存状态，`gbarrier` 同步多 chunk 并行与串行衔接；
3. **块间贡献回投**：`Y_inter = Q @ S_prev`（GEMM）。

配套：causal_conv1d（短核滑窗，warp 内 shuffle 搬边界元素）；Mamba-2 的 SSD 形式（标量 decay、矩阵二次型）天然更贴 GEMM，**优先实现 Mamba-2 形态**，Mamba-1 用等价转换。

资源约束检查：d_state=16/128、d_inner≤2K 时状态矩阵 LMEM 常驻可行（16KB 紧张 → 再次指向 §9 P0 的 LMEM 扩容）；exp 软肋直接决定 chunk 内 decay 计算成本 → §9 P1。

验收：与参考实现（mamba-ssm CUDA kernel 输出或 FLA torch 参考）rtol 1e-3；prefill 吞吐 vs 串行 scan 加速比报告。

### 8.6 VLM 路线

视觉塔（ViT/CLIP，§8.3 路线复用）→ projector（单 GEMM + act）→ LLM（§8.4）。中间张量（image tokens）常驻显存；图像预处理（resize/normalize）放 CPU dataloader。验收：图文检索/字幕任务指标对齐参考 ±1%。

---

## 9. 硬件侧配合工作

> 原则：软件栈（M1–M6）设计为**不依赖任何新硬件**就能全部落地（用现有可配项组合出 DL 档）；以下条目按"解锁性能/生态上限"排序，每项给出动机与建议档位。均以 `VX_config.toml` 配置或既有 proposal 形式推进，不动软件 ABI。

### P0（阻塞级：没有它，上层栈只能"能跑不能好用"）

1. **DL 推荐配置档固化 + CI 常态化**：`NT=32`、`NW≥8`、多核 + L2/L3 开、`EXT_A_ENABLE`、`EXT_TCU_ENABLE` + `TCU_WGMMA_ENABLE`、DXA 开。定义 `CONFIGS` 预设（如 `DL32` / `DL16`），全部 DL CI 跑该档。
2. **LMEM 扩容 16KB→64–128KB**：Triton/CUTLASS 双缓冲 tile 的硬上限（BLOCK_M=64×BLOCK_K=32 fp16 双缓冲已 >24KB）；**注意** WGMMA smem 描述符的偏移字段 16bit → 64KB 是当前指令编码的自然上限，扩到 128KB 需要同步改描述符编码（ABI 变更，越早定越好）。RTL 侧 `LMEM_LOG_SIZE` 单参数，面积代价见 asic_gate/fpga_gate 基线。
3. **TCU dtype 使能验证（INT8 优先）**：清 §7.1 表中的开放 bug，逐 dtype 过 simx→rtlsim→门禁三关。这是 M5 的硬前置。

### P1（性能级：决定"好不好用"）

4. **SFU 数学单元（exp2/log2/rsqrt，或 packed fp16 SIMD）**：softmax/attention/Mamba 每层都吃软件 exp；一个 8/16 lane 的多项式 exp2 单元即可把这类 kernel 从 ALU bound 解放。备选：不占 SFU 的乘加耦合多项式 + 查表（软件优化先顶，硬件随后）。
5. **多队列 CP / 独立 DMA 引擎**：当前单 CP 队列 = 全设备串行；PyTorch 的多流 overlap（compute+copy）、vLLM 式 pipeline 全指望它。RTL 已有 `VX_CP_NUM_QUEUES` 参数与引擎化结构，重点是每队列独立 doorbell + 完成通道（软件 HAL 已就绪）。
6. **设备堆（device malloc）**：kernel 侧 `_sbrk` 从 ebreak 改为 LMEM/global bump 分配器（纯软件可先行：`libhip_device` 提供malloc/free，底层 global 区保留池）。
7. **barrier 槽与 MAX_BAR_EVENTS 扩容**：驻留 CTA 数 × 每 CTA barrier 需求（双缓冲 2–3 个/barrier id）+ DXA expect_tx 深度。

### P2（架构级：规模与带宽）

8. **HBM/多通道内存交织**（`docs/proposals/hbm_bank_interleaving_proposal.md` 已有）：LLM decode 与 KV 读的带宽地板；配合 `PLATFORM_MEMORY_NUM_BANKS` 上调与 AXI bank 数。
9. **寄存器文件扩容**（32 int + 32 fp/线程）：Triton 大 tile 的寄存器压力；改动大，先靠 tile 策略规避，ASIC 定档前评估。
10. **L2/L3 容量与 MSHR 深度 DL 档调优**（用 MPM 计数器数据驱动，perf_gate 防回退）。
11. **FP16 打包数据通路（half2/bf162）**：TCU 之外的逐元素 fp16（act/norm）目前要转换；可选。

### P3（远期）

12. 多卡：P2P/集合通信（RCCL 算法层移植）+ 物理互联。
13. 抢占/时间片（trap foundation 已铺路，`docs/designs/trap_and_exception_foundation.md`）。
14. SV39 MMU for rv64（当前 SV32 仅 rv32）+ 统一内存语义（`hipMallocManaged`）。

---

## 10. 测试、CI 与工具链工程

### 10.1 三级验证体系（继承现有机制，向上延伸）

| 级别 | 载体 | 用途 | 频率 |
|---|---|---|---|
| 功能（快） | simx（+`SIMX_FUNCTIONAL`） | 全量算子/模型数值回归 | 每 PR |
| 周期（准） | rtlsim + `model_parity` + `perf_gate` | 关键算子周期基线 ±2% 门禁 | 每 PR（抽样）+ nightly（全量） |
| 实测（真） | FPGA（U55C/V80 aved）/ASIC 门禁 | PPA 与真实延迟 | nightly/周 |

### 10.2 新增 CI 目录规划

```
ci/testcases/
  hip_native.yaml      # M1：libhip_vortex API smoke + 内联指令暴露（sgemm_tcu_wg on HIP）
  prim.yaml blas.yaml dnn.yaml rand.yaml attention.yaml   # M2
  triton.yaml          # M3：tutorials + 对拍
  pytorch.yaml         # M4：模型 harness（simx 全量/rtlsim 抽样）
  quant.yaml           # M5：dtype×方案矩阵 + 精度门禁（PPL/mAP 阈值）
  models.yaml          # M6：端到端模型 zoo
```

### 10.3 数值容差策略（全栈统一，写进测试框架）

fp32 rtol 1e-5 / fp16 1e-2 / bf16 2e-2 / int8 逐位 / fp8 统计分布检验；量化模型见 §7.5。golden 来源优先级：同源 CPU 参考 > PyTorch CPU > 人工构造。

### 10.4 性能工具

- 现有：MPM 计数器（含 TCU/DXA 类）、`ci/roofline.py`、`ci/perfetto.py`（kernel 时间线 + CP 队列可视化）、`vx_event_get_profiling`（op 级）。
- 新增：算子库 autotuner 报告（tile/stages 搜索结果归档）；PyTorch `torch.profiler` 接 Vortex 时间线；模型级 tokens/s、imgs/s 仪表盘（CI 产物 JSON + 趋势图）。

### 10.5 工具链发布

- 沿用 `ci/toolchain_install.sh` 预编译 bundle 模式，新增 `--hipvortex --torch --triton` 组件；`VERSION` 文件加 pin（`TRITON_REV`、`PYTORCH_REV`）。
- TheRock 的 artifact 分包（dev/run/dbg/doc 组件）与 wheel 夜间构建流程值得作为 torch-vortex 发布的参考蓝本。

---

## 11. 里程碑、工作量与团队规划

### 11.1 里程碑表（相对时间，假设 8 人团队并行）

| 里程碑 | 内容 | 工作量 | 关键交付 | 出口判据 |
|---|---|---|---|---|
| **M0（0.5 月）** | DL 配置档定档、开放 bug 清零（wgmma fp64/rtlsim 等）、golden 框架 | 1–2 人月 | `CONFIGS` 预设 + CI 档位 | DL 档全量 regression 绿 |
| **M1（2–3 月）** | 原生 HIP：libhip_vortex + hipcc + 设备头 | 4–6 人月 | wheel 前身：libhip_vortex.so + hipcc-vortex | §3.5 全绿；HIP 源码 sgemm_tcu_wg 与原生对拍 |
| **M2（3–5 月，与 M3 并行）** | prim/blas/dnn/rand/attention | 8–12 人月 | 四个库 + autotuner | P0 算子全绿；fp16 GEMM 达 G1 利用率 |
| **M3（3–5 月，与 M2 并行）** | Triton 后端 | 6–10 人月 | third_party/vortex | tutorials 绿；matmul ±20% 对拍 |
| **M4（2–4 月）** | torch-vortex | 6–9 人月 | pip wheel | resnet50 eager 全绿；compile 阶段 a/b 就绪 |
| **M5（2–3 月）** | 量化（波次 1–2） | 4–6 人月 | vortex-quant + torch.vortex.quant | INT8/W4A16 GEMM + LLM PPL 达标 |
| **M6（3–6 月）** | 模型点亮 | 6–12 人月 | 模型 zoo + CI | §8 各模型验收指标 |
| 合计 | | **~35–55 人月** | | 12–18 个月日历时间 |

### 11.2 团队技能配置建议（8 人参考）

- 编译器 2：VOLT/LLVM（M1 hipcc、M3 lowering）——一名 MLIR/Triton 背景牵头 M3；
- 运行时 1.5：vortex2.h 演进 + libhip_vortex + PyTorch allocator；
- 算子库 2.5：GEMM/conv/attention/prim（CUDA kernel 经验优先）；
- 框架 1：torch-vortex + inductor patch；
- 模型/量化 1：selective_scan、KV cache、量化方案。
- 硬件接口兼职 1（LMEM 扩容、多队列 CP 等提案推进，与 RTL 团队协同）。

### 11.3 关键路径

`M0 → M1 →（M2 ∥ M3）→ M4 → M6（CNN→ViT→LLM/Mamba）`；M5 挂在 M2 的 GEMM 与 §9 P0 的 INT8 使能之后，与 M4 并行。**最长杆是 M3（Triton）**，应最早启动预研（layout 对齐原型可以在 M1 完成前用 LLVM IR 手写实验）。

---

## 12. 风险登记册

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| R1 | WGMMA fragment 布局映射错 → 静默数值错（Triton/库层） | 高 | RS 先行、逐 tile 对拍、`wgmma_context` 布局公式写成单测 |
| R2 | Triton/PyTorch 上游 API 漂移 | 高 | 版本 pin + 每季度重定基窗口；patch 集最小化并持续上游化 |
| R3 | LMEM 16KB 限制使 tile 太小，GEMM/attention 效率上不去 | 高 | §9 P0 早启动；软件侧 tile 表按 16KB 先跑通保功能 |
| R4 | TCU dtype 使能后面积/频率回退（ASIC）/资源爆（FPGA） | 中 | 每波次走 asic_gate/fpga_gate 基线；分档开 dtype |
| R5 | 单 CP 队列串行使多流/overlap 方案失效 | 中 | §9 P1；先靠批量提交摊薄 launch 开销 |
| R6 | 软件 exp 成为 softmax/attention/Mamba 瓶颈 | 中 | 多项式+查表软件优化先行；SFU 数学单元提案 |
| R7 | rtlsim 速度不足以跑模型级周期回归 | 中 | 模型级只跑 simx 功能 + FPGA 实测；算子级跑 rtlsim 门禁 |
| R8 | chipStar 路线用户迁移断档 | 低 | M1 后 `tests/hip` 双轨跑两个后端一季，再切默认 |
| R9 | TFR 定点树 vs IEEE 参考的数值差异在低精度（fp8/fp4）放大 | 中 | `fedp.py` 式宿主 golden 逐 dtype 建档；容差进 CI |
| R10 | 复合技能（RISC-V+GPU+ML 编译）人员不足 | 高 | M3/M1 找有 Triton/LLVM 经验外援；本方案文档 + 上游化降低维护孤岛 |

---

## 13. 附录：映射表与资料索引

### 13.1 仓库内关键文档索引（本方案的依据）

| 主题 | 文档 |
|---|---|
| 软件系统总览（中文） | [docs/mydocs/vortex_software_system.md](../mydocs/vortex_software_system.md) |
| 运行时 API 设计 | `docs/designs/vortex_runtime_api.md` |
| CP 命令处理器 | `docs/designs/command_processor.md` |
| CTA 调度 / 内核入口 | `docs/designs/cta_dispatch_architecture.md`、`kernel_entry_and_dispatch.md` |
| 张量核（WGMMA/稀疏/MX） | `docs/designs/tensor_core_wgmma_engine.md` |
| DXA 异步拷贝/多播 | `docs/designs/dxa_async_copy_multicast.md` |
| 缓存子系统 / 内存属性 | `docs/designs/cache_subsystem.md`、`memory_fabric_attributes.md` |
| 原子 / 一致性 | `docs/designs/atomic_memory_operations.md`、`multicache_amo_coherence.md` |
| 微架构 | `docs/designs/microarchitecture.md` |
| OpenCL / HIP / Vulkan 接入 | `docs/designs/opencl_on_vortex.md`、`hip_on_vortex_chipstar.md`、`vortexpipe_architecture.md` |
| 自定义加速器 ISA 扩展指南 | `docs/designs/custom_accelerator_isa_extensions.md`（新内联指令暴露必读） |
| 虚拟内存 / 陷阱 | `docs/designs/virtual_memory_subsystem.md`、`trap_and_exception_foundation.md` |
| 工具链构建 | `docs/building_toolchain.md`（`+xvortex` 特性名、无默认 triple 等坑） |
| HBM 交织提案 | `docs/proposals/hbm_bank_interleaving_proposal.md` |
| CI / 性能门禁 | `docs/designs/continuous_integration.md`、`docs/testing.md` |

### 13.2 HIP → vortex2.h 映射速查（M1 实现基线，详见 §3.1）

```
hipSetDevice            → vx_device_open(idx)（进程设备表）
hipMalloc/Free          → vx_buffer_create / vx_release
hipMemcpyAsync          → vx_enqueue_{copy,read,write}（+ _rect）
hipMemsetAsync          → vx_enqueue_fill_buffer
hipStreamCreate         → vx_queue_create
hipStreamSynchronize    → vx_queue_finish
hipEventRecord          → vx_event_signal（时间线值）
hipStreamWaitEvent      → enqueue 的 wait-list
hipModuleLoadData       → vx_module_load_bytes（vxbin）
hipModuleGetFunction    → vx_module_get_kernel（VXSYMTAB）
hipModuleLaunchKernel   → vx_enqueue_launch（grid/block/cluster/lmem/args blob）
hipHostMalloc           → vx_buffer_create + VX_MEM_PIN_MEMORY / VX_MEM_HOST
hipDeviceProp           → vx_device_query(VX_CAPS_*) 全字段映射
```

### 13.3 Triton → Vortex 概念映射速查（M3 实现基线，详见 §5.3）

```
num_warps × 32 lanes       → 一个 CTA（BLOCK_SIZE = num_warps 个 Vortex warp，NT=32）
tl.dot                     → WGMMA（SS: smem_matrix_desc；RS: 寄存器 A）
  acc fp32 / in fp16|bf16|i8 → 对应 TCU dtype 使能位
num_stages 双缓冲            → DXA issue + vx_barrier_expect_tx + arrive/wait
shared memory              → CTA LMEM（上限写进 parse_options；描述符 16bit 偏移≤64KB）
libdevice（__nv_expf...）    → vortex_libdevice → musl libm / 快速多项式
shfl/ballot/syncwarp       → vx_shfl_* / vx_vote_* / vx_wsync（inline-asm）
atomic                     → RVA amo*（inline-asm）
cubin/hsaco                → vxbin（VXSYMTAB）+ metadata json
PTXAS/ROCm objcopy         → VOLT clang/lld + sw/kernel/scripts（vxbin.py）
launcher（cuLaunchKernel）  → libhip_vortex hipModuleLaunchKernel
```

### 13.4 外部参考资料

- VOLT（Vortex 编译器）：`github.com/vortexgpgpu/Volt`（CC'26 论文）；llvm fork `vortex_3.x`
- PoCL fork：`github.com/vortexgpgpu/pocl`；chipStar fork：`github.com/vortexgpgpu/chipStar`；Mesa fork：`github.com/vortexgpgpu/mesa`
- TheRock（ROCm 构建/发布系统，库源码移植参考）：`github.com/ROCm/TheRock`（本地 `~/aichip/TheRock`）
- Triton 后端参考：`openai/triton`（`third_party/nvidia|amd`）；MLIR TritonGPU dialect
- CUTLASS（tile 编程范式）、rocPRIM/hipBLASLt/rocRAND/RCCL（TheRock 内源码）
- 模型参考实现：`mamba-msm`（selective_scan CUDA）、`flash-linear-attention`（chunked 算法）、`vllm-project`（paged attention）、`ggml`（llama.cpp 算子面）
- 量化：torchao、AWQ/GPTQ/SmoothQuant 论文与开源实现；OCP MX 格式规范（MXFP8/4、NVFP4 块缩放定义）

---

*文档版本 v1.0（2026-09-10），基于 Vortex v3.0-911-gbba80f8de 撰写。维护约定：随里程碑推进更新 §11 状态列与 §12 风险状态；重大架构决策（warpSize 档位、LMEM 扩容、多队列 CP）落定后回写 §2/§9。*

## 14. 面向 AI 开发的可执行工作包

本节把前面的路线拆成可以交给一个 AI 开发代理的最小工作包。每个工作包只允许有一个主要目的；代理提交代码时必须同时提交测试、配置和验证证据。`P0/P1/P2` 表示优先级，`S`、`M`、`L` 分别表示约半天、1–3 天、1 周以上的估算，不等同于人月承诺。

### 14.1 工作包清单

| ID | 优先级/规模 | 交付物（建议路径） | 主要动作 | 可验证出口 |
|---|---|---|---|---|
| M0-01 | P0/M | `configs/dl32.toml` 或等价 `CONFIGS` 预设、`ci/testcases/dl_config.yaml` | 固化 NT=32、多核、L2/L3、A/TCU/DXA 的功能档和周期档；记录每个派生参数 | 从干净 `build/` 运行 `../configure`；simx/rtlsim 基础 regression 全绿；生成配置快照 |
| M0-02 | P0/M | `tests/dl/golden/`、`ci/dl_compare.py` | 建立 CPU/PyTorch golden、dtype 容差和 NaN/Inf 检查；输入使用固定 seed | vecadd、sgemm、softmax、reduce 在 rv32/rv64（适用时）逐元素或统计对拍 |
| M0-03 | P0/M | `tests/hip/native/` 骨架、`ci/testcases/hip_native.yaml` | 建立原生 HIP 最小测试，不依赖 chipStar；先覆盖 module、launch、copy、event、atomic | simx 与 rtlsim 均通过；失败时输出 kernel 名、配置、args 大小和首个错误元素 |
| HIP-01 | P0/L | `hipvortex/runtime/`、`libhip_vortex.so` | 将 HIP device/memory/queue/event/module/kernel API 映射到 `vortex2.h`；句柄生命周期和错误码先完成 | API 单测覆盖 create/destroy、双重释放、越界 copy、同步/异步错误；ASAN 主机测试通过 |
| HIP-02 | P0/M | `hipvortex/device/include/hip/` | 实现线程索引、warp vote/shuffle、barrier、atomic、printf、half/bfloat 类型 | 每个 intrinsic 有单独 kernel；结果与 CPU golden 对拍；A 扩展关闭时返回明确编译或运行错误 |
| HIP-03 | P0/M | `hipvortex/hipcc`、`hipvortex/pack_vxbin.py` | 接 VOLT clang/lld/link 脚本/vxbin，支持多个 `__global__` 入口和 fatbin | 一个源文件含 2 个 kernel；`hipModuleGetFunction` 按名取得并分别 launch；检查 ELF/vxbin 符号表 |
| HIP-04 | P0/M | `hipvortex/tests/tcu/` | 用 HIP 重写 `sgemm_tcu_wg`、DXA 双缓冲、稀疏/MX 的最小样例 | 与原生 regression 数值一致；rtlsim 周期误差纳入 perf 基线；失败时 dump tile/descriptor |
| LIB-01 | P0/L | `vortex-prim/` | 实现 block/device reduce、scan、sort、gather/scatter；所有 warp 假设参数化 | 随机尺寸、非 2 次幂尺寸、部分 warp、重复键测试；CPU/PyTorch 对拍 |
| LIB-02 | P0/L | `vortex-blas/` | 从 `tests/regression/sgemm_tcu*` 抽取 GEMM tile/mainloop/epilogue；先 FP16/BF16，后 INT8 | NN/NT/TN/TT、batch、skinny GEMM；误差和吞吐报告；无静默越界 |
| LIB-03 | P0/L | `vortex-dnn/`、`vortex-attention/` | 实现 conv2d、norm、activation、pool、softmax、flash attention 前向 | ResNet/ViT 形状集逐算子对拍；长序列和非对齐 stride 必须覆盖 |
| LIB-04 | P1/M | `vortex-rand/` | 实现 Philox 状态、uniform/normal/dropout 及 host seed/offset | 固定 seed 下与 PyTorch CPU/CUDA 参考统计一致；并发 stream 不重复序列 |
| TRI-01 | P0/M | Triton fork `third_party/vortex/backend/` | 锁定 Triton minor 版本；实现 target、driver、缓存键和最小 LLVM IR 编译路径 | `add`/`mul`/load/store tutorial 在 simx 通过；编译缓存命中率可观测 |
| TRI-02 | P0/L | `third_party/vortex/libdevice/`、Triton lowering | 完成 blocked layout、reduction、barrier、RS `tl.dot`；再做 SS/WGMMA 和 DXA | vecadd、softmax、matmul 逐元素对拍；RS 与 SS 的 tile 结果一致 |
| TRI-03 | P1/M | Triton launcher、autotune 适配 | 将 metadata 的 num_warps/shared memory/args 映射到 `vx_launch_info_t` | `@triton.autotune` 至少选择两个配置；错误的 LMEM 请求被拒绝并带原因 |
| TORCH-01 | P0/M | `torch-vortex/` Python 包、C++ extension | 注册 PrivateUse1、设备/stream/event guard、allocator、copy/to/empty | `torch.randn(..., device='vortex')`、跨设备 copy、释放和 OOM 重试通过 |
| TORCH-02 | P0/L | `torch-vortex/ops/` | 接入 GEMM、conv、norm、activation、reduce、embedding；未实现算子采用可观测 CPU fallback | 每次 fallback 记录算子名和搬运字节数；ResNet50 eager 前向通过 |
| TORCH-03 | P1/L | `torch-vortex/inductor/` | 先接 `aot_eager`，再接 Triton/Inductor；固定 PyTorch 版本和最小 patch 集 | 同一模型 eager/compile 输出一致；graph break 数量和原因进入日志 |
| QNT-01 | P0/M | `vortex-quant/pack.py`、`.vxq` 格式 | 先实现 W4A16/W8A16 分组打包、scale/zero-point 元数据和校验器 | pack→unpack 位级一致；fused dequant GEMM 与 FP16 参考误差达标 |
| QNT-02 | P1/L | INT8/FP8/MX/SP kernel 与 TCU 配置 | 按“simx 数值→rtlsim 数值/周期→FPGA/ASIC 门禁”逐 dtype 开启 | 每种 dtype 有独立 golden、性能基线、面积/频率记录；禁止一次性全开 |
| MOD-01 | P0/M | `tests/models/resnet50/`、`models.yaml` | 先固定小权重/小输入，完成 ResNet50 端到端和预处理 | top-1/中间层误差门禁；simx 每 PR，rtlsim nightly |
| MOD-02 | P0/M | YOLO/SAM/DINO harness | 按算子缺口逐个补齐 interpolate、NMS、window attention、RoPE、grid_sample | 固定数据集子集，mAP/IoU/特征余弦门禁；输出失败层定位信息 |
| MOD-03 | P1/L | `vortex-llm/`、`vortex-mamba/` | 实现 RMSNorm/RoPE/GEMV/paged KV/sampling 和 chunked selective scan；先 tiny 模型 | tiny LLaMA/Mamba 与参考逐 token 对拍；KV 地址、page 表和状态转移可 dump |
| HW-01 | P0/M | `docs/proposals/dl_config.md`、配置/门禁改动 | 评估 LMEM 64/128KB、TCU dtype、A 扩展对面积频率影响 | `fpga_gate.py`/`asic_gate.py` 产出可审计报告；配置进入 catalog 后才能作为默认档 |
| HW-02 | P1/L | CP 多队列、SFU、HBM 交织 proposal/RTL | 先用 MPM 证明瓶颈，再做 RTL；每个硬件变更同步更新 SimX | model_parity 通过；新增计数器能解释性能变化；无放宽既有容差 |

依赖顺序是 `M0-01/M0-02 → HIP-01/02/03 → LIB-01/02 → TRI-01/02 → TORCH-01/02 → QNT-01 → MOD-01/02 → MOD-03`。`HW-01` 可与 HIP/LIB 并行，但任何改变周期或内存行为的 RTL 工作必须和对应 SimX 修改同一变更提交。

### 14.2 AI 代理的单任务执行协议

每个代理领取一个工作包后，按以下顺序执行，并在提交说明中逐项填写：

1. **建立基线**：从仓库根目录检查 `git status`；进入独立 build 目录运行 `../configure`；记录 commit、VORTEX_VERSION、XLEN、CONFIGS、工具链版本。不得复用不兼容的旧 build。
2. **定位现有实现**：先读本表引用的设计文档和实际源文件，再写最小复现。修改前用 `rg` 找到所有调用者、对应 SimX/RTL 实现和现有测试。
3. **先写可失败测试**：测试必须能在修改前稳定失败，并说明失败原因；随机测试固定 seed，模型测试固定权重版本和输入文件哈希。
4. **最小实现**：只改变工作包范围内的接口；跨层共享 ABI 放 `sw/common/`；不得从 `sw/kernel` 引用 `sim`/`hw`，也不得反向引用。
5. **双模型验证**：功能变更至少跑 simx；涉及周期、仲裁、缓存、队列或同步时同时跑 rtlsim，并检查 `model_parity`。不要用放宽 tolerance 掩盖差异。
6. **配置同步**：修改任何 TOML、Makefile 或参与构建的目录后，从 build 目录重新运行 `../configure`；确认生成的 `sw/VX_config.h` 与 `hw/*.vh` 的时间戳和内容已更新。
7. **证据归档**：在 `artifacts/<work-package>/<timestamp>/` 保存命令、配置快照、测试摘要、数值误差、周期/带宽计数器和失败样例；大模型权重不入 git，只保存下载地址、版本和 SHA256。
8. **交付检查**：提交代码、测试、文档和回滚说明；列出未覆盖的 API/数据类型/形状。一个工作包的验收条件未满足时，不得宣称里程碑完成。

### 14.3 需要用户确认的技术决策

以下决策会改变公共 ABI 或长期维护成本，建议在开始对应里程碑前确认；在没有新指示时，本文采用“推荐”列。

| 决策 | 推荐值 | 需要确认的原因 |
|---|---|---|
| 原生 HIP API 范围 | HIP Runtime + Module + 基础 cooperative groups，暂不承诺完整 ROCm | 决定头文件兼容度、测试规模和是否需要 HSA/KFD |
| HIP 实现位置 | 独立 `hipvortex` 仓库，Vortex 只保留接口/测试/构建 glue | 避免把外部依赖和工具链源码塞入硬件仓库；若希望单仓库可改为顶层子目录 |
| Triton 版本 | 固定一个 3.x minor，先完成后再升级 | 后端 API、LLVM IR 和 Python ABI 都会漂移 |
| PyTorch 版本 | 固定一个 2.x minor，PrivateUse1 优先 | 决定 dispatcher、allocator 和 Inductor patch 面 |
| warp size | DL 默认档 NT=32，保留 NT=16 功能档 | 32 与主流生态一致，但面积/仿真成本更高 |
| LMEM | 先验证 64KB；只有 tile/attention 证据不足时再评估 128KB | 描述符偏移位宽、BRAM 面积和 ABI 可能变化 |
| 数学单元 | 先软件 fast-math + profiling，再决定 SFU | 避免在没有 softmax/attention 计数器证据前固定 RTL 成本 |
| 分布式 | 首版单卡；DDP 走 CPU gloo；RCCL/P2P 后置 | 多卡互联不是 HIP/PyTorch 单卡功能的前置条件 |
| 训练支持 | 先推理，再 autograd/训练 | 训练需要反向算子、优化器、检查点和更高内存容量 |

如果这些决策中任一项需要不同选择，应在对应工作包开始前更新本表、§2.3、§9 和里程碑出口判据，并增加一份迁移测试；不要在实现中隐式改变。

## 15. 变更记录

| 日期 | 变更 | 原因 |
|---|---|---|
| 2026-09-10 | v1.0：完成现状盘点、六阶段路线、硬件配合、测试和风险登记 | 建立深度学习全栈基线 |
| 2026-09-11 | 增加第 14 节工作包、AI 执行协议和决策确认表 | 将路线转成可分派、可验证的开发任务 |


