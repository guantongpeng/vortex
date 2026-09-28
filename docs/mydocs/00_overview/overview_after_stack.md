# 按照 overview_plan.md 完成后，Vortex 软件栈会变成什么样

> 面向不了解 Vortex、GPU、编译器和 PyTorch 的读者。
>
> 本文解释的是：**把 [overview_plan.md](overview_plan.md) 中的工作全部完成后，目标软件栈的结构、能力、使用方式和边界**。它不是当前完成度报告，也不把路线图中的目标误写成今天已经交付的功能。
>
> 代码基线：d57c50d11（以 overview_plan.md 记录为准，2026-09-22）。
>
> 相关背景：
>
> - [当前 Vortex 软件系统详解](overview_software_system.md)
> - [统一实施计划](overview_plan.md)
> - [PyTorch 现状审计](../05_pytorch/torch_plan.md)
> - [原生 HIP / PyTorch / Triton 实施方案](overview_dl_fullstack_implementation.md)

---

## 1. 先给一个不带术语的结论

Vortex 原本是一个**可编程的 RISC-V GPGPU 平台**：它有自己的 GPU 硬件、设备内核编译器、运行时、模拟器和 FPGA/ASIC 接口，主要用于运行原生 C/C++、OpenCL 和 Vulkan 程序，也有一条通过 chipStar/PoCL 运行 HIP 的兼容路径。

完成 overview_plan.md 后，它会在这个基础上变成一个**面向深度学习推理、并逐步覆盖训练的完整设备软件栈**：

- 普通用户可以用 PyTorch 创建 device="vortex" 的张量，运行常见 CNN、Transformer、LLM 和视觉模型。
- GPU 内核可以用原生 HIP C/C++ 编写，也可以用 Triton 的 kernel[grid] (...) 形式编写。
- PyTorch eager、torch.export、torch.compile/Inductor、Triton 和底层 Vortex kernel 会共用同一个设备运行时、内存模型、队列、事件和二进制格式。
- FP32、FP16、BF16、INT8、FP8，以及部分 INT4/W4A16、MX/NVFP4、2:4 稀疏路径会有明确的编码、算子、能力查询和测试证据。
- 同一个程序先在 SimX 功能/周期模型上验证，再在 Verilator RTL 上做模型一致性检查，最后可在 XRT FPGA 上做真正的板卡集成和性能测量。
- 没有实现的算子、数据类型、硬件单元或模型组合会在启动前给出明确错误，或者按文档声明的方式执行可观测的软件路径；不会因为“能跑出一个结果”就被宣称为完整支持。

可以把最终目标理解成：

> **Vortex 硬件是地基，vortex2.h 是统一设备接口，原生 HIP 和 Triton 是内核编程入口，sw/dl 是可复用算子库，torch-vortex 是 PyTorch 设备后端，SimX/RTL/XRT 是三种验证和运行后端。**

---

## 2. 先认识几个词

| 词 | 对初学者的含义 |
|---|---|
| **主机（host）** | 运行 Linux、Python、PyTorch 和 C++ 主程序的 CPU 机器。 |
| **设备（device）** | Vortex 的 RISC-V GPGPU。真正执行 kernel 的地方。 |
| **kernel** | 在设备上并行执行的小程序，例如向量加法、矩阵乘法、卷积。 |
| **运行时（runtime）** | 主机端库，负责分配设备内存、拷贝数据、加载 kernel、提交执行和等待完成。 |
| **队列/stream** | 一条有顺序的设备工作队列。拷贝、kernel、事件等待都会排在队列里。 |
| **event** | 用来表示“某个设备工作已经完成”的可等待标记。 |
| **CP（Command Processor）** | 设备里的命令处理器。主机把命令放入命令环，CP 负责取出并执行。 |
| **KMU（Kernel Management Unit）** | 设备里的 kernel 管理单元。根据 grid/block 配置启动 CTA。 |
| **CTA / block** | 一组一起运行、可以使用共享本地内存和屏障的线程。CUDA/HIP 中通常叫 block。 |
| **SIMT** | GPU 的并行执行方式：多个线程共享指令流，同时处理不同数据。 |
| **LMEM** | 每个核心附近的本地内存，类似 CUDA 的 shared memory，适合做 tile 暂存。 |
| **TCU** | Tensor Core Unit，面向矩阵乘法和低精度计算的专用单元。 |
| **DXA** | Data eXchange Accelerator，负责异步数据搬运和多播。 |
| **.vxbin** | Vortex 的设备可执行文件。它包含设备代码映像和 kernel 入口信息。 |
| **SimX** | C++ 编写的 Vortex 模拟器，适合快速验证功能和周期行为。 |
| **rtlsim** | 用 Verilator 运行真实 RTL 的仿真后端，比 SimX 更接近硬件实现。 |
| **XRT** | Xilinx FPGA 的运行时接口。它能让同一个 Vortex runtime 访问真正的 FPGA bitstream。 |
| **PrivateUse1** | PyTorch 为外部设备后端预留的 dispatch 槽位。Vortex 不需要修改 PyTorch 核心，就能注册为 vortex 设备。 |
| **Triton** | 用 Python 描述 GPU kernel，由编译器生成设备代码的编程系统。 |
| **Inductor** | PyTorch torch.compile 使用的图编译器。它会把多个算子编译、融合并安排到目标设备上。 |
| **eager** | PyTorch 逐个执行算子的模式，例如先执行 relu，再执行 mm。 |
| **graph / graph capture** | 把一串算子的依赖关系收集成图，再整体编译或优化。 |

---

## 3. 当前基线：原始 Vortex 软件栈是什么样

这里的“原始”指**完成深度学习全栈计划之前的 Vortex 基础系统**。仓库当前已经比最早的上游版本丰富，且部分深度学习功能已经落地，所以需要把“稳定基础”和“仍在建设的上层”分开看。

### 3.1 基础系统的分层

~~~mermaid
flowchart TB
    APP["用户程序<br/>C/C++ / OpenCL / Vulkan / 部分 HIP"]
    API["上层 API<br/>原生 vortex2.h / OpenCL / Vulkan / chipStar HIP"]
    RT["主机端 runtime<br/>libvortex.so<br/>设备、内存、队列、事件、模块、DMA"]
    HAL["后端适配层<br/>SimX / rtlsim / OPAE / XRT / FireSim / gem5"]
    CP["设备控制面<br/>CP 命令环、DCR、DMA、事件"]
    KMU["KMU<br/>grid/block/cluster 调度"]
    CORE["Vortex RISC-V SIMT 核心<br/>warp、线程、缓存、LMEM"]
    FU["专用单元<br/>FPU / TCU / DXA / TEX / RASTER / OM / RTU"]
    MEM["设备内存系统<br/>L1/L2/L3、DRAM、虚拟内存"]
    BIN["设备程序<br/>VOLT clang -> ELF -> vxbin"]
    APP --> API --> RT --> HAL --> CP --> KMU --> CORE
    CORE --> FU
    CORE --> MEM
    BIN --> RT
~~~

这套基础栈已经解决了几件关键事情：

1. **如何编译设备程序**  
   VOLT 是带 Vortex 扩展的 LLVM/RISC-V 工具链。设备代码会被编译成 RISC-V ELF，再由链接脚本和 vxbin.py 打包成 .vxbin。

2. **如何把程序装载到设备**  
   主机 runtime 解析 .vxbin，为它预留设备地址，上传代码和全局数据，并根据符号表找到 kernel 入口。

3. **如何启动 kernel**  
   主机把 grid、block、参数地址、LMEM 大小等信息写入 CP 命令环。CP 再编程 KMU，KMU 按 CTA 启动设备 kernel。

4. **如何搬运数据**  
   H2D、D2H、D2D 和模块上传都通过 CP 的 DMA 命令完成。上层库不直接绕过 runtime 操作设备内存。

5. **如何在不同环境运行同一程序**  
   VORTEX_DRIVER 选择 SimX、rtlsim、OPAE、XRT 等后端。上层程序尽量不改变，变化的是底层设备实现。

### 3.2 基础目录各自负责什么

| 目录 | 当前作用 |
|---|---|
| sw/kernel/ | 设备端启动代码、SIMT intrinsic、屏障、printf、TCU/DXA/图形/光追头文件和设备静态库。 |
| sw/runtime/ | 主机端 libvortex.so、vortex2.h、队列、事件、内存、module loader 和后端适配。 |
| sw/common/ | 主机、设备、模拟器共用的 ABI、内存记账、图形/张量配置和软件参考模型。 |
| sw/dl/ | 已有的 BLAS、DNN、primitive、attention、LLM、Mamba、RNG、量化和稀疏 kernel 库；但计划完成前并未全部接入 PyTorch。 |
| sw/hip/ | 原生 HIP runtime 子集 libhip_vortex。 |
| torch-vortex/ | PyTorch PrivateUse1 后端原型，已有设备、分配器、部分 eager 算子和 MiniResNet 资产。 |
| triton-vortex/ | Triton 后端原型，已有部分 TTIR 转换、HIP binding 和 vecadd/softmax 类实验。 |
| sim/ | SimX、rtlsim 和共享的 CP/ELF/DRAM/内存基础设施。 |
| hw/ | Vortex 处理器、CP、KMU、缓存、TCU、DXA、图形和 FPGA AFU 的 RTL。 |
| tests/ | regression、runtime、OpenCL、HIP、DL、Vulkan、图形、光追等测试。 |
| ci/ | configure 生成脚本、blackbox 运行器、pytest/YAML 测试矩阵、性能和综合门禁。 |

### 3.3 计划开始前，深度学习部分的真实状态

计划不是从零开始，也不是已经完成。当前大致是下面的状态：

| 层 | 已经有的东西 | 仍然缺什么 |
|---|---|---|
| HIP | chipStar → PoCL → SPIR-V 的兼容路径；原生 HIP 编译探针、hipcc-vortex、libhip_vortex 子集 | 三尖括号、shared、half/bfloat、完整 module/stream/event/atomic 语义和完整 conformance |
| DL kernel | sw/dl 已有 GEMM、conv、norm、activation、attention、RNG、量化等一批 kernel | dtype/layout/shape 边界、统一参数 ABI、更多 batched/epilogue 和硬件变体 |
| PyTorch | PrivateUse1、基础设备 tensor、部分 ATen eager、MiniResNet 前向资产 | 完整 tensor 语义、allocator、stream/event、RNG、checkpoint、schema 覆盖、FakeTensor、export、Inductor、Autograd |
| Triton | TTIR → C → vxbin 原型，vecadd/softmax/reduce 路径 | 标准 kernel[grid] launcher、正式 MLIR lowering、tl.dot、shared/barrier、PyTorch/Inductor 互操作 |
| 低精度 | FP16/BF16 GEMM、INT8/FP8/MX/NVFP4/2:4 的部分软件编码或 kernel | PyTorch/torchao 接入、TCU 真硬件能力 gate、bit-level golden、模型精度和性能报告 |
| 硬件验证 | SimX 主线证据，部分 rtlsim 和 capability/CI | 系统性 SimX-retired-instruction parity、XRT 全路径、真实 FPGA 性能和安装交付 |
| 工程化 | build 树、阶段性测试和报告 | 环境 manifest、干净安装、wheel+SDK、完整 CI 分层、错误诊断和可复现证据 |

因此，当前能运行一个 vecadd、某些 DL kernel 或 MiniResNet，并不等于“Vortex 已经是完整 PyTorch GPU”。

### 3.4 当前验证主线应该怎样理解

overview_plan.md 当前锁定的正式验证主线是 **RV64、Python 3.10、PyTorch 2.14、SimX**。已有记录显示，部分 PyTorch/算子阶段已经有成批通过的测试，例如 W0/W1、W2、统一 primitive 以及 bmm、cat/stack、gather/scatter、pool、index_add 等专项用例；这些证据说明相应子集已经能工作，但不等于完整 eager、完整模型、Triton 编译链或 FPGA 已交付。

有些旧记录使用过临时的 Python/PyTorch 组合。它们可以帮助定位问题，却不能自动扩大正式支持矩阵。判断某个功能是否完成，应同时看代码、对应测试、环境 manifest、后端标签和错误边界。

---

## 4. 完成计划后的目标软件栈

### 4.1 最终全栈图

~~~mermaid
flowchart TB
    subgraph USER["用户和模型层"]
        PY["PyTorch eager<br/>torch.device('vortex')"]
        COMP["torch.compile / Inductor<br/>torch.export / FX"]
        TRI["Triton Python<br/>kernel[grid] (...)"]
        HIP["原生 HIP C/C++<br/>hipcc-vortex"]
        NATIVE["原生 C/C++ / OpenCL / Vulkan"]
        MODEL["CNN / ViT / Mamba / LLM / VLM<br/>推理，后续训练"]
    end

    subgraph FRAMEWORK["框架适配层"]
        TV["torch-vortex<br/>PrivateUse1、ATen、allocator、guard、RNG、Autograd"]
        IV["Inductor Vortex backend<br/>图编译、融合、memory planning"]
        TVX["triton-vortex<br/>标准 launcher、driver、compiler、autotune"]
        HIPRT["libhip_vortex<br/>HIP runtime/driver API"]
        POCL["PoCL / chipStar<br/>兼容 HIP/OpenCL 路径"]
    end

    subgraph LIBS["Vortex kernel 与算子层"]
        DL["sw/dl 统一算子库<br/>BLAS / DNN / prim / attention / RNG / quant"]
        KERN["sw/kernel<br/>启动、SIMT、barrier、TCU、DXA、设备 libc"]
        VXBIN["统一设备二进制<br/>VOLT -> ELF -> vxbin"]
    end

    subgraph HOST["统一主机运行时"]
        ABI["vortex2.h contract<br/>Device / Buffer / Queue / Event / Module / Kernel"]
        CPAPI["CP 命令环 API<br/>launch、copy、event、DCR、batch"]
        META["metadata 与能力校验<br/>ABI、XLEN、ISA、资源、dtype"]
        ALLOC["异步/缓存 allocator<br/>stream-ordered free、pinned、统计"]
        LOADER["多镜像 loader<br/>torch、DL、Triton JIT 共存"]
    end

    subgraph BACKENDS["执行后端"]
        SIMX["SimX<br/>功能/周期模型"]
        RTL["rtlsim<br/>Verilator RTL + parity"]
        XRT["XRT FPGA<br/>真实 AFU / DMA / 内存"]
        OTHER["OPAE / FireSim / gem5 / ASIC<br/>按配置提供"]
    end

    subgraph HW["Vortex 硬件"]
        CP["CP"]
        KMU["KMU"]
        CORE["SIMT RISC-V cores"]
        TCU["TCU / FPU / DXA / SFU"]
        CACHE["L1/L2/L3 / LMEM / VM / DRAM"]
    end

    MODEL --> PY
    MODEL --> COMP
    PY --> TV
    COMP --> IV
    TRI --> TVX
    HIP --> HIPRT
    NATIVE --> ABI
    TV --> ABI
    IV --> TVX
    IV --> DL
    TVX --> ABI
    HIPRT --> ABI
    POCL --> ABI
    DL --> VXBIN
    KERN --> VXBIN
    VXBIN --> ABI
    ABI --> CPAPI
    ABI --> META
    ABI --> ALLOC
    ABI --> LOADER
    CPAPI --> SIMX
    CPAPI --> RTL
    CPAPI --> XRT
    CPAPI --> OTHER
    SIMX --> CP
    RTL --> CP
    XRT --> CP
    CP --> KMU --> CORE
    CORE --> TCU
    CORE --> CACHE
~~~

这张图中最重要的不是模块数量，而是**所有上层入口最终汇聚到同一条主干**：

~~~text
PyTorch / Triton / HIP / 原生 C++
        ↓
统一的 kernel 参数和 .vxbin
        ↓
vortex2.h
        ↓
Queue / Event / Buffer / Module / Launch
        ↓
CP 命令环
        ↓
KMU + Vortex SIMT 核心
~~~

这样做的结果是：PyTorch、Triton 和 HIP 不会各自发明一套内存地址、事件、参数打包和 kernel 二进制格式。

---

## 5. 计划具体改变了什么，以及为什么要改

### 阶段 0：基线、环境和 CI

**做什么：**

- 锁定正式支持的 Python、PyTorch、Triton、C++ ABI、VOLT、TOOLCHAIN_REV、XLEN、driver、CONFIGS 和源码 commit。
- 为功能模拟、RTL 和 FPGA 建立独立配置档。
- 把 runtime、HIP、DL 算子、PyTorch eager、Triton、模型和硬件集成测试分层。
- 每次测试保存 manifest、配置快照、镜像 hash、退出码、复现命令和数值误差。
- 把 SimX 与 rtlsim 的 retired instruction 和周期一致性纳入 CI。

**为什么：**

GPU 软件同时依赖编译器、硬件配置、Python 包和后端。没有环境锁定，同一个测试可能只是因为换了 Python 或 build 目录就得到不同结果。阶段 0 让“通过”变成别人能复现、能追责的证据。

---

### 阶段 1：收紧 runtime、ABI、资源和 module loader

**做什么：**

- 让 host/device 共享并校验参数结构的 sizeof、alignof、offsetof 和指针宽度。
- 对超出 4 KiB 的参数块、错误 XLEN、错误 ISA、错误 TCU/dtype、过大的 block/LMEM 在提交前拒绝。
- 完善 queue、stream、event、异步释放、pinned memory、allocator 统计和异常清理。
- 解决多个 .vxbin 固定地址冲突，让 PyTorch kernel、sw/dl kernel 和多个 Triton JIT 镜像可以交替加载。
- 完善 module unload/reload、引用计数和退出顺序。
- 让 hipFuncGetAttribute、Triton autotune 和 PyTorch loader 读取真实设备能力，而不是使用硬编码猜测。

**为什么：**

上层框架最难排查的问题往往不是乘法公式，而是“参数按错偏移”“kernel 还没执行就释放了输入”“两个镜像加载到同一地址”“设备资源超限却静默启动”。这些问题必须在 CP 提交前被发现，否则错误会表现为随机数值、偶发挂死或退出竞态。

---

### 阶段 2：PyTorch eager 单设备推理

**做什么：**

- 完成 torch-vortex 的 PrivateUse1 设备注册、allocator、DeviceGuard、current stream、factory 和错误状态。
- 修复 relu 非原地语义、addmm/mm/linear 转置和 alpha/beta、BatchNorm 通道索引、copy/to、view/reshape/stride/alias 等基础语义。
- 逐步覆盖 sum/mean/max/argmax、bmm、cat/stack、gather/scatter、avg_pool2d、index_add，以及常用 dtype/layout/broadcast/out/in-place overload。
- 让 conv、pool、BN、GEMM、norm、activation 和归约都统一调用 sw/dl，删除 torch-vortex 与 sw/dl 的重复 kernel 语义。
- 接入 Philox RNG、seed/state/fork、checkpoint、state_dict、CPU↔Vortex 权重迁移。
- 按 MiniResNet → 小输入 ResNet-18 → 目标输入大小逐级验收，再进入 tiny Transformer、prefill/decode 和 KV cache。

**为什么：**

PyTorch 的“设备可用”不只是 empty() 能返回一个指针。view 要保持别名关系，copy 要理解 dtype 和 stride，stream 要保证异步工作顺序，RNG 要可复现，checkpoint 要能在新进程恢复。先把这些基础语义做对，模型覆盖才有意义。

---

### 阶段 3：低精度、TCU 和量化

**做什么：**

- 完成 FP16/BF16 的分配、copy/cast、GEMM、conv、norm 和序列化路径。
- 明确舍入、NaN、Inf、subnormal、累加 dtype 和输出 dtype。
- 建立 TCU capability ID 和真实配置构建树。
- 为 TCU GEMM/conv 增加尾 tile、对齐、workspace、LMEM 和资源检查。
- 为 W4A16、INT8、FP8、MXFP8、NVFP4、2:4 sparse 建立稳定的 op/schema、权重打包、scale/zero-point/group size 和 metadata 消费。
- 通过 bit-level golden、模型精度、内存和吞吐报告证明使用了什么格式、什么 kernel、什么硬件单元。

**为什么：**

“代码里有 FP8 类型”不代表设备真的能做 FP8；“配置打开了 TCU”也不代表某次 GEMM 用到了 TCU。计划要求把格式编码、通用 FPU 路径和真实 TCU 加速分开记录，避免把软件模拟或类型转换误报成硬件加速。

---

### 阶段 4：Triton、FakeTensor、FX、export 和 Inductor

**做什么：**

- 实现标准 Triton kernel[grid] (...) launcher，正确传入 Vortex Tensor 的 data_ptr、grid、stream、warp size、XLEN 和资源属性。
- 建立 Triton-Vortex 的正式 lowering：program-id、lane、blocked layout、shared/LMEM、barrier、reduction、FP16/BF16、二维 tile 和 tl.dot。
- 把 tl.dot 映射到 FPU 或 TCU，把异步 copy 映射到 DXA；用真实 capability 约束 autotune 候选。
- 让 Triton kernel 能在同一进程中与 ATen 消费、与 PyTorch stream 同步，并共存多个 JIT 镜像。
- 修复 FakeTensor 和 Meta 路径：只做形状/别名推导，不分配设备内存、不发射真实 kernel。
- 建立 FX backend、torch.export save/load 和有限动态 shape 规则。
- 接入 Inductor 的 device interface、scheduler、codegen、wrapper、runtime launch、fusion、memory planning、编译缓存和错误传播。

**为什么：**

PyTorch eager 是“一个算子一个算子地调用”，适合先做正确性；Inductor 和 Triton 则会在运行前生成新的设备代码。没有标准 launcher 和正式 lowering，Triton 只能靠手写脚本；没有 Fake/Meta，PyTorch 编译器无法在不真正分配设备内存的情况下分析图；没有真实 codegen 证据，torch.compile 可能只是 CPU 或解释器重放。

---

### 阶段 5：模型级交付

**做什么：**

按模型依赖逐步验收：

- ResNet-18：固定 torchvision、权重、预处理、输入尺寸和中间层误差。
- YOLO：补 upsample、动态 batch、检测头和 NMS。
- SAM/DINOv3：补 ViT/window attention、grid_sample 等视觉算子。
- Mamba：完成 selective scan、chunked state continuation 和逐 token 对拍。
- Llama/Qwen：补 embedding、GQA、RoPE、RMSNorm、paged KV、prefill/decode 和 sampling。
- VLM：让视觉塔、projector、图像 token 和 LLM 路径在设备上连接。
- 每个模型记录 checkpoint hash、许可证、输入、算子清单、未支持项、CPU reference、误差、fallback、内存和性能。

**为什么：**

“一个 MiniResNet 前向通过”只能说明一小条路径可用。真实模型会暴露 shape、layout、RNG、KV cache、长序列、内存容量和未覆盖算子问题。分级模型验收可以明确问题出在哪一层。

---

### 阶段 6：XRT、FPGA 和真实性能

**做什么：**

- 完成 AFU、CP、DMA、设备内存和 XRT 的完整路径。
- 按 copy/launch → 单算子 → 小模型 → 目标模型推进。
- 分开统计 warmup、JIT、稳态 latency、throughput、峰值内存、TCU 利用率和内存带宽。
- 为 NT、core、L2/L3、LMEM、A/TCU/DXA 等建立推荐配置。
- 用综合门禁评估 LMEM 扩容、descriptor 偏移、多队列、DMA、SFU、寄存器和缓存调整。

**为什么：**

SimX 上主机程序跑了几秒，不等于 FPGA 上 kernel 就跑了几秒。模拟器用于功能和架构验证，FPGA 才能给出目标板卡上的真实吞吐和资源结果。两者必须分开报告。

---

### 阶段 7：CI、安装和文档交付

**做什么：**

- PR 测语义、ABI、资源负例和 MiniResNet。
- nightly 测更多 shape/dtype、编译路径和 RTL。
- FPGA/性能在专用机器上测试。
- 提供 pyproject、预编译 C++ extension、kernel 资源、SDK/runtime 发现和 ABI/version 检查。
- 在干净虚拟环境安装 wheel+SDK，运行 Tensor、CNN 和 Triton 编译示例。
- 删除个人绝对路径、过时 fallback 描述和不再存在的 API 文档。

**为什么：**

如果安装包仍然依赖源码 checkout、个人 build 目录或 import 时临时编译，别人无法把它当作 SDK 使用。阶段 7 把研究代码变成可安装、可诊断、可持续验证的产品形态。

---

### 阶段 8：训练和多设备生态

**做什么：**

- 审计并实现 AutogradPrivateUse1 的 backward、saved tensor、view/mutation 和版本检查。
- 覆盖 matmul/linear、activation、norm、loss，再扩 conv/attention。
- 实现梯度累加、SGD/Adam、GradScaler、finite/unscale、溢出跳步和混合精度优化器。
- 在 eager backward 正确后接 AOTAutograd/Inductor backward。
- 只有在真实多设备和通信需求出现后，再实现 peer copy、collective、ProcessGroup 和 DDP。

**为什么：**

训练比推理多出梯度图、激活生命周期、随机状态、参数更新和溢出处理。把训练和多卡放在推理闭环之后，可以避免在基础 forward 还不稳定时同时排查十几类问题。

---

## 6. 最终各层的职责边界

### 6.1 用户层：用户会看到什么

完成后，用户大致可以从下面几种入口选择：

~~~python
# PyTorch eager
import torch

x = torch.randn((1024, 1024), device="vortex")
y = torch.nn.functional.relu(x)
z = x @ y
~~~

~~~python
# 计划目标：标准 Triton launcher
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

# 完成标准 backend 后使用 kernel[grid] (...)
~~~

~~~cpp
// 原生 HIP C++ kernel
__global__ void vecadd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
~~~

这些入口看起来不同，但最终都需要：

1. 设备指针和参数按照 Vortex ABI 打包；
2. 通过 VOLT 编译成 RISC-V/Vortex 设备代码；
3. 打包成 .vxbin；
4. 通过 vortex2.h 的 Queue、Module、Kernel、Event、Buffer 提交；
5. 由 CP、KMU 和 SIMT 核心执行。

### 6.2 PyTorch 后端层

torch-vortex 的职责不是重新实现一个 CUDA runtime，而是把 PyTorch 的抽象翻译到 Vortex：

| PyTorch 概念 | Vortex 对应物 |
|---|---|
| torch.device("vortex") | PrivateUse1 设备注册 |
| Tensor storage | Vortex Buffer + allocator |
| current stream | Vortex Queue |
| CUDA/HIP event | Vortex Event timeline |
| ATen operator | sw/dl 或设备 kernel dispatch |
| torch.compile | Inductor Vortex backend |
| FakeTensor/Meta | 只做形状、dtype、stride、alias 推导 |
| Generator | Vortex Philox state/offset |
| state_dict | 设备/主机之间有明确的 storage 迁移语义 |
| Autograd | 经过验证的 Vortex backward kernel 和 saved tensor 生命周期 |

### 6.3 Triton 层

Triton 完成后不只是“能把一段 Python 转成 C”：

~~~text
Triton Python
   ↓
TTIR / TTGIR
   ↓
Vortex layout / shared / barrier / dot lowering
   ↓
LLVM IR / VOLT
   ↓
RISC-V ELF + vxbin
   ↓
标准 launcher → vortex2.h → queue/event
~~~

它必须能：

- 知道当前使用的 Vortex 设备和 stream；
- 使用真实的设备 tensor 指针；
- 按 Vortex 的 warp、LMEM、寄存器和 block 限制做 autotune；
- 在错误 dtype、shape、资源超限时明确失败；
- 在同一进程共享 buffer、stream 和 event；
- 让 Inductor 产生的代码真正被设备执行，而不是用 CPU 解释器或 FX replay 冒充。

### 6.4 sw/dl 层

sw/dl 会成为**唯一的可复用深度学习设备 kernel 库**，大致包括：

- BLAS：GEMM、batched GEMM、linear epilogue；
- primitive：elementwise、reduce、cast、copy、broadcast；
- DNN：conv、pool、batch norm、layer norm、RMSNorm；
- attention/LLM：RoPE、softmax、SwiGLU、KV append/read、sampling；
- RNG：Philox 和设备随机状态；
- quant：INT8、FP8、W4A16、MX/NVFP4、2:4 sparse；
- Mamba：selective scan 和 state update。

PyTorch、HIP 和 Triton 可以调用这些 kernel，但它们不应该各自维护一份 GEMM、conv、norm 的行为定义。这样才能避免一个算子在 PyTorch 中正确、在 HIP 中错误、在 Triton 中又是另一套参数布局。

---

## 7. “原始”与“完成后”的直接对比

| 方面 | 计划前的基线 | 计划全部完成后的目标 |
|---|---|---|
| 主要定位 | RISC-V GPGPU、OpenCL/Vulkan/原生 kernel 平台 | 可安装的 GPGPU + 深度学习推理/训练平台 |
| HIP | chipStar → PoCL 的兼容路径，加上原生 HIP 子集原型 | 兼容路径保留；原生 hipcc-vortex + libhip_vortex 成为正式入口 |
| 设备编译 | VOLT → ELF → vxbin 已存在 | HIP、Triton、PyTorch 生成的 kernel 全部统一走这条链 |
| runtime | 已有 Device/Buffer/Queue/Event/Module 基础 | 参数、资源、异步生命周期、allocator、多镜像和错误契约完整 |
| DL kernel | sw/dl 有多种 kernel，但接入不完整 | PyTorch/HIP/Triton 共享统一 kernel 库和 ABI |
| PyTorch | PrivateUse1 原型、FP32 eager 子集、MiniResNet 资产 | eager 单设备推理、RNG/checkpoint、Fake/FX/export、Inductor、再到 Autograd |
| Triton | TTIR→C 原型、手工 launcher | 标准 kernel[grid]、正式 lowering、tl.dot、autotune 和 Inductor 互操作 |
| 低精度 | 软件编码和部分 kernel | 能力 gate、bit-level golden、PyTorch/torchao 接入、TCU 证据和模型报告 |
| 模型 | 局部算子和小模型 | ResNet、检测、ViT、Mamba、LLM/VLM 按阶段验收 |
| 验证 | SimX 为主，部分 rtlsim/回归 | SimX 功能 + rtlsim parity + XRT 集成 + FPGA 性能 |
| 安装 | 开发者需要 build 树和工具链 | wheel + SDK + runtime 发现，干净环境可安装 |
| CI | 传统 regression 加阶段性 DL 测试 | 语义、ABI、编译、模型、RTL、FPGA、性能分层门禁 |
| 训练 | 不支持或没有完成验证 | 在 forward 稳定后，逐算子实现 Autograd、优化器、AMP 和可选多设备 |

---

## 8. 一个 PyTorch 请求在最终系统里如何运行

以：

~~~python
y = torch.nn.functional.linear(x, weight, bias)
~~~

为例，计划完成后的路径会是：

~~~mermaid
sequenceDiagram
    participant P as PyTorch
    participant TV as torch-vortex
    participant DL as sw/dl
    participant V as vortex2.h
    participant CP as CP/KMU
    participant D as Vortex device

    P->>TV: dispatch 到 PrivateUse1 linear
    TV->>TV: 检查 dtype、shape、stride、device、stream
    TV->>DL: 选择 GEMM/linear kernel 变体
    DL->>V: 准备 vx_launch_info 与参数 blob
    V->>V: 校验 ABI、资源、ISA、TCU capability
    V->>CP: 写入 copy/launch/event 命令
    CP->>CP: DMA 参数并编程 KMU
    CP->>D: 启动 CTA
    D->>D: 读取 x/weight，执行 FPU 或 TCU，写回 y
    D-->>CP: 完成并更新 sequence/event
    CP-->>V: queue/event 完成
    V-->>TV: 返回设备 Tensor
    TV-->>P: 返回 y
~~~

这里有几个关键点：

- PyTorch 不直接访问 RTL，也不直接操作 CP 寄存器。
- sw/dl 负责 kernel 和参数定义，torch-vortex 负责 PyTorch 语义。
- runtime 负责异步顺序和生命周期，避免 weight 在 kernel 完成前被释放。
- 设备能否使用 TCU，要由 capability 和 metadata 决定；不能仅凭 dtype 名称猜测。
- 如果某个 layout 或 dtype 尚未支持，应该在 launch 前报出可读错误。

---

## 9. 你最终可以期待什么，不能自动期待什么

### 可以期待的事情

完成计划并通过相应验收后，可以期待：

1. 用统一的 Vortex SDK 编译和运行原生 C/C++、HIP、OpenCL、Vulkan、Triton 和 PyTorch 程序。
2. 在 SimX 上快速验证功能，在 rtlsim 上检查 RTL 行为，在 XRT FPGA 上测真实硬件。
3. 让 PyTorch、Triton 和 HIP 共享设备内存、stream、event、module loader 和 kernel ABI。
4. 对常见 CNN、Transformer、LLM/VLM 和低精度 kernel 得到可重复的正确性、内存和性能报告。
5. 通过 CI 发现参数 ABI 漂移、资源超限、模型 parity、性能回归和安装问题。
6. 在明确的支持矩阵内使用 eager、export、Triton 和 Inductor。

### 不能自动期待的事情

即使完成全部计划，也不应把它理解成：

- 所有 CUDA 程序都能不改代码运行；
- 所有 HIP API、CUDA library、cuDNN、NCCL、cuBLAS 都完全兼容；
- 所有 PyTorch ATen operator、所有 dtype、所有 layout、所有动态 shape 都已支持；
- 任意 Triton 程序都能直接编译；
- SimX 上的执行时间就是 FPGA 或 ASIC 的性能；
- 开启一个 TCU 配置就代表所有 GEMM 都由 TCU 执行；
- 完成推理就自动意味着 Autograd、优化器、AMP 和多卡训练已经可靠；
- 有 CPU fallback 的结果就可以替代真实设备路径的验收。

Vortex 的兼容性最终应以“支持矩阵 + 测试证据 + 错误边界”为准，而不是以某个示例程序是否偶然跑完为准。

---

## 10. 对初学者最重要的使用顺序

如果你以后要使用这个项目，建议按下面的顺序理解：

1. **先把 Vortex 当成设备平台**  
   学会 configure、build 目录、VORTEX_DRIVER=simx 和 ci/blackbox.sh。

2. **再理解一条原生 kernel 路径**  
   从 vecadd 看：设备 kernel → VOLT → .vxbin → vortex2.h → CP → SimX。

3. **再理解 HIP**  
   HIP 只是更熟悉的 C++ kernel 和 runtime 表面；它最终仍然调用 Vortex 的 queue、buffer、module 和 launch。

4. **再理解 sw/dl**  
   它是可复用的设备算子库，不是 PyTorch 本身。

5. **再理解 PyTorch eager**  
   torch-vortex 把 ATen 的 Tensor/operator 语义翻译到底层 Vortex kernel。

6. **最后理解 Triton 和 Inductor**  
   它们会动态生成 kernel，因此需要完整的 compiler、launcher、cache、resource query 和多镜像 loader。

简单地说：

~~~text
先学会“设备怎样执行一个 kernel”
再学会“HIP 怎样提交这个 kernel”
再学会“PyTorch 怎样把一个算子变成这个 kernel”
最后学会“Triton/Inductor 怎样自动生成和组合这些 kernel”
~~~

---

## 11. 最终交付时应查看哪些证据

判断“计划是否真的完成”，建议查看这些证据，而不是只看 README 中的宣传语：

- 环境 manifest：Python、PyTorch、Triton、VOLT、TOOLCHAIN_REV、XLEN、CONFIGS、driver、commit。
- runtime 测试：参数 ABI、metadata、stream/event、提前释放、module reload、错误退出。
- kernel 测试：SimX 正确性、rtlsim retired-instruction parity、周期容差。
- PyTorch 测试：dtype、layout、broadcast、alias、特殊值、RNG、checkpoint、模型中间层。
- Triton 测试：标准 launcher、设备指针、跨 stream、多个 JIT 镜像、非法 dtype/shape。
- 编译测试：FakeTensor、FX、export save/load、真实 Inductor codegen。
- 低精度测试：bit-level golden、scale/zero-point、尾部 shape、TCU capability 和真实计数器。
- 模型报告：checkpoint hash、输入、算子清单、误差、fallback、内存、吞吐和 latency。
- FPGA 报告：板卡、bitstream、时钟、内存、资源、warmup、稳态和带宽。
- 安装测试：干净虚拟环境中 wheel + SDK，不依赖源码 checkout 或个人绝对路径。

---

## 12. 最后用一句话记住整体变化

完成 overview_plan.md 后，Vortex 不再只是“有一个 GPU 核心和几个模拟器的研究平台”，而会成为：

> **从 Python/PyTorch、Triton、HIP、OpenCL 到 Vortex RTL/FPGA 的统一、可验证、可安装的深度学习 GPGPU 软件栈；上层入口可以不同，但内存、队列、事件、参数 ABI、.vxbin 和硬件执行路径保持统一。**

这个目标仍然受硬件资源、已验证 dtype、支持矩阵和具体模型算子集合限制。真正的完成标准是每个能力都有对应实现、错误边界、测试证据和可复现命令。

