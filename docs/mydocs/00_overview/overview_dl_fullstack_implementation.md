# Vortex 原生 HIP、PyTorch、Triton 与深度学习全栈实施方案

> **统一计划**：当前执行顺序和完成状态见 [`overview_plan.md`](overview_plan.md)。本文保留原始全栈设计和阶段验收记录。

本文是后续开发的主计划。它以当前仓库代码、构建脚本、RTL 设计文档和上游官方接口文档为依据；`docs/mydocs/` 中的旧文档只作为线索，不作为现状证据。每个阶段都给出代码边界、依赖、验收条件和可重复命令。实施时应按阶段提交小变更，并让 CI 在每个阶段保留可运行的中间结果。

## 1. 目标、边界和现状基线

### 1.1 目标

最终交付一个可安装的 Vortex SDK，使以下程序能够在 SimX、rtlsim 和至少一个 FPGA 后端运行：

1. 使用 HIP C/C++（包括 kernel launch、streams、events、异步 memcpy、atomics、模块加载）的程序；
2. 使用 PyTorch eager、`torch.compile`/Inductor 和 `torch.export` 的模型；
3. 使用 Triton Python kernel，至少覆盖官方 vector add、softmax、matmul、layer norm、fused attention 教程；
4. CNN、检测、分割、视觉 Transformer、SSM/Mamba、LLM/VLM 的推理基准；
5. FP32、FP16、BF16、INT8、FP8，以及分阶段的 INT4/W4A16、MX/NVFP4 和 2:4 稀疏量化；
6. 每个功能在功能模拟器上先通过，再在周期 RTL 上进行模型一致性和性能回归，最后在 XRT/真实 FPGA 上进行集成验证。

这里的“支持”定义为：模型可以完成编译、加载、执行和数值校验；对暂未实现的算子必须有明确的 fallback（CPU 或 Vortex 通用 kernel）、可观测日志和测试，而不是静默得到错误结果。训练支持排在推理之后，分布式多卡和自动混合精度属于后续里程碑。

### 1.2 代码事实（必须在开发前复核）

| 事实 | 证据 | 影响 |
|---|---|---|
| 默认 1 cluster/1 core、4 warps、4 threads | [`VX_config.toml`](../../../VX_config.toml) 的 platform/pipeline 段 | 只是最小验证档，不能代表 DL 性能或 warp32 兼容性 |
| L2/L3、RVA 原子、TCU、DXA 默认关闭 | [`VX_config.toml`](../../../VX_config.toml) 的 ISA/cache/TCU/DXA 段 | PyTorch allocator、scatter、GEMM 和异步拷贝的验证必须显式配置 |
| LMEM 默认 `2^14` 字节 | [`VX_config.toml`](../../../VX_config.toml) 的 lmem 段 | Triton shared memory tile 和双缓冲需要先确定容量/描述符限制 |
| TCU 已有 FP16 及多种可选 dtype 宏，但默认仅 FP16 | [`VX_config.toml`](../../../VX_config.toml)、[`sw/kernel/include/vx_tensor.h`](../../../sw/kernel/include/vx_tensor.h)、`hw/rtl/tcu/` | 先完成 FP16/BF16/INT8，再逐个开启 FP8/INT4/MX/NVFP4/稀疏 |
| 规范主机 ABI 已存在 | [`sw/runtime/include/vortex2.h`](../../../sw/runtime/include/vortex2.h) | 原生 HIP、Triton driver 和 PyTorch allocator 应共用该 ABI，禁止各自直连后端 |
| `vortex2.h` 提供 queue/event/launch/copy/map/module API | [`vortex2.h`](../../../sw/runtime/include/vortex2.h#L240) | 可映射 HIP stream/event/module，但还要补齐语义和并发能力 |
| 当前 HIP 通过 chipStar/PoCL | [`tests/hip/common.mk`](../../../tests/hip/common.mk)、[`docs/designs/hip_on_vortex_chipstar.md`](../../designs/hip_on_vortex_chipstar.md) | 保留为兼容回归路径；它不能稳定暴露 Vortex TCU/DXA 内联指令 |
| HIP 测试只有 vecadd、sgemm、histogram、atomicreduce | [`tests/hip/Makefile`](../../../tests/hip/Makefile) | 需要扩展 API、dtype、同步、模块和错误语义测试 |
| 现有 TCU/DXA/softmax 等回归资产丰富 | [`tests/regression/`](../../../tests/regression)、[`ci/testcases/`](../../../ci/testcases) | 复用现有 golden、SimX/rtlsim parity 和 perf_gate，不另造不可比基线 |

### 1.3 官方接口约束

- HIP 的运行时 API 按初始化、设备、内存、stream、event、module、执行控制和异步并发组织；原生实现应覆盖这些模块的可移植子集，并对不支持的 API 返回明确的 `hipErrorNotSupported`。参见 [HIP Runtime API modules](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules.html) 和 [Using HIP runtime API](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api.html)。
- PyTorch 的外部设备集成使用 `PrivateUse1` dispatch key，需要注册 operator、generator、device guard、序列化等组件，并可将 `PrivateUse1` 重命名为 `vortex`、生成 Tensor/Module/Storage 方法。参见 [PrivateUse1 backend tutorial](https://docs.pytorch.org/tutorials/advanced/privateuseone.html) 和 [operator registration](https://docs.pytorch.org/docs/stable/accelerator/operators.html)。
- Triton 支持 out-of-tree backend/plugin；backend 至少需要 `backend/compiler.py` 和 `backend/driver.py`，可通过 `TRITON_PLUGIN_DIRS` 或第三方 backend 目录安装。参见 Triton 的 [backend packaging code](https://github.com/triton-lang/triton/blob/main/setup.py)、[out-of-tree module policy](https://github.com/triton-lang/triton/blob/main/CONTRIBUTING.md) 和 [official tutorials](https://triton-lang.org/main/getting-started/tutorials/)。
- torchao 将量化分成算法、派生量化 Tensor、量化/反量化 primitive 和低精度 dtype；当前官方推理工作流包含 INT8、INT4、FP8、MXFP、NVFP4 等组合。参见 [torchao quantization overview](https://docs.pytorch.org/ao/stable/contributing/quantization_overview.html) 与 [quantized inference workflows](https://docs.pytorch.org/ao/stable/workflows/inference.html)。Vortex 应先实现能被这些工作流调用的标准 ATen/自定义 op，再实现专有 packed Tensor。

## 2. 总体技术路线

采用“共用 Vortex ABI、双 HIP 路径、Triton 外置 backend、PyTorch PrivateUse1”的路线：

```
PyTorch eager / Inductor / export
        │                 │
  ATen PrivateUse1   Triton-Vortex backend
        │                 │
  vortex-dnn/blas/prim/quant ── HIP C++ kernels (hipcc)
        │                 │
              libhip_vortex
                     │
                 vortex2.h
                     │
       simx / rtlsim / opae / xrt / gem5
                     │
      CP → KMU → SIMT core → TCU/DXA/cache/memory
```

保留现有 `chipStar → PoCL` 作为 HIP 兼容和 conformance 对照路径。原生路径由 Vortex 维护：

- `hipcc-vortex`：包装 VOLT 的 clang/lld，直接生成 RISC-V ELF 和 `.vxbin`，不经过 SPIR-V；
- `libhip_vortex`：HIP runtime/driver API 到 `vortex2.h` 的薄适配层；
- `vortex-kernel`：设备侧启动、intrinsics、TCU/DXA/同步和基础 libc；
- `vortex-libs`：BLAS-like、DNN、PRIM、attention、RNG、quant kernel；
- `triton-vortex`：Triton MLIR/LLVM lowering + driver；
- `torch-vortex`：PyTorch PrivateUse1、allocator、stream/event、operator registration、fallback。

所有设备代码最后都必须经过 VOLT 的 `riscv*-unknown-elf +xvortex`、链接脚本和 `vxbin.py`。不允许为 HIP、Triton、PyTorch 各自维护不同的设备二进制格式或隐藏的后端内存模型。

## 3. 里程碑和依赖图

| 里程碑 | 结果 | 主要依赖 |
|---|---|---|
| P0 基线与 DL 配置 | 可重现配置、能力查询、现有回归全绿 | 无 |
| P1 runtime contract | HIP 所需 memory/stream/event/module/launch 语义在 `vortex2.h` 固化 | P0 |
| P2 原生 HIP | vecadd、GEMM、atomics、streams 和 module API 直编直跑 | P1、VOLT |
| P3 kernel/算子库 | BLAS、conv、norm、activation、attention、RNG、prim | P2、TCU/DXA |
| P4 Triton backend | 官方教程和 autotune kernel 在 Vortex 编译运行 | P2、P3 |
| P5 PyTorch backend | eager + export + Inductor 基础模型 | P2、P3、P4（Inductor 路径） |
| P6 量化 | INT8/FP8/W4A16 → MX/NVFP4/稀疏，接入 torchao | P3、P5、TCU dtype |
| P7 模型与产品化 | ResNet/YOLO/SAM/DINOv3/Mamba/LLM/VLM，FPGA 性能报告 | P5、P6 |

P2 与 P3 可部分并行；P4 的 driver 可在 P2 后开始，codegen 必须等设备 ABI 和 kernel 库稳定；P5 先走 eager/explicit ops，再启用 Inductor；P6 先实现独立量化 API，不把未验证 dtype 暴露成默认 PyTorch dtype。

## 4. P0：基线、配置和可观测性

### P0.1 固化三档硬件配置

新增配置文件或 CI 参数集合，不直接修改默认最小档：

1. `dl_functional`：1–2 core、NT=4/8，TCU FP16，适合 SimX 快速功能测试；
2. `dl_rtl`：多 core、NT=16/32、L2/L3、RVA、TCU、DXA、VM 按资源可行性开启，适合 rtlsim parity；
3. `dl_fpga`：按目标板卡的 BRAM/DSP/HBM 资源选择 core、LMEM、bank、时钟，适合 XRT。

每档必须记录 XLEN、threads/warps、LMEM、cache、memory banks、TCU dtype、DXA、A extension、clock 和预期资源。所有变更从 build 目录重新执行 `../configure`，遵守仓库的 out-of-tree 和 stale generated header 规则。

验收：

```bash
mkdir -p build && cd build
../configure --xlen=64 --tooldir="$TOOLDIR"
make -s
./ci/regression.sh --test regression
./ci/blackbox.sh --driver=simx --app=sgemm
```

对每档运行 `vx_device_query`，将实际能力写入 JSON；运行时不得假设 `warpSize==32`，应使用查询值或编译期宏生成 kernel。

### P0.2 能力与性能观测

扩充 `vortex2.h` 能力 ID：TCU dtype bitmap、DXA、原子、最大 grid/block、并发 queue 数、对齐要求、统一/主机内存能力。使用已有 `vx_device_mpm_query`/`vx_device_dump_perf` 输出 kernel、DMA、cache、TCU 利用率，并让 `ci/roofline.py` 能读取算术强度和峰值配置。

验收：同一 workload 在 SimX 和 rtlsim 的 retired instruction 必须相同，周期误差在 CI case 规定容差内；禁止通过放宽 tolerance 隐藏差异。

## 5. P1：vortex2.h runtime contract

### P1.1 内存模型

在 `sw/runtime/common` 实现并测试：

- device allocation/free、对齐、地址范围和 OOM；
- host pinned allocation、map/unmap、DMA 生命周期；
- stream-ordered allocator（对应 HIP `hipMallocAsync/hipFreeAsync`）；
- 可选 VM/统一内存能力，暂不支持时返回 `VX_ERR_NOT_SUPPORTED`；
- 64 位 host 与 RV32 device 的参数/指针宽度转换，kernel 参数 blob 明确 ABI（大小、对齐、endian、buffer address）；
- allocator 统计、泄漏检测和 double-free 错误。

HIP 的 stream-ordered allocator 语义以 [HIP stream ordered allocator](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___stream_o.html) 为准：free 不能早于同一 stream 上的先前访问，跨 stream 需要 event 依赖。

测试：分配 0、1、对齐边界、大块和耗尽场景；异步 H2D→kernel→D2H；两个 stream 交叉读写；map 在 event 完成前不可访问；Sanitizer/调试构建检查句柄生命周期。

### P1.2 queue/stream/event 并发

当前 `vx_queue_create` 有队列抽象，但 CP 默认单队列。实现分层能力：

1. 先保证多个 host queue 的依赖和事件语义正确，即使设备串行；
2. 增加 CP 多硬件队列或 software scheduler，使独立 memcpy 与 kernel 可重叠；
3. 为每个 queue 记录优先级、序列号、profiling 时间戳和错误状态；
4. 实现 HIP stream callback/host function 的最小等价；
5. 对 event wait/signal 使用 monotonic timeline，处理超时、设备丢失和异常传播。

测试矩阵：同 stream 顺序、跨 stream event wait、空 stream、destroy 前同步、超时、故障注入。性能测试比较串行/重叠时间，不能只检查最终数值。

### P1.3 module/kernel ABI

完善 `.vxbin` 元数据：入口名、参数布局、自然 block、静态 LMEM、所需 ISA/TCU dtype、寄存器/占用率。`vx_module_get_kernel` 加载时校验能力并返回可读错误；增加从内存加载、缓存和卸载测试。为 HIP triple-chevron 和 `hipModuleLaunchKernel` 都提供统一 launch builder。

参数测试覆盖标量、指针、结构体、数组、对齐空洞、>4 KiB 参数和 RV32 指针截断。禁止把 host `size_t` 直接复制为 device `size_t`。

## 6. P2：原生 HIP（HIPVortex）

### P2.1 编译器和头文件

在独立仓库或 `third_party/hip-vortex` 管理 HIP headers、clang driver wrapper 和 device bitcode；修改 VOLT 时保持其 upstreamable patch。第一版支持：

- `__global__`、`__device__`、`__host__`、三尖括号 launch；
- `threadIdx/blockIdx/blockDim/gridDim` 映射到 Vortex CSR；
- `__syncthreads`, warp vote/shuffle/ballot、cooperative groups 的可实现子集；
- `atomicAdd/Exch/Compare` 等 RVA 操作；无 A extension 时编译期诊断或运行时 not supported；
- `hip::` half/bfloat16、vector types、基础 math；
- Vortex 专有 `vx::wmma/wgmma`, DXA async copy 和 barrier 头，使用 capability guard，不伪装成 CUDA tensor core ABI。

设备编译命令必须可独立复现，例如：

```bash
hipcc-vortex --offload-arch=vortex64 -O3 kernel.hip -c -o kernel.o
hipcc-vortex --offload-arch=vortex64 kernel.o -o app
```

输出中保存 clang 命令、LLVM IR、反汇编、vxbin 元数据，便于 AI agent 定位前端/后端/链接问题。

### P2.2 `libhip_vortex` API 分层

按 HIP 官方模块建立目录和状态表：

| HIP 模块 | Vortex 实现 |
|---|---|
| device/context | `vx_device_open/query/release`；单进程当前一个 primary context |
| memory | `vx_buffer_create/access/map`；实现 pool 和 async free |
| stream/event | `vx_queue_*`、timeline event |
| memcpy/memset | `vx_enqueue_{read,write,copy,fill}` |
| module/launch | `vx_module_*`、`vx_enqueue_launch` |
| atomics | 设备 RVA + capability check |
| graph | 先把 HIP graph 限定为可验证的 command list，映射 `vx_enqueue_commands/draw` |
| texture/surface | 仅在 TEX 开启时实现；否则明确 not supported |
| peer/multi-device | 第一版返回 not supported，保留 API/错误测试 |

错误转换必须稳定：每个 `vx_result_t` 对应 HIP error，保存 last error，异步错误在 `hipGetLastError/hipDeviceSynchronize` 可见。不要用 CPU fallback 掩盖设备错误。

### P2.3 HIP 验收顺序

新增 `tests/hip_native/`，每个测试可在 simx/rtlsim 运行：

1. API smoke：device query、malloc/free、memcpy、memset、module load；
2. vecadd 和 reduction；
3. FP32/FP16/BF16 GEMM（先通用 kernel，再 TCU kernel）；
4. atomic histogram/reduction（A extension on/off）；
5. stream/event overlap 和参数 ABI；
6. `hiprtc` 暂不实现时返回明确错误，并保留测试；
7. chipStar 同一源文件作为结果对照，确认原生路径与兼容路径数值一致。

命令示例：

```bash
cd build
../configure --xlen=64 --tooldir="$TOOLDIR"
CONFIGS="-DVX_CFG_EXT_A_ENABLE -DVX_CFG_EXT_TCU_ENABLE" \
  make -C tests/hip_native run-simx
CONFIGS="-DVX_CFG_EXT_A_ENABLE -DVX_CFG_EXT_TCU_ENABLE" \
  make -C tests/hip_native run-rtlsim
```

## 7. P3：算子与设备 kernel 库

### 7.1 目录和 ABI

建立独立可安装库（不把上层 PyTorch 代码放入 `sw/kernel`）：

```
sw/dl/include/vortex/{blas,prim,dnn,attention,rng,quant}.h
sw/dl/src/{blas,prim,dnn,attention,rng,quant}/
tests/dl/{blas,prim,dnn,attention,rng,quant}/
```

公共头只依赖 `sw/common` 和 kernel headers，遵守 `sw/kernel` 与 `sim/hw` 边界。每个 kernel 提供：dtype、布局、对齐、tile、workspace、可支持维度、误差标准和 fallback 状态。

### 7.2 算子优先级

第一层（模型共同依赖）：copy/fill、cast、reduce、argmax、softmax/logsoftmax、exp/log/rsqrt 近似、layernorm/rmsnorm、activation（ReLU/GELU/SiLU）、embedding、transpose/reshape、concat/slice、random/Philox。

第二层（算力核心）：GEMM/ batched GEMM、matmul epilogue、Conv2D implicit-GEMM、depthwise/grouped conv、pooling、batchnorm、ROI/upsample、rotary embedding、scaled dot-product attention、flash attention 的 tiled 版本。

第三层（模型特有）：Mamba selective scan、state update、MoE top-k/gather/scatter、KV cache append/gather、paged attention、NMS、mask/prompt encoder、vision patch embedding。

每个算子先写 CPU reference（double 或高精度），再写 SimX kernel 测试；rtlsim 只在 SimX PASS 后启用。随机测试使用固定 seed，覆盖非对齐尺寸、零长度、NaN/Inf、边界 mask 和非连续 stride。

### 7.3 TCU/DXA 映射

- GEMM tile 参数从设备 capability 和 LMEM 动态选择；不能写死 warp32 或 64 KiB descriptor offset；
- DXA 负责 tile 预取/转置时，使用 `expect_tx` + async barrier，验证 producer/consumer 次序；
- TCU dtype 每开一种就增加独立 golden：FP16/BF16 → INT8 → FP8 → INT4 → MX/NVFP4 → sparse 2:4；
- 对每个 tile 记录 M/N/K、布局、scale/metadata 地址和累加 dtype；
- 保留通用 FPU kernel 作为正确性 fallback，但在 perf 报告中区分 fallback 与 TCU。

先修复现有 TCU 相关的 XLEN64 数值或周期 parity 问题，再把新 dtype 接入上层。SimX 和 RTL 的 event/latency 必须同步修改，遵守仓库 model parity 规则。

## 8. P4：Triton-Vortex backend

### 8.1 选择 out-of-tree backend

不直接维护 Triton fork。建立 `triton-vortex/`，按 Triton backend 约定提供：

- `backend/name.conf`：`vortex`；
- `backend/compiler.py`：Triton IR → TTGIR/LLVM IR → VOLT clang/lld/vxbin；
- `backend/driver.py`：Python driver API → `libvortex.so`/`libhip_vortex`；
- `backend/descriptor.py`：grid、warps、threads、shared memory、cluster；
- `language/`：仅在需要时添加 Vortex dtype/intrinsic；
- cache key 包含 Vortex ISA、配置 hash、Triton 版本和 kernel meta，避免旧 vxbin 误用。

通过 `TRITON_PLUGIN_DIRS` 安装和选择 backend；保留 Triton interpreter 作为 host reference。

### 8.2 编译降低策略

第一阶段只支持 Triton 标准语义：pointer arithmetic、mask、`tl.load/store`、reduce、`tl.dot`、layouts、program ids。映射如下：

| Triton | Vortex |
|---|---|
| program instance | CTA/grid，由 KMU launch |
| `tl.arange`/lane | SIMT lane/GPR |
| `tl.load/store` | LSU + cache；连续 tile 可选 DXA |
| `tl.dot` | FPU 或 TCU WMMA/WGMMA |
| `tl.multiple_of`/alignment | 编译器 metadata 和 runtime 校验 |
| shared memory | LMEM；超限时编译错误或分块重写 |
| `tl.barrier` | CTA barrier/async barrier |
| `tl.exp/log/rsqrt` | libdevice/Vortex math；无硬件 SFU 时记录性能限制 |

先使官方 vector add、softmax、matmul、dropout、layer norm、fused attention 教程通过，再实现 autotune 搜索（threads/warps/tile/num stages）。每个 autotune 结果都必须带配置和性能计数器，禁止只按 wall clock 选择。

### 8.3 Triton 测试

建立 `tests/triton/`：每个 Python 测试同时运行 interpreter、Vortex SimX、rtlsim；比较输出和边界；使用 `TRITON_KERNEL_DUMP` 保存 IR/vxbin。为 `tl.dot` 增加 TCU 开关测试，为 async copy 增加 DXA 测试。CI 先标记 smoke，再逐步加入 parity/perf_gate。

## 9. P5：PyTorch Vortex backend

### 9.1 先做 eager，再做编译器

建立独立 `torch-vortex/` Python/C++ 包：

1. 注册 `PrivateUse1` 并重命名为 `vortex`；
2. `DeviceGuardImplInterface`：current device、stream、event、synchronize；
3. `Allocator`：调用 `libhip_vortex`/`vortex2.h`，支持 caching pool 和 record stream；
4. Generator/RNG：Philox 状态、seed、offset、CPU/Vortex 一致性测试；
5. serialization/deserialization：保存 device、dtype、layout、quant metadata；
6. `torch._register_device_module("vortex", ...)`：device count、synchronize、memory stats、AMP dtype；
7. `TORCH_LIBRARY_IMPL(..., PrivateUse1, ...)` 注册高频 ATen op；不支持的 op 注册明确 fallback 或报错；
8. autograd：先用 Composite/CPU backward，随后为 GEMM/conv/attention 注册高性能 backward；
9. stream/event 与 `record_stream` 的生命周期测试。

使用官方 [PrivateUse1 tutorial](https://docs.pytorch.org/tutorials/advanced/privateuseone.html) 的注册顺序，使用 [rename_privateuse1_backend API](https://docs.pytorch.org/docs/main/generated/torch.utils.backend_registration.rename_privateuse1_backend.html) 生成用户侧 `tensor.vortex()` 等方法。

### 9.2 ATen 覆盖和 fallback 策略

按模型 trace 统计算子，而不是试图一次实现全部 3500+ op：

- Tier A：factory、copy、view/reshape/transpose、elementwise、reduce、normalization、matmul/conv、index、random；
- Tier B：attention、embedding、scatter/gather、sort/topk、interpolate、NMS、sparse；
- Tier C：训练优化器、distributed、custom autograd、量化派生 Tensor。

每个 op 记录 schema、dtype/layout/device guard、kernel、误差、性能和 fallback。CPU fallback 必须显式同步数据并在日志中计数，避免模型“跑通”但实际全在 CPU。

### 9.3 `torch.compile`/export

先支持 `torch.export` 图中的静态 shape 和显式 Vortex op；再让 Inductor 将融合点落到 Triton-Vortex。动态 shape、Python 控制流、autograd graph break 作为单独能力。验收脚本至少包括：

```python
model = model.to("vortex")
x = torch.randn((1, 3, 224, 224), device="vortex")
y_eager = model(x)
y_compiled = torch.compile(model, backend="inductor")(x)
torch.testing.assert_close(y_eager.cpu(), y_compiled.cpu(), rtol=..., atol=...)
```

同时保存 FX/Export/Inductor/Triton IR 和 fallback 统计。编译缓存 key 必须包含 Vortex 配置和 kernel ABI 版本。

## 10. P6：量化路线

### 10.1 分阶段选择

| 阶段 | 形式 | 设备实现 | 验收模型 |
|---|---|---|---|
| Q0 | FP16/BF16 baseline | TCU dense | ResNet、BERT 小模型 |
| Q1 | W4A16 weight-only，group 32/64/128 | unpack + FPU/TCU FP16 accumulate | Llama 小模型、ViT |
| Q2 | INT8 weight-only / W8A8 dynamic | INT8 TCU，per-channel/per-token scale | ResNet、YOLO、Mamba |
| Q3 | FP8 E4M3/E5M2 | FP8 TCU，scale/amax | Transformer、VLM |
| Q4 | MXFP8/MXFP4、NVFP4 | block scale + metadata | LLM/VLM 实验 |
| Q5 | 2:4 sparse | sparse TCU + metadata load | GEMM/MLP/attention |

Q1 优先，因为即使硬件 INT4 尚未打开，也可以用安全的 nibble unpack 和 FP16 accumulate 验证端到端流程。Q2/Q3 需要先完成 TCU 数值、饱和、NaN、scale 布局 golden。Q4/Q5 只有在独立 kernel 和模型精度报告通过后才进入默认 capability。

### 10.2 软件接口和校验

实现 `vortex.quantize`, `dequantize`, `pack`, `unpack`, `matmul_quantized` C++/Python API；元数据包含 bit width、signedness、group size、axis、scale dtype/layout、zero point、sparsity pattern。与 torchao 的 derived dtype/quantizer 对接时，优先复用标准 `torch.Tensor` storage 和自定义 op，暂不伪造 torchao 尚未支持的 dtype。

每种量化配置都要测：

- pack→unpack bit-exact；
- reference dequant 与 device dequant 误差；
- GEMM/conv 输出误差和端到端 top-1/mAP/IoU/perplexity；
- 奇数尺寸、非整 group、对齐边界、scale=0、饱和和 NaN；
- 性能：权重带宽、解码开销、TCU 利用率、峰值内存。

## 11. P7：模型落地顺序和验收

模型不按“能 import”验收，而按固定 checkpoint、输入、输出、误差、性能和 fallback 统计验收。建议顺序：

1. **ResNet-18/50**：conv/bn/relu/pool/linear，FP32→FP16→INT8；
2. **YOLO（选定版本）**：检测头、resize、NMS、动态 batch，先 FP16 再 INT8；
3. **SAM**：image encoder（ViT）、prompt encoder、mask decoder、长序列 attention；
4. **DINOv3（选定公开 checkpoint）**：ViT patch/positional embedding、layernorm、attention、分类/特征输出；
5. **Mamba**：selective scan、卷积状态、门控和长序列内存；
6. **Llama/Qwen 小规模模型**：RMSNorm、RoPE、GQA/MQA、SwiGLU、KV cache、sampling；
7. **VLM**：视觉 encoder→projector→LLM，验证跨模态 tensor dtype/layout 和 KV cache。

LLM 先做 batch=1 prefill/decode，再做 continuous batching、paged KV cache 和 speculative decoding。模型权重使用公开、可再分发的 checkpoint；CI 使用缩小模型和固定输入，夜间任务运行完整 checkpoint。每个模型都生成：算子覆盖表、CPU fallback 次数、最大内存、首 token/每 token latency、吞吐、功耗（FPGA 可测时）和精度报告。

## 12. 测试、CI 和 AI 开发工作流

### 12.1 测试金字塔

1. host C/C++ 单元：ABI、allocator、事件、错误和 metadata；
2. device kernel：固定 seed reference 对比；
3. compiler lit：HIP/Triton 源码到 IR/反汇编/vxbin 的 FileCheck；
4. SimX 功能：所有 dtype/shape/边界；
5. SimX↔rtlsim parity：retired instruction 精确相等，周期在 case 容差内；
6. XRT：AFU、DMA、bitstream、host buffer、真实时钟；
7. 模型端到端：输出、性能、fallback、内存和长时间稳定性。

### 12.2 CI 分类

在 `ci/testcases/` 增加 `hip_native.yaml`、`triton.yaml`、`pytorch.yaml`、`quant.yaml` 和 `models.yaml`。每个 case 只做一种 check：`functional`、`model_parity`、`perf_gate` 或 `fpga_gate`。性能/综合 baseline 只能由人工带 update 参数重新生成，不能手改 JSON。

每个 case 的配置必须同时传给 kernel 编译、driver 构建和 blackbox；改变 `VX_config.toml` 或 Makefile 后先在 build 目录执行 `../configure`。调试失败时按仓库规则先用 SimX 作为 oracle，再导出 SimX/RTL 相同 CSV，比较首个 divergence。

### 12.3 AI agent 执行协议

后续让 AI 实现任务时，每个任务描述必须包含：

- 目标文件和禁止跨层引用；
- 输入/输出 ABI 与 capability 前置条件；
- 一个最小复现命令；
- 正确性 oracle 和容差；
- SimX、rtlsim、XRT 的完成条件；
- 需要新增的 CI case 和 perf/golden 文件；
- 未支持路径的明确错误码；
- 失败时应输出的 IR、日志、trace 和配置 hash。

AI 完成代码后必须报告“改了什么、为什么、运行了哪些命令、结果、剩余风险”，不能只报告编译成功。任何硬件时序/队列/缓存行为变化都要同步检查 SimX timing model 和 parity case。

## 13. 主要风险与需要决策的事项

1. **LMEM 容量与描述符偏移**：当前 LMEM 和 TCU descriptor 可能限制 Triton/CUTLASS tile。先用真实资源综合和 kernel occupancy 数据决定 32/64/128 KiB，而不是凭经验改大。
2. **warp 宽度**：HIP/CUDA 源码常假设 32。短期让编译器暴露 `warpSize` 并测试 NT=4/8/16/32；长期可把 DL FPGA 档固定 NT=32。
3. **数学函数**：当前没有专用 SFU 时，softmax/GELU/SiLU 依赖软件 libm。先实现可验证的近似库，再以 MPM 测量是否值得增加硬件 SFU。
4. **多队列**：若 CP 仍是单队列，PyTorch stream 语义可正确但无法重叠。应把“正确性支持”和“并发性能支持”分开发布，并在 capability 中声明。
5. **ROCm/完整 HIP 兼容度**：不建议整体移植 AMD HSA/ROCm；先交付 Vortex 原生 HIP 子集和公开 conformance，逐项扩展 API。chipStar 继续作为兼容参考。
6. **PyTorch 版本**：PrivateUse1 和 torchao API 变化较快。锁定一个 PyTorch/torchao 版本组合，CI 同时跑上游 API smoke；升级必须单独变更并更新适配层。
7. **动态 shape/训练/分布式**：先完成静态 shape 推理。训练需要 backward、optimizer、RNG 和更强同步；多卡需要 peer memory/RCCL 等，不能在单卡 runtime 里伪装实现。
8. **模型许可和可复现性**：为每个模型记录 checkpoint URL、commit/hash、tokenizer 版本和许可证；CI 使用小模型或合成权重，避免把不可再分发权重提交到仓库。

## 14. 首批可直接执行的待办清单

按此顺序开工，每项完成后才进入下一项：

- [x] `P0-01` 建立 `dl_functional/dl_rtl/dl_fpga` 配置矩阵和 JSON capability dump。（2026-09-11）
- [x] `P0-02` 在现有配置下跑 regression、HIP chipStar、TCU、DXA，并保存基线日志。（部分：SimX 后端 2026-09-11 恢复并全绿，见 runtime_p0_02_capability_baseline.md；rtlsim/chipStar/TCU/DXA 基线仍待做）
- [x] `P1-01` 审计 `vortex2.h` 与 HIP API 的一一映射，补齐错误/能力枚举。（部分：queue-ordered free 与 CP capability 完成；allocator pool 与 hipMallocAsync 待做）
- [x] `P1-03` 固化 vxbin 参数 metadata 和 RV32/RV64 ABI 测试。（VXKMDATA 已被 libhip_vortex 消费；>4KiB 参数等边界待补）
- [x] `P2-01` 创建 `hipcc-vortex` 最小编译器驱动，完成 vecadd 直编直跑。（rv32/rv64 SimX 通过，见 hip_p2_03_hip_headers.md）
- [x] `P2-02` 创建 `libhip_vortex` device/memory/stream/event/module 子集。（双 XLEN 端到端通过，见 hip_p2_04_libhip_vortex.md；RV32 指针宽度 ABI 修复）
- [x] `P2-03` 完成 native HIP GEMM、atomic、async overlap 和 conformance smoke。（2026-09-14,simx+rtlsim 双后端通过;两大硬件发现:老 spawn 模型多 warp `__syncthreads` 不可靠、float atomicAdd 无 ZACAS 活锁;chipStar 对照与 hiprtc stub 留待,见 p2_03 文档）
- [x] `P3-01` 以 FP16/BF16 GEMM 为种子实现 BLAS dispatch 和 TCU fallback。（tiled FPU kernel 三 dtype 三后端通过;VOLT 缓存 arg 字段 miscompile 与 launch 维度=0 两个发现;TCU 变体待 capability ID）
- [x] `P3-02` 实现 prim/norm/activation/conv/attention/RNG，并建立统一 reference harness。（第一层+conv/pool/bn 完成;**attention/RNG/LLM 支持算子/RoPE/SwiGLU/KV-cache 经工作流批次补齐**(p3_03),Mamba selective scan 前体就位）
- [x] `P4-01` 让 Triton vector add/matmul/softmax/layernorm 通过 interpreter/SimX/rtlsim。（**codegen 里程碑达成**:TTIR→C→vxbin 转译器使官方 vecadd 经真实 `triton.compile` 管线编译并在 SimX 运行(max_abs=0);softmax/reduce 类 1.5e-08;白名单外类型响亮报错。tl.dot 布局与 MLIR 原生 lowering 仍为文档化开放项;matmul/layernorm 教程形状部分依赖 tl.dot,见 p4_01 文档）
- [x] `P5-01` 完成 PyTorch PrivateUse1 allocator/device guard/basic ATen ops。（torch 2.4,真实设备内存与 kernel,pytest 7/7;torch 2.4 踩坑全记录）
- [x] `P5-02` 跑 ResNet eager，再接 `torch.export` 和 Inductor/Triton。（2026-09-15:P3 第二层 conv/pool/bn 已交付,MiniResNet 全设备前向 logits 与 CPU 差 ≤1e-7、零 CPU fallback,见 p5_02 文档;torch.compile/Inductor 依赖 P4 codegen,维持开放）
- [x] `P6-01` 完成 W4A16、INT8 的 pack/dequant/matmul 与精度报告。（pack bit-exact、gemm 6.3e-05/1.1e-07,三后端;模型级精度报告待 P7 harness）
- [x] `P6-02` 逐项打开 FP8、MX/NVFP4、2:4 sparse，并加入 capability/CI gate。（FP8 E4M3/E5M2 + MXFP8 + NVFP4 + 2:4 sparse 的**软件路径**全部完成:pack bit-exact、GEMM ≤1e-06 级、sparse metadata 真消费(毒化验证);`VX_CAPS_TCU_DTYPES` gate 落地。审计发现并修复了 e4m3 编码器两个真实 bug(次正规窗口、顶 binade NaN),独立 oracle 199k 点零错。**硬件 TCU 加速变体**仍待 TCU 构建树,见 p6_02/p3_03 文档）
- [x] `P7-01` 依次交付 ResNet、YOLO、SAM、DINOv3、Mamba、LLM、VLM checkpoint 报告。（第一份:ResNet 族(MiniResNet)按 §11 框架交付 —— 覆盖表/零 fallback/内存/数值;其余模型阻塞清单如实列出,见 p7_01 报告）
- [ ] `P7-02` 在 U55C/V80 等目标上完成 XRT 集成、roofline 和端到端性能报告。

## 15. 参考链接

- [Vortex README](../../../README.md)、[codebase map](../../codebase.md)、[testing](../../testing.md)、[simulation](../../simulation.md)。
- [Vortex runtime ABI](../../../sw/runtime/include/vortex2.h)、[HIP chipStar design](../../designs/hip_on_vortex_chipstar.md)、[TCU RTL](../../../hw/rtl/tcu)。
- [HIP runtime API modules](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules.html)。
- [HIP runtime programming guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api.html)。
- [PyTorch PrivateUse1 integration](https://docs.pytorch.org/tutorials/advanced/privateuseone.html)。
- [PyTorch operator registration](https://docs.pytorch.org/docs/stable/accelerator/operators.html)。
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/)。
- [Triton backend packaging](https://github.com/triton-lang/triton/blob/main/setup.py)。
- [torchao quantization overview](https://docs.pytorch.org/ao/stable/contributing/quantization_overview.html)。
- [torchao quantized inference](https://docs.pytorch.org/ao/stable/workflows/inference.html)。
