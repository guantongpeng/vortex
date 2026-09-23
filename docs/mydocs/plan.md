# Vortex 深度学习全栈统一实施计划

> 本文是 `docs/mydocs` 的唯一执行计划，整合了 `pytorch_plan.md`、
> `vortex_dl_fullstack_implementation_plan.md`、`vortex_dl_fullstack_roadmap.md`
> 以及 P0–P7 开发记录。状态依据代码和后续开发记录更新，而不是机械沿用旧计划中的复选框。
>
> 更新时间：2026-09-22  
> 代码基线：`d57c50d11`  
> 当前正式验证主线：RV64、PyTorch 2.14、Python 3.10、SimX；其他组合必须在结果中单独注明。

## 状态和执行规则

- `[x]` 已实现并有对应测试证据。
- `[~]` 已实现子集，但阶段目标仍缺后端、配置、边界或模型级验收。
- `[ ]` 尚未完成，或只有局部原型，不能对外宣称已支持。
- 所有任务按本文件的阶段顺序执行；改变 RTL 时必须同步修改 SimX 并通过 model parity。
- 所有测试从配置后的 build 目录执行，修改配置、Makefile、`torch-vortex` 或生成文件后先重新运行 `configure`。
- 不能用 CPU fallback、FX 重放、SimX 宿主机耗时或手工脚本替代真实设备、XRT 或编译路径的验收。

## 一、当前状态总览

### 已完成或已具备可复用基础

- [x] P0.1：`dl_functional`、`dl_rtl`、`dl_fpga` 配置矩阵和 capability JSON。
- [x] P1.1：queue-ordered free、跨队列等待和异步释放基础语义。
- [x] P1.2：CP capability 查询和 runtime contract 基础实现。
- [~] P1.3：VXKMDATA、RV32/RV64 参数 ABI 和基础 module load/launch 测试；sidecar、超大参数和提交前全量校验仍待。
- [~] P2.1：native HIPVortex 探针、`hipcc-vortex`、HIP headers、vecadd 直编直跑；三尖括号、shared、half/bfloat 和完整 conformance 仍待。
- [~] P2.2：`libhip_vortex` 的 device、memory、stream、event、module 子集；完整 HIP 扩展和 rtlsim/parity 仍待。
- [~] P2.3：native HIP reduction、FP32 GEMM、整数 atomic、stream overlap 的 SimX/部分 rtlsim 验证；A 扩展 rtlsim、chipStar 和精确 parity 仍待。
- [~] P3.1：FP32/FP16/BF16 FPU GEMM、BLAS dispatch 和统一 reference harness 基础；TCU 变体和完整 batched/epilogue 仍待。
- [~] P3.2：prim、norm、activation、conv、pool、BN、attention、RoPE、SwiGLU、KV append、Philox 等软件 kernel；PyTorch 桥接和模型级验收仍待。
- [~] P4.1：Triton TTIR→C→vxbin 子集，vecadd、softmax/reduce 已通过 SimX；正式 MLIR lowering、`tl.dot` 和标准 launcher 仍待。
- [~] P5.1/P5.2：PrivateUse1、allocator、device guard、基础 eager ATen、MiniResNet 全设备前向；完整 eager schema、编译路径仍待。
- [~] P5.3/P5.4：W1 语义修复、W2 生命周期、stream/event、设备模块、退出和基础多镜像共存；query、caching allocator、pinned 和 XRT 仍待。
- [~] P5.5–P5.15：一元/归约、norm、softmax、argmax、bmm、cat/stack、gather/scatter、avg_pool2d、index_add 的基础路径；高级 overload、dtype、layout 和缺失算子仍待。
- [~] P6.1/P6.2 软件侧：W4A16、INT8、FP8、MXFP8、NVFP4、2:4 sparse 的编码、通用 kernel 和部分 capability gate；PyTorch/torchao、TCU 和模型报告仍待。

### 尚未形成完整交付的里程碑

- [ ] P0.2 完整基线：rtlsim、chipStar、TCU、DXA 和 FPGA 配置仍缺系统性记录。
- [ ] M2 单设备推理：RNG、checkpoint、完整 ResNet-18、tiny Transformer 和更完整算子语义仍缺。
- [ ] M3 编译子集：标准 Triton launcher、FakeTensor、FX/export、真实 Inductor codegen 未完成。
- [ ] M4 FPGA 交付：XRT、roofline、端到端硬件性能和可安装 SDK 未完成。
- [ ] M5 训练生态：Autograd、优化器、AMP 训练、编译训练和多设备通信未完成。

### 已有验证证据

主线在 `build_dl64`、SimX、Python 3.10、PyTorch 2.14 的阶段性结果包括：P5.3 的 W0/W1 全量测试 86 passed，P5.4 的 W2 全量测试 117 passed，P5.6 的统一 prim 测试 238 passed，P5.9 的对抗性复审 288 passed；P5.11–P5.15 还分别覆盖 bmm 25、cat/stack 11、gather/scatter 8、pool 12、index_add 4 个用例。P5.10 的 28 passed 使用的是临时 PyTorch 2.4/Python 3.8 环境，正式支持表未改变，不能当作正式组合的完整验收。

## 二、按顺序执行的工作包

## 阶段 0：可复现基线和持续验证（M0，最高优先级）

### 0.1 环境和版本锁定

- [~] 记录正式组合的 Python、PyTorch、Triton、C++ ABI、VOLT、`TOOLCHAIN_REV`、驱动、XLEN、CONFIGS、build 目录和源码 commit（RV64/SimX 主线）；torchvision 仍需单独安装并验证。
- [x] 选择并锁定正式支持的 Python/PyTorch/Triton 组合；未验证组合已在支持表中拒绝或标为实验环境。
- [ ] 为 TCU 变体和所有 RV32/rtlsim/FPGA 组合建立独立 build 目录并纳入同一 manifest。
- [ ] 为每次 CI/后端测试自动保存环境 manifest、配置快照、镜像 hash、退出码和复现命令（当前仅主线测试具备）。

### 0.2 干净 out-of-tree 构建

- [x] 修复 `configure`、Python 包、C++/HIP 源码和 torch kernel 的主要 build-tree 依赖。
- [x] 修复 `tests/dl/*` 等仍使用相对 `main.cpp` 的路径问题，消除手工拷贝才能成功的条件（历史提交 `f94e3a309` 已完成）。
- [ ] 将 cache/output 放入对应 build 目录，并把配置、ABI、工具链和源码变化纳入失效规则。
- [ ] 从空 build 目录仅执行 `configure/build/test` 即可运行所有基础测试，源码树不产生构建物（torch 主线已验证，DL 全集仍待）。

### 0.3 测试分层、观测和 parity

- [~] 将现有测试按 runtime、HIP、DL 算子、PyTorch eager、Triton 编译、模型、硬件集成分层，并为主要 pytest tier 设置超时；native HIP 功能矩阵已收口，DL/硬件全集仍待。
- [~] 已补齐并校准 `ci/testcases/hip_native.yaml` 的 native HIP 功能入口和后端标签；SimX、rtlsim、XRT 的统一通过条件及 DL parity case 仍待。
- [x] native HIP 功能 case 已按真实证据登记：默认算子覆盖 SimX/rtlsim，A-extension atomic 保留 SimX，`bigargs` 验证 4 KiB 参数上限负向契约；详见 [`p0_03_hip_native_ci.md`](p0_03_hip_native_ci.md)。
- [x] `hip_native:parity_module_api` 已接入 `check: model_parity`，比较 RV32 SimX/rtlsim 的 retired instructions 和周期；完整 native HIP parity 矩阵及 DL parity 仍待，详见 [`p0_04_native_hip_parity.md`](p0_04_native_hip_parity.md)。
- [x] 记录基础 launch、H2D/D2H/D2D、同步、分配和 host 数值计算计数；完整 ATen op→kernel→queue/event、活跃/峰值内存关联仍待 W7.1。
- [ ] 接入 profiler/trace，区分 Python/dispatcher、JIT、提交、DMA 和设备执行时间。
- [ ] 完成 P0.2 中 rtlsim、chipStar、TCU、DXA 的基线；不能把 SimX 通过记为全部后端通过。

**阶段验收：**干净环境可复现；同一 workload 的 SimX/rtlsim retired instruction 一致，周期误差在既有 CI 容差内；失败日志包含配置、镜像、kernel、参数大小和首个错误元素。

## 阶段 1：Runtime、ABI 和 HIP 合同收口（M1 收尾）

### 1.1 参数 ABI 和 module metadata

- [x] host/device 共用参数结构，基础 `sizeof`、RV32/RV64 对齐和 VXKMDATA 已落地。
- [ ] 将所有 `args_size`、`alignof`、`offsetof`、ISA、TCU/DXA、block/LMEM 校验改为单一来源。
- [x] 对超过 `VX_KERNEL_ARGS_MAX_BYTES`（4 KiB）的参数块返回明确错误，不允许静默截断。
- [x] 在 `vx_enqueue_launch`/批量命令构建前校验已发布的 `args_size` 和参数块边界。
- [~] 在提交前继续补齐 metadata、XLEN、ISA、配置和 block/LMEM 的全量校验；当前已拒绝 metadata `max_block`、`static_lmem_bytes`、`required_isa` 不匹配及零维度/零 block，XLEN/配置/features 仍待。
- [x] runtime 已将 metadata 资源约束统一应用于单次和批量 launch，并补充提交前失败、无 event 的回归；详见 [`p1_05_metadata_resource_validation.md`](p1_05_metadata_resource_validation.md)。
- [x] DL BN kernel 的 `max_block` 元数据已与 host 固定的 16-thread block 对齐；缺失资源字段会在 runtime 提交前按保守默认拒绝，详见 [`p1_07_dnn_metadata_resource_contract.md`](p1_07_dnn_metadata_resource_contract.md)。
- [~] HIP `hipFuncGetAttribute` 已映射 max threads、静态/动态 shared memory 和寄存器字段；Triton autotune、PyTorch loader 以及其余 HIP attributes 仍待，详见 [`p1_06_hip_function_attributes.md`](p1_06_hip_function_attributes.md)。
- [ ] 补齐 RV32 metadata 全量漂移测试；现有测试不能只解析 `ARGS_SIZE` 一列。

### 1.2 HIP runtime 未覆盖接口

- [x] device、memory、stream、event、module、kernel launch 子集。
- [ ] `hipMallocAsync` 和 memory pool；定义普通 `hipFree` 与 stream-ordered free 的区别。
- [ ] 补 `vx_kernel_set_arg`、`vx_event_set_callback`、host pointer/export 查询等 runtime API 缺口。
- [ ] graphs、texture/surface、peer/multi-device、hiprtc、完整 `hipLaunchKernelGGL` 逐项决定支持范围并实现或明确拒绝。
- [ ] 补 HIP half/bfloat 类型、三尖括号语法、静态 `__shared__` 和 KMU 多 warp barrier 的正式支持边界。
- [ ] 明确 `hipMallocManaged`/统一内存、cooperative launch/cluster、`hipFuncGetAttributes` 和 occupancy API 的支持范围。
- [ ] 补 `hipMemcpy2D/3D`、`hipHostRegister`、pointer attributes 等内存 API，或为不支持路径提供可诊断错误。
- [ ] 补 `hipPointerGetAttributes` 所需 host-mapped/export 查询。
- [ ] 完成 module unload/reload、refcount、混合 pinned-host copy 和多镜像压力测试。
- [ ] 完成 chipStar 同源对照、A-extension rtlsim、ZACAS float atomicAdd 和 SimX↔rtlsim retired-instruction parity。

### 1.3 allocator、stream 和退出语义

- [x] 延迟释放、跨流依赖、thread-local device/stream、事件 record/wait、设备级同步和异常退出基础路径。
- [ ] 实现 caching allocator、allocator 统计、OOM/异常清理、RAII scratch 管理和 pinned memory。
- [ ] 实现或明确拒绝 `queryStream/queryEvent`，补 stream destroy hook、`hipMemGetInfo` 和多线程行为测试。
- [ ] 验证初始化失败回滚、fork 后行为、未完成任务销毁顺序和无设备/错版本库诊断。

### 1.4 多模块 loader

- [x] 基于 image slot 的基础多镜像共存和地址冲突拒绝。
- [ ] 完成链接地址分配与可重定位镜像方案评估，覆盖绝对引用、入口 PC、符号、rodata/bss、缓存和卸载。
- [ ] 同进程交替加载 torch、DL 和多个 Triton JIT 镜像；SimX 与 XRT 都验证地址可复用和引用安全。
- [ ] 若需要工具链修改，按组件→prebuilt→Vortex 顺序发布。

**阶段验收：**提前释放不产生 use-after-free；错误 XLEN、损坏 metadata、超限参数和不兼容镜像在提交前失败；多模块交替 launch 后进程稳定退出。

## 阶段 2：PyTorch eager 推理覆盖（M2）

### 2.1 Tensor 语义和边界

- [x] 非原地 ReLU、addmm/mm/linear 基础语义、BN batch 修复、max pool 特殊值、基础 view/copy 和 PrivateUse1 factory 隔离。
- [x] `copy/to` 已实现 FP32/FP16/BF16/int32/int64/bool 的设备端 dtype conversion、broadcast copy、非连续源/目标、storage offset 和 D2D/H2D/D2H 路径；重叠 copy、pinned/non-blocking 的完整语义仍列为边界，详见 [`p2_01_tensor_copy_dtype.md`](p2_01_tensor_copy_dtype.md)。
- [x] 完整实现 view/reshape/transpose/permute/slice/select/as_strided 的 alias、stride、版本计数和必要复制；详见 [`p2_07_view_alias.md`](p2_07_view_alias.md)。
- [x] 支持或明确拒绝空张量、K=0、零 launch、越界 shape、非法 stride/padding、LMEM/block/grid 超限；详见 [`p2_08_shape_bounds.md`](p2_08_shape_bounds.md)。
- [x] 保持 CPU、Meta、其他设备 factory 不受影响；FakeTensor 阶段不分配设备内存、不发射 HIP；详见 [`p2_09_factory_fake.md`](p2_09_factory_fake.md)。

### 2.2 DL 与 torch-vortex 统一

- [x] conv、pool、BN、GEMM、norm、一元和归约的主要路径已迁移到 `sw/dl`。
- [x] 给 `sw/dl` prim 增加二元 elementwise 入口，迁移 `binary_op`、scalar 和 broadcast kernel；旧 torch 镜像入口已删除，详见 [`p2_04_prim_binary.md`](p2_04_prim_binary.md)。
- [ ] 为 DL library 使用调用方 device/context/queue，消除新的全局 singleton。
- [x] BLAS GEMM 增加 `transb`，linear 直接读取 `[N,K]` 权重，消除 `w.t().contiguous()` 临时拷贝；详见 [`p2_02_gemm_transb.md`](p2_02_gemm_transb.md)。
- [x] `bmm` 已改为单次 grid-z batch-stride/tiled GEMM，按真实 Tensor batch stride 传参；addmm/linear epilogue 保持独立，详见 [`p2_03_batched_gemm.md`](p2_03_batched_gemm.md)。
- [x] 二元 elementwise 操作码改用 `vortex/prim.h` 公共定义，并由 host/device static_assert 校验；mxfp8 ABI 漂移仍待，详见 [`p2_10_binary_enum_abi.md`](p2_10_binary_enum_abi.md)。
- [ ] 将重复状态枚举统一为可检查的公共定义，并把 `mxfp8_args.h` 纳入 ABI 漂移测试。

### 2.3 通用 Tensor 和基础算子

- [x] FP32 连续布局的 `sum/mean/max/argmax`、`bmm`、`cat/stack`、`gather/scatter.src`、`index_add`、`avg_pool2d` 基础路径。
- [x] `gather.out` 与 `scatter.src_out` 复用同一 device index kernel 和 out storage；其余 out/in-place overload 仍待，详见 [`p2_11_index_out.md`](p2_11_index_out.md)。
- [x] `scatter_.src`、`index_add.out/_`、`cat.out`、`stack.out` 已补齐并复用既有 device kernel；详见 [`p2_12_out_inplace_overloads.md`](p2_12_out_inplace_overloads.md)。
- [~] `topk`、`sort`：基础 FP32 values/indices、任意 dim 和 k=0 已落地；stable、dimname、out、低精度/整数 dtype 与大行并行优化仍待，详见 [`p2_17_sort_topk.md`](p2_17_sort_topk.md)。
- [x] FP32 `amin` 与 `amin.out`，采用独立静态 MIN reduction kernel；详见 [`p2_05_amin.md`](p2_05_amin.md)。
- [x] FP32 `min`、`min.dim`、`argmin` 及 value/index 配对；详见 [`p2_06_min_argmin.md`](p2_06_min_argmin.md)。
- [x] FP32 `nll_loss_forward` 的 none/sum/mean、weight、ignore_index 与越界校验；详见 [`p2_13_nll_loss.md`](p2_13_nll_loss.md)。
- [ ] `interpolate`、`ceil_mode`、`divisor_override`、`group_norm`。
- [ ] stride-aware elementwise/reduction，避免转置和非连续输入先拷贝成连续布局。
- [ ] cast、clone/contiguous、transpose/permute/slice/select/expand、dtype promotion 和 bool/int64 数据路径。
- [x] `arange` 的 int64/int32/float32 设备端生成、正负步长和空范围边界；double、half/bfloat16、bool、复数与 `out=` 仍明确拒绝，详见 [`p2_15_arange.md`](p2_15_arange.md)。
- [ ] 补 broadcast add/mul/sub/div、标量 alpha、GELU/SiLU、dropout，以及模型需要的 in-place/out overload。
- [ ] 补 `gather.out`、`scatter_`、reduce/scalar overload、`index_add_`/`index_add.out`、`cat.out`/`stack.out`。
- [ ] 对每个 schema 建立 overload×dtype×layout×shape 支持矩阵，unsupported 必须在 launch 前明确报错。

### 2.4 CNN eager 和模型分级

- [x] conv 支持 groups/depthwise、dilation，并按每组输入通道计算权重 staging 与 LMEM；大 kernel 和完整模型分级仍待，详见 [`p2_14_grouped_conv.md`](p2_14_grouped_conv.md)、[`p2_16_conv_dilation.md`](p2_16_conv_dilation.md)。
- [ ] conv 支持大通道和大 kernel 的权重 tiling，并检查 LMEM/资源上限。
- [x] BN inference optional affine 与 linear 前导 batch 维已支持；详见 [`p2_18_linear_batch.md`](p2_18_linear_batch.md)。
- [x] avg_pool2d `divisor_override` 已接入统一 DNN kernel；ceil_mode、pool dilation、return_indices 和其余 pool 形态仍待，详见 [`p2_19_pool_divisor.md`](p2_19_pool_divisor.md)。
- [x] `relu_` 已复用统一 prim unary in-place kernel，并保留 storage/version mutation 语义；剩余 pool 形态仍待补齐。
- [ ] 完成 MiniResNet→小输入 ResNet-18→目标输入尺寸的分级测试，固定 torchvision、权重和预处理。
- [ ] 输出中间层、logits、fallback、传输、内存和性能证据，不能只检查最终输出。

### 2.5 Transformer、LLM、RNG 和 checkpoint

- [ ] 将已有 Philox 接入 PyTorch Generator、seed/offset/state、`rand/randn` 和模型随机操作。
- [ ] 实现 `torch.vortex` seed/state/fork、跨调用/跨流推进和 checkpoint 后可复现。
- [ ] 接入 layernorm/rmsnorm、embedding、高级 gather、RoPE、GELU/SiLU/SwiGLU、attention 自定义 op 或 ATen schema。
- [ ] 验证 mask、causal、scale、head layout，再补 GQA、长序列和 paged KV。
- [ ] 实现 KV cache append/read、prefill、逐 token decode、采样/top-k，并统计禁止的 host 往返。
- [ ] 验证 `state_dict`、save/load、map_location、CPU↔Vortex 权重迁移和 storage 方法。
- [ ] Mamba 接入 selective scan/state update，解除小 `d_state` 限制并测试 chunked state continuation。

**阶段验收：**正式 PyTorch/Python 组合下，ResNet-18 分级、tiny Transformer/decoder、RNG 和 checkpoint 通过；稳态前向没有未记录的 host 数值计算或隐式 CPU fallback。

## 阶段 3：低精度、量化和 TCU（M2/M4 前置）

### 3.1 FP16/BF16 和 AMP

- [x] 独立 `sw/dl` kernel 已具备 FP16/BF16 GEMM 和相关 reference 验证。
- [ ] 在 PyTorch 路径完成 FP16/BF16 分配、copy/cast、基础运算、norm、GEMM、conv、序列化全路径。
- [ ] 明确舍入、NaN/subnormal、累加和输出 dtype，补 BF16 舍入及 `_Float16` 边界测试。
- [ ] 注册 Vortex autocast 策略，区分矩阵算子与敏感 reduction/norm，验证嵌套 autocast 和 cast cache。

### 3.2 TCU capability 和 kernel

- [x] `VX_CAPS_TCU_DTYPES` 查询和软件 gate 基础已落地。
- [ ] 建立完整 TCU 配置构建树和 capability ID，并接入真实硬件 kernel。
- [ ] 接入 GEMM/conv 的 TCU 变体，覆盖尾 tile、对齐、workspace、LMEM 和资源限制。
- [ ] 用反汇编/计数器证明实际使用 TCU；能力关闭或配置不匹配时返回明确错误或已声明的 FPU 路径。
- [ ] 每种 dtype 通过 SimX 数值、rtlsim 数值/周期、FPGA/ASIC 门禁三关。

### 3.3 PyTorch 量化接入

- [ ] 为 W4A16、INT8、FP8、MXFP8、NVFP4、2:4 sparse 建稳定自定义 op schema。
- [ ] 接入 Linear 权重打包、设备迁移、state_dict、fake/meta 和 torchao 适配评估。
- [ ] 分开记录格式编码、通用设备计算和硬件加速；验证 scale、zero-point、group size、饱和、尾部和 sparse metadata 消费。
- [ ] 为每种格式建立 bit-level golden、模型精度、内存和吞吐报告；无硬件支持时不宣称原生加速。

## 阶段 4：Triton、图捕获和 Inductor（M3）

### 4.1 Triton 标准 launcher

- [ ] 实现标准 `kernel[grid](...)` launcher、metadata、参数打包和真实 Vortex Tensor `data_ptr`。
- [ ] 对接 active device、properties、XLEN、warp size、current stream 和 benchmark synchronize。
- [ ] cache key 纳入工具链、后端代码、配置、XLEN、选项、ABI 和 dirty source。
- [ ] 在同一进程运行 Triton vecadd/softmax 并交给 ATen 消费，至少共存两个不同 JIT 镜像。

### 4.2 正式 Triton lowering

- [x] TTIR→C→vxbin 的 vecadd、softmax/reduce 原型和非法类型负例。
- [ ] 编写 Vortex `ttgir→LLVM IR` MLIR pass，明确 program-id/lane、blocked layout 和 `vx_spawn2` 映射。
- [ ] 设计并实现 layout、shared memory、barrier、reduction、FP16/BF16、二维 tile、`tl.dot`→FPU/TCU lowering。
- [ ] 将已支持子集加入 CI，并覆盖特殊 shape、资源上限、非法 dtype、broadcast、控制流和 dot 负例。
- [ ] 修复 VOLT `-O3` 参数缓存误编译根因；需要临时 workaround 时按仓库规则标记并建立后续任务。
- [ ] 完善 libdevice 的 exp/log/rsqrt 等数学路径；先验证软件近似，再用 MPM 决定是否需要 SFU。
- [ ] 接入 autotune 计数器和 MPM，禁止只用 wall-clock 选择配置。

### 4.3 FakeTensor、FX 和 export

- [ ] 修复 device module、factory 和 FakeTensor 的第一处真实失败；复用 ATen Meta/decomposition。
- [ ] 为自定义 op 补 fake/meta、alias/mutation 和有限动态 shape 规则。
- [ ] 增加明确命名的 FX backend，记录 graph break、图数量和设备执行证据。
- [ ] 验证 `torch.export.export`、save/load、shape constraints 和新进程中的 Vortex 执行；区分 FX replay 与真正 codegen。

### 4.4 Inductor

- [ ] 接入设备接口、scheduler、codegen、wrapper、runtime launch、DL kernel 调用和 stream 管理。
- [ ] 先实现 pointwise/reduction fusion，再接 Triton matmul、autotune 和 memory planning。
- [ ] 支持编译缓存、重编译条件、动态 shape 限制和失败诊断；禁止吞掉编译错误或静默 CPU 回退。
- [ ] 用真实 Vortex Tensor 验证 `torch.compile` 产生并执行目标代码，比较 eager/compiled 数值、graph break、编译时间和稳态时间。

## 阶段 5：模型级交付（M2/M4）

- [ ] ResNet-18：固定 torchvision 版本、权重、预处理和目标输入，完成中间层/精度/内存/性能报告。
- [ ] YOLO：补 upsample、动态 batch、检测头、NMS，并建立固定数据子集 mAP 门禁。
- [ ] SAM/DINOv3：补 ViT/window attention、grid_sample 等算子，建立 IoU/特征余弦门禁。
- [ ] Mamba：完成端到端 selective scan、分块状态延续和 tiny checkpoint 逐 token 对拍。
- [ ] Llama/Qwen：完成 embedding、GQA、RoPE、RMSNorm、paged KV、prefill/decode、sampling，并明确容量上限。
- [ ] VLM：复用视觉塔和 LLM 路径，完成 projector、图像 token 常驻设备和端到端指标。
- [ ] 每个模型记录 checkpoint URL/hash、许可证、输入尺寸、op 清单、未支持项、CPU reference、误差、fallback、内存和性能。

## 阶段 6：XRT、硬件性能和配置档（M4）

### 6.1 XRT 和性能报告

- [ ] 在 U55C/V80 等目标完成 AFU/CP/DMA/内存完整 XRT 路径。
- [ ] 按 copy/launch→单算子→小模型→目标模型推进，记录板卡、bitstream、时钟、内存容量和资源。
- [ ] 分离 warmup、JIT、稳态，报告 latency、throughput、峰值内存、TCU 利用率和内存带宽。
- [ ] 完成 roofline 和端到端性能报告；SimX 宿主机运行秒数不能作为 FPGA 性能。

### 6.2 硬件配置和性能前置

- [ ] 固化 NT=32、多核、L2/L3、A/TCU/DXA 的 DL 推荐配置并纳入 CI。
- [ ] 评估 LMEM 16 KiB→64/128 KiB，评估 WGMMA descriptor 偏移和 ABI 影响，并通过综合门禁。
- [ ] 评估多队列 CP/独立 DMA、SFU、device malloc、barrier 槽扩展；先用 MPM 数据证明瓶颈。
- [ ] 后续评估 HBM/interleave、寄存器文件、L2/L3/MSHR、FP16 packed datapath。
- [ ] 多卡、P2P、collective、抢占和 SV39/统一内存列为远期，不在单卡阶段伪造支持。

## 阶段 7：CI、安装交付和文档（与阶段 0 起持续）

- [ ] 增加 PyTorch/Triton CI 类别：PR 跑语义、ABI、资源负例和 MiniResNet；nightly 跑扩展 shape/dtype、编译和 RTL；专机跑 FPGA/性能。
- [ ] CI 检查测试确实执行，缺依赖、漏生成目录或错误退出不能被记为全绿；保留 manifest、数值差异和退出码。
- [ ] 增加 `pyproject`、预编译 C++ extension、kernel 资源、SDK/runtime 发现和 ABI/version 检查。
- [ ] 安装产物不依赖源码 checkout、个人路径或 import 时 C++ 编译器；明确 JIT 开发模式与发布包边界。
- [ ] 更新 README、支持矩阵、示例、报错和故障排查；删除过时的 fallback/API/版本描述。
- [ ] 在干净虚拟环境安装 wheel+SDK，运行 Tensor、CNN 和 Triton 编译示例。

## 阶段 8：训练和多设备生态（M5，推理稳定后）

- [ ] 审计 `AutogradPrivateUse1` fallthrough；按支持算子实现 backward、saved tensor、view/mutation 和版本检查。
- [ ] 覆盖 matmul/linear、activation、norm、loss 梯度，再扩 conv/attention；实现梯度累加和优化器算子。
- [ ] 用高精度参考、有限差分或方向导数验证梯度，完成小型 MLP 的多步 SGD/Adam。
- [ ] 实现 GradScaler、finite/unscale、溢出跳步、loss scale 和混合精度优化器路径。
- [ ] 在 eager 梯度正确后接 AOTAutograd/Inductor backward，验证 RNG、动态 shape 和激活生命周期。
- [ ] 有真实多设备硬件和通信需求后，再实现 peer copy、collective、ProcessGroup 和 DDP；定义实际支持范围。

## 三、依赖关系和建议执行顺序

```text
阶段0 基线/CI
  ├─ 阶段1 Runtime/ABI/loader
  │    └─ 阶段2 PyTorch eager
  │         ├─ 阶段3 低精度/TCU/量化
  │         ├─ 阶段4 Triton/Fake/FX/Inductor
  │         └─ 阶段5 模型
  ├─ 阶段6 XRT/硬件性能（可与阶段2–5 并行，但依赖稳定配置）
  └─ 阶段7 CI/安装（从阶段0开始持续）
阶段5 + 阶段7 → 阶段8 训练/多设备
```

建议的近期顺序：

1. 完成阶段 0.1–0.3，锁定正式环境并补全 parity/CI 证据。
2. 完成阶段 1.1–1.4，尤其是 metadata 单一来源、allocator 边界和 module loader 压力测试。
3. 完成阶段 2.3 的通用 Tensor 缺口、RNG/checkpoint，以及 ResNet-18/tiny Transformer。
4. 并行推进阶段 3.1–3.2 和阶段 4.1–4.3；先得到可验证的低精度和编译最小闭环。
5. 推进阶段 5 的目标模型，再做阶段 6 的 XRT/roofline/端到端性能。
6. 完成阶段 7 发布交付，最后进入阶段 8 训练和多设备。

## 四、统一验收矩阵

| 维度 | 最低要求 |
|---|---|
| Tensor 语义 | 标量、空张量、奇数/非 tile shape、broadcast、transpose、offset、stride、alias、mutation |
| 数值 | 固定 seed、NaN/Inf、极值、低精度边界；报告 max_abs、max_rel、必要的 ULP 和特殊值行为 |
| 异步和设备 | CPU↔Vortex、D2D、默认/非默认流、跨线程、提前释放、OOM、异常退出 |
| ABI/配置 | RV64；RV32 已支持的入口；XLEN/CONFIGS/LMEM/ISA 不匹配在运行前失败 |
| 框架共存 | CPU/Meta/factory 不被污染；FakeTensor 不 launch；多镜像可交替加载和卸载 |
| 图路径 | FX、export save/load、Triton JIT、Inductor 分别有真实设备证据 |
| 模型 | MiniResNet、ResNet 分级、tiny Transformer、prefill/decode；中间层和未支持项可定位 |
| 硬件 | SimX 功能、rtlsim parity、XRT 全集成、真实 FPGA 性能；明确区分模拟与实测 |

## 五、来源和状态追踪

- P0/P1：[`p0_dl_capability_baseline.md`](p0_dl_capability_baseline.md)、[`p1_async_free.md`](p1_async_free.md)、[`p1_cp_capabilities.md`](p1_cp_capabilities.md)、[`p1_module_metadata.md`](p1_module_metadata.md)。
- P2：[`p2_02_libhip_vortex.md`](p2_02_libhip_vortex.md)、[`p2_03_native_hip_kernels.md`](p2_03_native_hip_kernels.md)、[`p2_1c_hip_vortex_headers.md`](p2_1c_hip_vortex_headers.md)。
- P3/P4：[`p3_01_blas_dispatch.md`](p3_01_blas_dispatch.md)、[`p3_02_prim_kernels.md`](p3_02_prim_kernels.md)、[`p3_03_attn_llm_mamba_rng.md`](p3_03_attn_llm_mamba_rng.md)、[`p4_01_triton_vortex.md`](p4_01_triton_vortex.md)。
- P5：[`p5_03_m0_m1_baseline.md`](p5_03_m0_m1_baseline.md) 至 [`p5_15_index_add.md`](p5_15_index_add.md)。
- P6/P7：[`p6_01_quantization.md`](p6_01_quantization.md)、[`p6_02_fp8.md`](p6_02_fp8.md)、[`p7_01_model_reports.md`](p7_01_model_reports.md)。
- 被整合的旧计划：[`pytorch_plan.md`](pytorch_plan.md)、[`vortex_dl_fullstack_implementation_plan.md`](vortex_dl_fullstack_implementation_plan.md)、[`vortex_dl_fullstack_roadmap.md`](vortex_dl_fullstack_roadmap.md)。

旧文档保留作为设计背景和阶段证据；新增或变更任务只更新本文件，完成后同时补充对应测试记录和来源文档。
