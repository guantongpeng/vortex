# Vortex 深度学习文档

这里集中放置 Vortex 深度学习软件栈的计划、设计、阶段记录和验证报告。文件按阅读主题分组。阶段文档统一采用 `<track>_p<阶段>_<序号>_<主题>.md`，例如 `runtime_p1_01_async_free.md` 和 `op_p2_02_gemm_transb.md`；总览文档使用 `overview_` 前缀。主题前缀用于区分并行工作流，阶段号保留原有里程碑含义，序号在同一工作流和阶段内唯一。非阶段计划使用 `<track>_plan.md`，例如 `torch_plan.md`。

## 推荐阅读顺序

1. [统一实施计划](00_overview/overview_plan.md)：当前唯一的执行状态和任务依赖入口。
2. [Vortex 软件系统详解](00_overview/overview_software_system.md)：理解硬件、工具链、运行时、模拟器和 CI 的整体结构。
3. [运行时与 ABI](01_runtime_abi/)：配置矩阵、能力查询、队列、模块元数据和 launch 合同。
4. [原生 HIP](02_hip/)：HIP runtime、HIPVortex 工具链、kernel 和 SimX/rtlsim 验证。
5. [算子库](03_operators/)：BLAS、prim、DNN 和 PyTorch 算子语义修复记录。
6. [Triton](04_triton/) 与 [PyTorch](05_pytorch/)：上层编程模型和后端接入。
7. [量化](06_quantization/) 与 [模型报告](07_models/)：低精度路径和端到端模型证据。

## 目录

| 目录 | 内容 | 维护入口 |
|---|---|---|
| [`00_overview/`](00_overview/) | 总体架构、统一计划、历史路线图和目标软件栈 | [`overview_plan.md`](00_overview/overview_plan.md) |
| [`01_runtime_abi/`](01_runtime_abi/) | DL 配置、capability、queue、CP、VXKMDATA、launch 合同 | [`runtime_p1_03_module_metadata.md`](01_runtime_abi/runtime_p1_03_module_metadata.md) |
| [`02_hip/`](02_hip/) | 原生 HIP API、工具链、kernel 和 parity | [`hip_p0_03_native_ci.md`](02_hip/hip_p0_03_native_ci.md) |
| [`03_operators/`](03_operators/) | BLAS、prim、DNN、attention 和 PyTorch 算子语义 | [`op_p2_25_schema_matrix.md`](03_operators/op_p2_25_schema_matrix.md) |
| [`04_triton/`](04_triton/) | Triton Vortex backend | [`triton_p4_01_vortex_backend.md`](04_triton/triton_p4_01_vortex_backend.md) |
| [`05_pytorch/`](05_pytorch/) | PrivateUse1 后端和 eager 算子覆盖阶段记录 | [`torch_plan.md`](05_pytorch/torch_plan.md) |
| [`06_quantization/`](06_quantization/) | W4A16、W8A8 和 FP8 | [`quant_p6_01_quantization.md`](06_quantization/quant_p6_01_quantization.md) |
| [`07_models/`](07_models/) | ResNet 和模型级验证报告 | [`model_p7_01_reports.md`](07_models/model_p7_01_reports.md) |

## 文档约定

- `overview_plan.md` 的状态以代码和可复现验证记录为准；其他计划文档保留设计背景或历史快照。
- 阶段开发记录说明实现、边界和验证证据，不替代 `docs/` 下的架构规范或 `AGENTS.md` 的工程规则。
- 修改源码、Makefile、配置或 `torch-vortex` 后，按仓库规则从配置好的 build 目录重新运行 `configure`，再复现文档中的测试。
