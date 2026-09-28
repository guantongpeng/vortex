# 阶段 2.3：`nll_loss_forward`

## 实现

`torch-vortex` 新增 `nll_loss_kernel` 和 `aten::nll_loss_forward` bridge，
覆盖 FP32 log-prob 输入的 1D `[C]`/scalar target 与 2D `[N,C]`/`[N]`
target：

- `reduction=none/sum/mean`；1D `none` 返回 0-D，2D `none` 返回 N 个值；
- 可选 contiguous FP32 class weight，并同时写回 `total_weight`；
- `ignore_index` 不贡献 loss 或 weight；全忽略的 mean 返回 NaN，符合 ATen；
- device kernel 标记负数/越界 target，host 在一次提交后同步 flag 并报告明确
  `out-of-range` 错误，避免越界读取 input。

kernel 的参数块由 `torch_kernel_args.h` 统一定义并通过 metadata 生成，未在
host 侧计算 loss 或偷偷转 CPU。当前实现为单 CTA 顺序扫描，优先保证无 float
atomic 的确定性和基础模型语义；性能优化另列为后续工作。

## 验证

新增 `test_nll.py` 为 **3 passed**，覆盖三种 reduction、class weight、1D
输入、ignore_index、越界 target；ABI 与 reductions 回归另有 **35 passed**。

## 边界

只支持 FP32、1D/2D、int64 target；高维 NLL、label smoothing、训练 backward
和其他 dtype 暂不宣称支持，schema 会在 launch 前拒绝。
