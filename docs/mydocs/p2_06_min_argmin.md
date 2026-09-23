# 阶段 2.3：`min`、`min(dim)` 和 `argmin`

## 实现范围

在 `amin` 的独立值归约之后，本节点补齐了同一 FP32 支持域的 index 变体：

- `prim_min_index_kernel` 使用固定的 `+INF` seed，和 `prim_min_kernel` 分离，
  避免把 MIN/ARGMIN 分支重新放回会触发 VOLT 误编译的共享归约 kernel；
- `vx_prim_index_reduce` 接受 `VX_PRIM_OP_ARGMIN`，输出每行 `uint32` index，
  可选地在同一遍写回 value；host bridge 再用现有 widen kernel 转为 ATen
  要求的 int64；
- PyTorch 注册 `argmin`、`min`（全量）和 `min.dim`，返回值/索引形状、
  keepdim、负维度和 0-D 输入沿用已验证的 `max`/`argmax` 路径。

## 数值规则

`argmin_better` 与 `argmax_better` 对称：NaN 优先、多个 NaN 取最早 index、
数值相等取最早 index。比较使用 `<`，因此 `+0/-0` 的 tie 保留第一个元素，
返回的 value 与返回的 index 来自同一候选，避免 value/index 不配对。

空 reduction 仍在 layout 检查阶段拒绝；没有为 `min` 或 `argmin` 伪造空
identity。所有提交通过调用方 queue，并受既有资源和参数大小校验保护。

## 验证

`build_dl64` 中重建 `prim.vxbin` 后，`torch-vortex/tests/test_argmax.py`
为 **41 passed**。测试覆盖 2D/3D/4D 的每个维度、负维度、keepdim、全量
0-D 结果、重复最小值、signed zero、NaN 及既有 max/argmax 回归。

## 尚未覆盖

`topk`、`sort`、`nll_loss_forward`，以及非 FP32 dtype 的 min/index reduction
仍保持明确拒绝，不能从这两个入口推断出 dtype promotion 或排序支持。
