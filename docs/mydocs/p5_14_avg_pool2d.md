# P5.14 W3.3：avg_pool2d 基础路径

实施日期：2026-09-22。承接 [p5_13_gather_scatter.md](p5_13_gather_scatter.md)。
本轮把已有 DL window-pool kernel 的平均模式接到 `aten::avg_pool2d`，基线为
`build_dl64`、XLEN=64、simx、Python 3.10.21 / PyTorch 2.14.0。

## 1. 实现

扩展 `vx_dnn_pool2d` 的 `op` 枚举：

- `0`：max pool；
- `1`：平均池化，分母只统计窗口内元素（`count_include_pad=False`）；
- `2`：平均池化，分母为完整 kernel 面积（`count_include_pad=True`）。

kernel 在 op=2 时仍只累加有效输入，最后除以 `kh * kw`，因此 padding 区域等价
于零；op=1 保留原来的 in-bounds count 语义。DL host 校验也从两个 op 扩展为
只接受 0/1/2。

PyTorch 侧注册 `aten::avg_pool2d`，复用现有 `pool_impl` 的 NCHW、FP32、连续
布局和窗口尺寸检查。`count_include_pad` 两种取值都支持；`ceil_mode=True`
和 `divisor_override` 暂未实现，调用在分配和发射前明确拒绝。

## 2. 验收用例

- `test_bounds.py::test_avg_pool2d_matches_cpu` 用带 padding 的输入分别验证两种
  `count_include_pad` 语义；
- `test_bounds.py::test_avg_pool2d_rejects_unimplemented_overloads` 验证 ceil 和
  divisor override 的错误路径；
- `test_dl_bridge.py::test_dl_pool_count_include_pad_true` 直接调用 DL ABI，确保
  ATen 路径与共享 kernel 使用相同的 op=2 实现。

## 3. 结果

```text
build_dl64: ../configure --xlen=64 --tooldir=/data/vortex-tools -> PASS
build_dl64: make -s -C sw/dl                              -> PASS
build_dl64: make -s -C torch-vortex/kernels                -> PASS
test_bounds.py -k 'avg_pool or pool'                       -> 8 passed
test_dl_bridge.py -k 'pool'                                -> 4 passed
```

## 4. 当前边界与后续

- 仅支持连续 FP32 NCHW；ceil mode、`divisor_override`、更高维 pooling 和
  interpolate 仍未实现。
- max pool、avg pool、adaptive_avg_pool2d(1) 共享 `vx_dnn_pool2d`，不会形成
  第二份 PyTorch kernel。
- topk、stride-aware elementwise/归约和 RNG 仍是后续节点。

## 5. 工作记录

| 步骤 | 操作 | 结果 |
|---|---|---|
| 1 | 审计 `vx_dnn_pool2d` 的 op=1 和 ATen avg_pool schema | 确认缺口只在 count_include_pad 的分母语义与注册 |
| 2 | 增加 DL op=2 并扩展 host 参数校验 | 默认 avg_pool2d 语义可由同一 kernel 计算 |
| 3 | 注册 `avg_pool2d`，保留 ceil/divisor 的显式拒绝 | shape/window 校验仍集中在 ATen 层 |
| 4 | 增加 CPU 对拍和直接 DL ABI 测试 | 两种分母语义与 op=2 均通过 |
| 5 | 更新 README、W3 覆盖矩阵和剩余项 | avg_pool2d 基础路径移入支持，interpolate 保留 |
