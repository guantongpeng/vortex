# 阶段 2.3：索引、拼接和累加的 out/in-place overload

## 实现

在基础 device kernel 不变的前提下，bridge 补齐了以下 schema：

- `scatter_.src`：计算临时结果后写回原 tensor，保留版本计数和 alias；
- `index_add.out` 与 `index_add_`：复用重复 index 的确定性 scan，并保留
  `alpha`；
- `cat.out` 与 `stack.out`：复用原有拼接 kernel，把结果写到调用方提供的
  output storage；
- 与前一节点的 `gather.out`、`scatter.src_out` 一起，所有已实现的 index/
  concat 基础路径在 out 形式下都进行相同的 dtype、shape、rank、index range
  检查。

out wrapper 先完成 device 计算，再通过 Vortex `copy_` 写入 out，因此 out
可以与输入 alias 而不会在读取前被覆盖；shape 不匹配在 copy 前拒绝。in-place
形式返回原 tensor，并由 PyTorch version counter 记录 mutation。

## 验证

索引、累加和拼接测试合计 **29 passed**：覆盖 int32/int64 index、重复 index、
空 index、越界、alpha、负维度、out storage identity、in-place version bump、
空拼接片段和 CPU 数值对拍。

## 边界

`scatter` 的 value-scalar overload、跨 dtype promotion 和非连续 index 仍保持
明确拒绝；这些不由 out wrapper 自动获得支持。
