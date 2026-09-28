# 阶段 2.3：`sort`/`topk` out overload

在已有 device row-wise sort kernel 上补齐 `aten::sort.values` 和
`aten::topk.values`。bridge 先完成计算，再检查 values/indices 位于同一 Vortex
device、dtype 分别为输入 dtype/int64、shape 与结果一致，最后通过 device
`copy_` 写入调用方提供的 storage；返回值保持对 out Tensor 的引用，符合 ATen
alias contract。

`test_sort_topk.py` 新增 sort/topk out 对拍和 data_ptr 检查，基础路径与 out
overload 共 **3 passed**。stable、dimname、低精度/整数 dtype 及大行并行排序仍是
后续边界。
