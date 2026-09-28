# 阶段 2.3：设备端 `sort`/`topk` 基础路径

## 实现

`torch_kernel_args.h` 新增共享的 `sort_args_t`，`ops.hip` 用一个 row-wise
insertion kernel 同时服务完整 `sort` 和 `topk`。每个 CTA 处理一个独立行，按
升序或降序把输入值和 int64 索引插入输出；top-k 只保留前 k 项，因此不需要
host-side value/index buffer 或 CPU 排序。

bridge 将任意 dim 移到尾维并 contiguous，提交后再把 values/indices view 移回
调用方的维度。`topk` 的 `largest`、`sorted`、负 dim、k=0 和 k 越界均在同一
路径处理；`sorted=False` 仍返回有序结果，这符合 API 对无序结果“不保证顺序”
的允许范围。rows、cols、k 使用 uint32 资源检查，零行或 k=0 只记 skipped
launch，不发射零 grid。

## 验证

`torch-vortex/tests/test_sort_topk.py` 对比随机/固定 FP32 输入的最后维、非最后维、
升降序、largest=True/False 以及 k=0。RV64/PyTorch 2.14/Python 3.10 主线结果为
**2 passed**。

## 边界

当前只支持 contiguous 计算后的 FP32 输入和基础 `aten::sort`/`aten::topk` schema；
stable、dimname、out overload、half/bfloat/int dtype、NaN 特殊排序规则以及大行的
并行排序优化仍待。selection kernel 的复杂度为 O(rows × cols × k)，用于推理小张量
和语义收口，性能优化另列节点。
