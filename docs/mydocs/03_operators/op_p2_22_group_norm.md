# 阶段 2.3：`group_norm`

## 实现

新增 `group_norm_args_t` 和 torch kernel image 中的 group normalization kernel。
每个 CTA 负责一个 `(batch, group)`，先在设备端计算 group 的均值和方差，再按
channel affine 写回同一 NCHW contiguous 输出。host bridge 支持 `[N,C,*]` 的任意
空间 rank（二维输入等价于每个 channel 一个元素），要求 `C % groups == 0`，并
允许 weight、bias 独立省略；`cudnn_enabled` 只作为 ATen schema 参数校验，不改变
设备路径。

输入、参数和输出都保持 Vortex FP32，空 batch 在提交前跳过 launch。参数块由共享
头生成 metadata，ABI 测试校验扩展加载时的 sizeof 一致性；没有 CPU 统计或 fallback。

## 验证

`test_group_norm.py` 覆盖 groups=1/2/4、无 affine、仅 weight、仅 bias、二维/三维/
四维/五维空间形状、非法 groups 和空 batch。RV64/PyTorch 2.14/Python 3.10 主线
`test_group_norm.py + test_abi.py` 为 **7 passed**。

## 边界

当前仅支持 contiguous FP32 inference；训练反向、低精度、动态 shape 编译和跨设备
参数仍未开放。
