# 阶段 2.1：view 家族的 alias 与 stride 语义

## 实现

Vortex Tensor 的 `view` 和 `as_strided` 直接调用 ATen 的原生 TensorImpl
构造路径：

- `view` 由 ATen 解析 `-1`、检查元素数量，并保留 storage/version counter；
- `as_strided` 保存 size、stride 和 storage offset，因而
  `transpose/permute/t/slice/select/expand` 等 metadata 操作都是同一 storage
  上的 view；
- `reshape` 沿用 ATen 的 view-or-copy 选择：可 view 时保留 alias，不能 view
  时产生 contiguous copy；
- `_copy_from`/`contiguous` 对 strided source 和 destination 使用已经完成的
  strided copy kernel，不把 storage 顺序误当作逻辑顺序。

代码没有手写第二套 TensorImpl，也没有把非连续输入静默复制给需要连续布局
的算子；算子本身仍会在 launch 前明确拒绝不支持的 stride。

## 验证重点

新增回归覆盖 slice 的 storage offset 和 stride、select 的 alias/version bump、
expand 的 zero stride，以及已有的 transpose/permute/t、非连续 reshape、
strided source/destination copy。所有结果都与 CPU 同输入对拍，且检查 data_ptr
和 shape/stride，而不只检查最终数值。

## 边界

本节点覆盖 metadata view 和必要的复制语义；autograd 训练、重叠 view 的
写入冲突、非 FP32 算子的 stride-aware device kernel 仍按各算子支持矩阵处理。
超过内部四维 strided copy 参数限制时，在提交前拒绝并给出维度错误。
