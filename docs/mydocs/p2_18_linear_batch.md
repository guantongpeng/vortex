# 阶段 2.4：linear 前导 batch 维

## 实现

`aten::linear` 现在接受任意非零 rank 的 FP32 Vortex 输入。最后一维作为
features，其余维度乘积折叠为 GEMM 的 M；结果完成 `[... , in_features] ×
[out_features, in_features]^T` 后恢复为 `[... , out_features]`。1D 输入、
`[batch, sequence, hidden]` 和更高阶 batch 均复用现有 `transb` GEMM，不创建
权重转置临时张量。bias 仍在二维展平结果上执行同一设备端 epilogue。

## 验证

`test_matmul.py::test_linear_leading_batch_dimensions` 对 1D、3D 和 4D 输入与
CPU `functional.linear` 对拍；完整 matmul 回归为 **27 passed**。测试覆盖无 bias
和既有 transposed-B launch 计数，确认前导维扩展没有回退到 host 计算。

## 边界

输入和权重仍要求 contiguous FP32 Vortex Tensor；稀疏权重、量化 dtype、训练/反向
传播和动态 shape 编译路径仍未开放。
