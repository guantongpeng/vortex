# 阶段 2.1：空张量和资源边界

## 已收口的规则

PyTorch bridge 现在在进入 runtime/DL launch 前明确处理以下边界：

- 结果为空的 conv、pool、elementwise、zero-row reduction 和 zero-batch
  bmm 返回正确 shape 并跳过 launch；计数器可以证明没有提交零 grid；
- GEMM/mm/bmm 的 `K=0` 返回零乘积（或 addmm 的 beta epilogue），不把零
  contraction 交给 BLAS；
- reduction 的空输入区分有 identity 的 sum/mean 和没有 identity 的
  max/min/argmax/argmin，后者报错并且不提交 kernel；
- conv/pool 在计算输出 shape 前检查 kernel、stride、padding、dilation 和
  ceil/divisor overload，避免负数转成 uint32 后形成超大 grid；
- 所有传给参数块的 shape、stride、元素数通过 `u32_dim/u32_numel` 检查，
  超过 uint32 在 host 侧失败；conv LMEM、block 上限和 metadata 资源限制由
  runtime 提交前拒绝。

不支持的非连续算子输入仍以明确的 contiguous 错误拒绝；strided copy/view
走阶段 2.1 的 metadata/copy 路径，不会静默改变布局。

## 证据

`test_bounds.py` 的 15 个边界用例和 `test_matmul.py` 的 26 个 GEMM/空形状
用例在 RV64 SimX 通过（本轮合计 **41 passed**）。覆盖 kernel 大于输入、
零 stride、负 padding、LMEM 超限、空 conv/pool、空 reduction、K=0、空 batch、
非连续 bmm 和 beta=0 不读取 NaN self。

## 边界声明

本节点解决的是已支持算子的 shape/资源提交契约，不等同于支持任意 dtype、
任意 layout 或动态超大 tensor；这些仍必须经过对应 operator schema 的支持矩阵
和参数 ABI 检查。
