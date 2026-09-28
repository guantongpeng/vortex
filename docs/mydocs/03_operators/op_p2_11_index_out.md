# 阶段 2.3：gather/scatter 的 `out` overload

## 实现

`gather.out` 和 `scatter.src_out` 现在注册到 PrivateUse1。两条路径先通过
已有的 device index kernel 完成边界检查和计算，再把结果写入调用方提供的
Vortex `out` tensor；返回对象就是同一个 `out` storage。输入 dtype、index
dtype/contiguity、rank/shape、越界 index 和 empty index 规则与非 out 版本完全
一致，避免 out overload 形成另一套未校验的 ABI。

`out` shape 在 copy 前比较，错误在 host 侧抛出；copy 本身走已经验证的
`copy_`/queue 顺序路径，因此不会把 CPU 或其他 device output 静默接入 Vortex
kernel。

## 验证

`test_index_ops.py` 现在为 **10 passed**，新增 gather/scatter out 结果、
storage identity 和 CPU 对拍，同时保留 int32 index、空 index、越界和非法
shape 回归。

## 边界

本节点覆盖 `gather.out` 与 `scatter.src_out`；`scatter_`、`index_add.out`、
`cat.out`/`stack.out` 仍待后续 overload 节点，计划总项继续保持未完成状态。
