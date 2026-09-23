# 阶段 2.2：二元逐元素统一到 prim

## 目标

`torch-vortex` 原先在 `torch_all.vxbin` 中维护了三套二元逐元素 kernel：
`binary_op_kernel` 处理两个同形状张量，`scalar_op_kernel` 处理张量和标量，
`broadcast_op_kernel` 处理带 stride 的广播。这样做使一元、归约、GEMM 等
已经迁移到 `sw/dl` 后，二元算子仍有独立的 ABI、镜像和生命周期，容易出现
队列、参数布局和模块加载行为不一致。

本节点将三种入口放入 `sw/dl` 的 prim 镜像，并让 PyTorch bridge 的
`add/sub/mul/div/minimum/maximum`、标量 overload、in-place overload 和
broadcast 路径都通过 `vx_prim_*` 提交。

## ABI 和实现

`sw/dl/src/prim_args.h` 是 host/device 共用的唯一参数定义：

| kernel | 参数 | RV64 大小 | 作用 |
| --- | --- | ---: | --- |
| `prim_binary_kernel` | `dst,a,b,n,op` | 32 B | 同形状连续 FP32 |
| `prim_scalar_kernel` | `dst,a,value,n,op,reverse` | 40 B | 标量在左或右 |
| `prim_broadcast_kernel` | 三个地址、维度、三组 shape/stride | 88 B | 最多四维、输出连续 |

二元操作码保持 `add/sub/mul/div/maximum/minimum` 的既有顺序，host 头文件和
device 头文件在 `prim_host.cpp` 通过 `static_assert` 锁定数值一致性。广播
stride 以 FP32 元素为单位，零 stride 表示沿该维重复同一个元素；PyTorch
bridge 仍负责用 ATen 的右对齐规则验证 shape，并在提交前把逻辑 stride 压入
32 位参数。

三种 kernel 都使用单一 grid-stride 循环，block 为 4 个线程；同一地址作为
输入和输出时，每个线程只读写自己的元素，因此支持现有 in-place schema。
`maximum/minimum` 显式传播 NaN，除法沿用 IEEE-754 的零除行为。

## bridge 迁移

`launch_binary_op`、`launch_scalar_op` 和 `launch_broadcast_op` 不再访问
`torch_all.vxbin` 的 function handle，而是调用 `vx_prim_binary`、
`vx_prim_scalar`、`vx_prim_broadcast`。这些调用经过 `DL_LAUNCH`，因此会
进入统一的设备工作计数和队列顺序路径。旧的三个 kernel、参数结构和镜像
metadata 已删除；`test_ops_elementwise.py` 检查旧入口不再出现在 extension
的参数表中，防止今后误把重复实现加回来。

## 边界

本节点仍保持 PyTorch bridge 的正式支持边界：输入为 Vortex 上的连续 FP32
张量，最多四维广播；scalar 是 PyTorch 包装的 CPU 0-d tensor，其他 dtype、
非连续输入和超过四维广播在 launch 前拒绝。dtype promotion、整数/布尔二元
运算和重叠 in-place view 不由本节点扩展，后续按阶段 2.1/2.3 的支持矩阵单独
处理。

## 验证

在 `build_dl64`（RV64、SimX、PyTorch 2.14、Python 3.10）重新执行
`configure`，强制重建 `prim.vxbin` 与 `torch_all.vxbin` 后：

```text
test_elementwise.py + test_ops_elementwise.py: 65 passed
test_reductions.py: 27 passed
```

既有的 NaN、带符号零、alpha、反向 scalar、in-place 版本计数和多维广播
用例均继续通过；新增的镜像检查确认三个旧二元入口已经从 torch 镜像移除。
