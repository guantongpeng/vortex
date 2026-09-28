# 阶段 2.3：`amin` 的独立归约入口

## 问题和取舍

`sum/mean/amax/argmax` 共用一个按 `op` 分支的 16-thread row reduction
kernel。直接把 MIN 加入同一 kernel，需要按 op 选择 `+INF` seed 和 MIN
shuffle；在 VOLT `-O3` 下这会让现有 kernel 的 LMEM store 被错误消除，结果
退化为 seed，并同时破坏已经通过的 sum/max/mean/argmax。这个问题不能用扩大
测试容差或 host 端修补掩盖。

本节点保留原归约 kernel 的代码和操作集合，新增静态的
`prim_min_kernel`。它只实现 FP32 行最小值，seed 固定为 `+INFINITY`，通过
`nan_min` 和 `warp_min` 合并 16-thread CTA 的局部结果。这样编译器不需要在
同一 kernel 中推断多种 seed/归约路径，原有归约 ABI 和结果保持不变。

## ABI 和 host 路径

MIN 与其他 row reduction 共用 `vx_prim_reduce_args_t`（RV64 40 B、RV32
28 B），但 metadata 有独立的 `prim_min_kernel` 记录，包含 256 B LMEM 和
`[16,1,1]` block 上限。`vx_prim_reduce` 对 `VX_PRIM_OP_MIN` 选择新 kernel，
其余 op 仍选择旧 kernel；module init/finalize 同时管理两个 handle。

PyTorch bridge 注册 `aten::amin` 和 `amin.out`，沿用已有的维度归一化、移动
到末维和 contiguous staging。NaN 传播、相等值的符号零和 `+/-INF` 均由
device reduction 决定；空 reduction 在 layout 检查阶段明确拒绝，不发射
kernel。空张量构造本身属于正常的输入分配，所以测试对该拒绝允许一次分配，
但仍要求没有 launch 或数据传输。

## 验证

在 `build_dl64` 重新运行 `configure`，删除旧 `prim.vxbin` 后重建，使新的
metadata 和镜像同时更新。`test_reductions.py` 共 **30 passed**，覆盖：

- 全量和按维度 `amin`，包括 keepdim 使用的通用 shape 路径；
- 含 NaN、`+INF`、负数的行；
- 空 reduction 的提交前拒绝及计数器不发射约束；
- 原有 sum、mean、amax、argmax、signed-zero 和非法维度回归。

## 当前边界

本节点只实现 FP32 `amin`；`min(dim=...)` 的值和 index、`argmin`、`topk`、
`sort` 和其他 dtype 仍是后续算子节点。独立 kernel 是针对 VOLT 编译器问题
的根因修复，不能把共享 kernel 的操作分支重新合并回去。
