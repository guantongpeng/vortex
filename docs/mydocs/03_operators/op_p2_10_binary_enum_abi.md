# 阶段 2.2：二元操作码公共定义

二元逐元素已经迁移到 `sw/dl` prim 后，bridge 仍在
`torch_kernel_args.h` 保留一份 `TORCH_BINARY_*` 枚举。它和
`vortex/prim.h` 的数值碰巧一致，但 host 代码实际把操作码传给 prim，任何
新增或重排都可能让 torch bridge 和 device image 静默选择不同操作。

本节点删除 torch header 的重复 enum，`vortex_ext.cpp` 直接使用
`VX_PRIM_BINARY_ADD/SUB/MUL/DIV/MAXIMUM/MINIMUM`。prim host 已有的 host/device
`static_assert` 继续锁定 public enum 与 `prim_args.h` 的 kernel enum；因此
操作码只有一个 public 定义和一个可编译检查的 device 对照。

这次只收口二元 elementwise。`mxfp8_args.h` 的其他状态/ABI 漂移测试仍未
完成，计划中的总项保持未勾选，避免把局部 enum 收口误报成所有 DL 状态已统一。

在 `build_dl64` 重跑 configure、torch kernel 构建后，elementwise 与 ABI 测试
为 **13 passed**，证明 bridge 编译、prim dispatch 和既有参数 metadata 均未
回归。
