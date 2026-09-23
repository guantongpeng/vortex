# 阶段 2.4：`relu_` 原地语义

`aten::relu_` 已通过统一的 `vx_prim_unary` in-place 路径执行，输入和输出地址
相同，每个设备线程只读写自己的元素。非原地 `relu` 继续分配独立输出，因此共享
输入的另一条分支不会被修改。ATen 的 TensorImpl 负责 version counter，现有测试
验证 `data_ptr`、输入保留、NaN/负零以及 `_version` 增长。

该节点没有新增 kernel 或 ABI；`test_ops_elementwise.py` 和
`test_elementwise.py` 已覆盖完整 unary/in-place 回归。其余 pool 形态和训练反向
仍未开放。
