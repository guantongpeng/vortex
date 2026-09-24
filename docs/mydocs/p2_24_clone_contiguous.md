# 阶段 2.3：clone 与 contiguous 语义

## 实现

`aten::clone` 和 `aten::contiguous` 现在在 Vortex PrivateUse1 后端有显式实现。实现通过
`empty_like` 分配目标 storage，再复用设备端 `copy_` 路径完成复制，因此
不会把数据拉回 CPU，也不会让结果与输入共享 storage。默认的
`MemoryFormat::Preserve` 保留非连续、无重叠 view 的 stride；调用方显式传入
`torch.contiguous_format` 时由 `empty_like` 生成连续目标。

## 验证

`tests/test_clone.py` 覆盖：

- clone 的独立 storage 和写入隔离；
- 转置 view 的默认 stride 保留；
- 显式 contiguous memory format。
- contiguous 对已有连续 tensor 返回原 alias，对转置 view 生成独立连续 storage。

在 RV64、SimX、PyTorch 2.14、Python 3.10 的 `build_dl64` 中，3 项测试通过。

## 边界

本节点只覆盖 clone/contiguous 的 FP32 eager 路径。cast、dtype promotion、重叠
stride 和训练 autograd 仍按总计划保留为后续工作；没有为这些路径增加 CPU
fallback。
