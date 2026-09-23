# 阶段 2.4：average pooling divisor override

`vx_dnn_pool2d_ex` 增加 divisor 字段，`avg_pool2d(...,
divisor_override=d)` 直接让 device kernel 使用 d 作为分母；`count_include_pad`
的既有两种语义仍由 op 字段选择。默认 `vx_dnn_pool2d` 保持原 ABI 入口并以
divisor=0 调用扩展入口，避免已有 DL 客户端重新编译才能运行。

bridge 在分配输出和提交前检查 divisor 为正且可放入 uint32，max pool 搭配该参数
会直接拒绝。`test_bounds.py` 保留 ceil_mode 的明确拒绝，并新增 divisor_override
对拍；RV64/PyTorch 2.14/Python 3.10 主线 `test_bounds.py` 为 **17 passed**。

ceil_mode、pool dilation、return_indices 和更高维 adaptive pooling 仍未实现，
不在本节点的支持声明内。
