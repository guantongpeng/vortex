# 阶段 2.4：average pooling divisor override

`vx_dnn_pool2d_ex` 增加 divisor 字段，`avg_pool2d(...,
divisor_override=d)` 直接让 device kernel 使用 d 作为分母；`count_include_pad`
的既有两种语义仍由 op 字段选择。默认 `vx_dnn_pool2d` 保持原 ABI 入口并以
divisor=0 调用扩展入口，避免已有 DL 客户端重新编译才能运行。

bridge 在分配输出和提交前检查 divisor 为正且可放入 uint32，max pool 搭配该参数
会直接拒绝。max/avg 的 `ceil_mode` 复用同一窗口 kernel：host 计算 ceil 输出尺寸，
kernel 对边界窗口按有效元素计数。`test_bounds.py` 新增 ceil_mode 与 divisor_override
对拍；RV64/PyTorch 2.14/Python 3.10 主线 `test_bounds.py` 为 **18 passed**。

为避免 ceil 输出宽度与 DL kernel 的行跨度分离，新增
`vx_dnn_pool2d_ex_mode` ABI 入口，显式传递 ceil 选择并由 host 计算后的输出
尺寸写入同一参数块；旧 `vx_dnn_pool2d_ex` 保持 floor 语义兼容。

pool dilation、return_indices 和更高维 adaptive pooling 仍未实现，
不在本节点的支持声明内。
