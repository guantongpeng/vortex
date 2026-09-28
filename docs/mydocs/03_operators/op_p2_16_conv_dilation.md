# 阶段 2.4：DNN convolution dilation

## 实现

在现有 grouped/depthwise DNN kernel 上增加 `dh`、`dw` 两个 ABI 字段。设备
地址计算使用 `oy * stride + kh * dh - padding` 和对应的宽度公式；权重布局、
分组通道映射、单组 LMEM staging 和 bias 路径保持不变。RV64/RV32 metadata
分别更新为 104/84 字节，并由共享 `dnn_args.h` 生成 host/device 一致的结构。

为了避免破坏已有直接 DL 调用，`vx_dnn_conv2d` 保留为 `dh=dw=1` 的兼容 wrapper，
新路径使用 `vx_dnn_conv2d_dilated`。PyTorch bridge 计算有效 kernel
`(k-1)*d+1` 后再做输出 shape、溢出、padding 和 LMEM 校验，校验通过才分配和
提交 kernel；因此非法 dilation 不会产生设备 launch。

## 验证

`test_bounds.py` 新增 1×3×9×10、3×2 filter、`dilation=(2,1)`、非方形 padding
的 CPU 对拍。RV64/PyTorch 2.14/Python 3.10 主线结果为 **17 passed**，覆盖已有
分组、depthwise、空输入、LMEM 和 pool 边界；旧的 direct `vx_dnn_conv2d` 调用
继续走 wrapper。

## 边界

transposed convolution、pool dilation、超出 uint32 的有效 kernel 和超过单滤波器
16 KiB LMEM 的大通道/大 kernel 仍明确拒绝。权重 tiling 是后续性能节点。
