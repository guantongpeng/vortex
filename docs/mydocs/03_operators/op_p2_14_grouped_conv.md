# 阶段 2.4：grouped/depthwise convolution

## ABI 和 kernel

`vx_dnn_conv_args_t` 新增 `groups` 字段，RV64 参数块从 88 B 变为 96 B，
RV32 为 76 B；DNN metadata、host struct 和 device kernel 同步更新。kernel
按 `ci_group = CI / groups` 计算每个 output channel 所属的 group：权重地址
仍遵循 PyTorch 的 `[CO][CI/groups][KH][KW]`，输入地址跳到对应 group 的
channel 区间，LMEM 只 stage 该 group 的权重。

ATen bridge 在分配输出前验证 `groups > 0`、`CI % groups == 0`、`CO % groups
== 0` 和 weight 的第二维等于 `CI/groups`；LMEM 限制也按每组输入通道计算。
因此 `groups=2` 和 `groups=CI` 的 depthwise 形式走同一真实 DNN kernel，
没有 host 分组循环或 CPU fallback。

## 验证

`test_bounds.py` 为 **16 passed**，新增 groups=2 与 depthwise 对拍；此前
`test_dl_bridge.py` 的 direct `vx_dnn_conv2d`/ATen kernel identity 也在新 ABI
下通过。原有 groups=1、stride、padding、空输出和 LMEM 边界保持回归覆盖。

## 当前边界

dilation 仍要求 1，transposed convolution、非连续输入和超过 LMEM 的单组
filter 仍明确拒绝；大 kernel 的权重 tiling 是后续性能节点。
