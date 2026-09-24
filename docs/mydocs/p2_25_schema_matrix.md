# 阶段 2.3：eager schema 支持矩阵

这份矩阵记录当前 RV64/SimX、PyTorch 2.14、Python 3.10 主线中真实设备
执行过的 schema。矩阵中的“拒绝”指 host bridge 在分配或 launch 前
`TORCH_CHECK`，不是 dispatcher 把请求转到 CPU。

| schema 家族 | overload | dtype | layout/shape | 结果 |
| --- | --- | --- | --- | --- |
| `copy_`, `_copy_from`, `to` | 非阻塞参数保持同步语义 | FP32/FP64/FP16/BF16/int32/int64/bool | 连续、非连续、storage offset、broadcast、D2D/H2D/D2H | 支持；重叠 copy 和 pinned 异步仍拒绝或保留边界 |
| `clone`, `contiguous` | memory format preserve/contiguous | 支持 copy dtype 集合 | 连续与非连续 view | 支持；独立 storage，连续输入 `contiguous()` 返回 alias |
| `add/sub/mul/div` | tensor、scalar、in-place、alpha | FP32 | ≤4D、右对齐 broadcast、stride | 支持；形状不可 broadcast 或超过 4D 在 launch 前拒绝 |
| unary/relu/GELU/SiLU | out-of-place、主要 in-place | FP32 | 连续和 elementwise broadcast 路径 | 支持；其他 dtype 在 bridge 拒绝 |
| `sum/mean/amax/amin/min/argmin/argmax` | dim、keepdim、out（已覆盖项） | FP32，indices=int64 | reduction dim 可移动到尾部；kernel 输入连续化 | 支持；空 reduction、非 FP32 和部分 dtype override 拒绝 |
| `gather/scatter/index_add` | base、out、in-place | data FP32，index int64/int32 依 schema | contiguous index；设备 data | 支持；非法 index/layout 在 launch 前拒绝 |
| `cat/stack` | base、out | FP32 | 连续输入，维度和 shape 经 ATen 校验 | 支持；不兼容 shape 直接拒绝 |
| `sort/topk` | base、values out | values FP32，indices=int64 | 任意 dim，k=0，基础稳定性未承诺 | 部分支持；stable、整数/低精度、dimname 和大行并行仍待 |
| `arange` | end/start/start_step | int32/int64/FP32 | 正负 step、空范围 | 支持；double/half/BF16/bool/complex/out 未实现 |
| `nll_loss_forward` | none/sum/mean | input/weight FP32，target=int64 | 1–2D target，ignore_index | 支持；训练/autograd 不在范围 |
| `group_norm` | optional weight/bias 独立省略 | FP32 | NCHW 及更多连续空间 rank，空 batch | 支持；非连续和其他 dtype 拒绝 |
| convolution | groups/depthwise/dilation | FP32 | NCHW、权重 staging ≤16 KiB | 支持；大通道/大 kernel tiling 仍待 |
| pool | max/avg、divisor_override、ceil_mode、max dilation | FP32 | NCHW 2D window | 支持；return_indices 和更高维 adaptive 仍待 |
| dropout | `dropout`, `native_dropout` inference | FP32 | `train=False` identity | 支持；`train=True` 在 RNG 接入前明确拒绝 |

## 明确未纳入矩阵的 schema

`interpolate`、stride-aware reduction、完整 cast/dtype promotion、训练随机
算子、autograd、checkpoint、Transformer/LLM 专用 op 和 Inductor codegen
仍由总计划单独跟踪。它们没有通过设备 kernel 验收前，不在支持声明中。

## 验证入口

矩阵对应的回归入口位于 `torch-vortex/tests/`：`test_copy.py`、
`test_clone.py`、`test_ops_elementwise.py`、`test_reductions.py`、
`test_index_ops.py`、`test_concat.py`、`test_sort_topk.py`、`test_arange.py`、
`test_nll.py`、`test_group_norm.py`、`test_bounds.py` 和 `test_dropout.py`。
任何新增 overload 必须先加入对应行的 dtype/layout/shape 证据，再更新计划
状态；unsupported 路径必须保留可诊断错误。
