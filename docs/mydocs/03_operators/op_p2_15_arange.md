# 阶段 2.3：设备端 `arange`

## 目标

位置索引、时间步和 mask 构造经常直接调用 `torch.arange(...,
device="vortex")`。此前该 factory 没有 PrivateUse1 实现，调用要么落到未实现路径，
要么只能在 CPU 生成后再搬运。这个节点把常用的整数和 FP32 范围生成放进
`torch-vortex` kernel image，生成结果全程留在设备内存。

## 实现

`arange_args_t` 与 device kernel 共用同一参数定义，包含输出地址、元素数、dtype、
浮点起点/步长和整数起点/步长。kernel 使用 grid-stride loop 写入三种格式：

- 整数边界默认 `int64`，显式 `dtype=torch.int32` 使用 32 位写入；
- 浮点边界默认 `float32`，显式 `dtype=torch.float32` 使用同一路径；
- 计算元素数只在 host 上进行边界和溢出检查，元素值由设备 kernel 计算，不生成
  host 数值数组。

bridge 注册 `aten::arange`、`aten::arange.start` 和
`aten::arange.start_step`。它检查 strided layout、Vortex device 0、非 pinned
memory，支持正负步长、空范围和 `uint32` 元素上限；空范围分配合法的零元素
Tensor，但跳过零 grid launch。`float64`、bool、half/bfloat16、复数和整型 dtype
配合小数步长在 launch 前明确拒绝。

## 验证

`torch-vortex/tests/test_arange.py` 覆盖默认 int64、显式 int32/float32、负步长、
空范围的 skipped-launch 计数、步长方向不一致时的空结果、非法 dtype、整型起点的
float32 输出和整型 dtype 的小数步长。RV64/PyTorch 2.14/Python 3.10 主线结果为
**17 passed**（含 ABI）；kernel metadata 由共享参数头重新生成，
并由既有 ABI 测试继续校验。

## 边界

当前不承诺 double、half、bfloat16、bool 或复数输出，也不提供 `out=` overload。
浮点起点和步长按 float32 kernel 语义写入，超出 `uint32` 元素数或非有限输入会在
提交前报错。
