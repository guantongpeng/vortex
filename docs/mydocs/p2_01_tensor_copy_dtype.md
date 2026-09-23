# P2.1：设备 Tensor copy/to 的 dtype 与 broadcast 语义

实施日期：2026-09-23。验证基线为 `build_dl64`、XLEN=64、SimX、Python
3.10.21、PyTorch 2.14.0。这个节点收口阶段 2.1 中最先阻塞模型权重迁移的
`copy_`/`to` 路径：目标和源不再必须是相同 dtype、相同 shape，也不再要求两边
都是连续布局。

## 实现

`torch-vortex/kernels/torch_kernel_args.h` 新增 `TorchCopyDType` 枚举，并把
源/目标 dtype 编码加入共享的 `copy_strided_args_t`。host、VOLT kernel 和
metadata 生成器使用同一结构，避免转换 kernel 另有一份 ABI。

`copy_strided_kernel` 现在按逻辑坐标读取源元素，经 FP32 中间值转换后写入
目标。支持的设备 copy 类型为：

| 类型 | 编码和行为 |
|---|---|
| FP32 | 原值传递 |
| FP16 | 使用 `vx_fp16_to_f32` / `vx_f32_to_fp16`，按 IEEE binary16 舍入 |
| BF16 | 使用 `vx_bf16_to_f32` / `vx_f32_to_bf16` |
| int32/int64 | 通过 FP32 转换，适用于模型权重和索引迁移范围 |
| bool | 非零写为 `true`，零写为 `false` |

host 源先以**源 dtype** staging 到设备，不能直接按目标 dtype 分配，否则整数
和浮点 bit pattern 会被误读。host 目标先接收一个连续设备 staging，再由 CPU
原生 copy 应用目标的 stride/storage offset；设备目标则直接由 kernel 写入。
同 dtype、同 shape、两边连续且 offset 为零时仍走 `hipMemcpy` 快路径。

copy 的 shape 校验按 PyTorch 的尾部维度广播规则执行。低 rank 源会在 kernel
参数中左侧补维，源 extent 为 1 的维度使用 stride 0；因此广播只读取一个源
元素，不会复制出一个临时 expanded Tensor。目标 stride 和 storage offset 则
始终按真实 Tensor metadata 传给设备 kernel。

## 验收

`torch-vortex/tests/test_copy.py` 新增：

- FP32 到 FP16、BF16、int32、int64、bool，以及各类型回到 FP32 的 round-trip；
- `(1, 3) -> (4, 3)` 的同 dtype broadcast copy；
- `(1, 2) FP32 -> (3, 2) FP16` 的 broadcast + conversion；
- 既有转置源、转置目标、D2D、空 Tensor、自 copy 和 shape 错误路径继续覆盖。

执行命令和结果：

```text
../configure --xlen=64 --tooldir=/data/vortex-tools                 PASS
CCACHE_DISABLE=1 make -s -C torch-vortex/kernels                    PASS
CCACHE_DISABLE=1 python -m pytest torch-vortex/tests/test_copy.py -q PASS (16 passed)
```

测试中的 `got.cpu()` 只用于把结果取回后与 CPU reference 比较；转换和广播本身
由 `copy_strided_kernel` 在设备上执行。`copy_` 的 H2D、D2H、D2D 字节统计仍按
真实搬运的源/目标 item size 记录。

## 边界

- complex、uint16/uint32/uint64、量化 dtype 尚未注册，遇到时在 launch 前报出
  `copy/to does not support dtype`。
- 两个不同 view 的重叠 self-copy 尚未实现 memmove 语义；只有同一 Tensor 的
  self-copy 明确是 no-op。重叠范围检测和临时缓冲属于后续节点。
- `non_blocking=True` 仍沿用 runtime 当前的同步 copy 合同，计数器会记录
  blocking sync；pinned host memory 和真正异步 H2D/D2H 尚未宣称支持。
- int64/FP32 之间的大整数转换受 FP32 中间值精度限制；它用于已声明的模型
  权重/索引迁移，不代表通用无损整数转换。

