# P5.13 W3.2：gather 与 scatter

实施日期：2026-09-22。承接 [p5_12_cat_stack.md](p5_12_cat_stack.md)。本轮
关闭 W3.2 的基础索引搬运缺口，基线为 `build_dl64`、XLEN=64、simx、Python
3.10.21 / PyTorch 2.14.0。

## 1. 实现

注册了 `aten::gather` 和 `aten::scatter.src`。两条路径都要求连续的 FP32
数据张量，索引为连续的 int32 或 int64 Vortex tensor，输入 rank 不超过 4。
维度支持负索引；除索引维之外，index 的各维不能超过 self 的对应维。
`gather` 的 `sparse_grad=True`、`scatter` 的 scalar/reduce/in-place/out
overload 仍明确拒绝。

新增 `index_args_t` 参数块和两个 kernel。每个线程把连续输出的线性位置还原为
多维坐标，在索引维读取 int32/int64 index，再用 self 的连续 stride 计算地址：

- `gather` 从 self 读取并写到 index 形状的输出；
- `scatter.src` 先用设备到设备拷贝复制 self，再把 src 的线性元素写到索引位置。

索引值的检查在设备 kernel 内完成。越界值只设置一个设备端错误标志，不执行
非法读写；host 在当前队列上同步读取 4 字节标志并报告明确的 `out-of-range`
错误。这样不需要把整个 index tensor 搬回 CPU，代价是该基础实现每次调用都要
完成一次小的校验同步。

空 index 不发射 kernel：gather 返回空输出，scatter 返回 self 的设备副本。
scatter 对重复索引沿用并行写入语义，当前只将无重复索引作为确定性验收范围。

## 2. 验收用例

新增 `torch-vortex/tests/test_index_ops.py`，覆盖：

- gather 在首维、中间维、负维上的 CPU 对比；
- int32/int64 索引和小于 self 的非索引维 shape；
- scatter.src 的 FP32 结果、int32 索引和一次索引 kernel 提交；
- 空 index 的结果和零 kernel 行为；
- 越界索引、非连续 index、错误 dtype、src/index shape 不匹配、
  `sparse_grad=True` 的拒绝路径。

## 3. 结果

```text
build_dl64: ../configure --xlen=64 --tooldir=/data/vortex-tools -> PASS
build_dl64: make -s -C torch-vortex/kernels                  -> PASS
build_dl64: pytest torch-vortex/tests/test_index_ops.py      -> 8 passed
```

测试使用项目记录的正式组合（Python 3.10.21、PyTorch 2.14.0、simx），扩展
manifest 判定为 `supported`。首次测试发现 scatter 参数块必须分别使用 self
的 stride/维度边界和 src 的数据指针；修正后 int32、int64、空 index 与错误
路径全部通过。

## 4. 当前边界与后续

- 当前只覆盖 `gather` 基础 schema 和 `scatter.src`；`gather.out`、
  `scatter_`、scalar/reduce overload 未实现。
- 不支持 sparse gradient、dtype promotion、非连续数据或 rank 大于 4。
- `index_add` 仍需处理重复索引的累加和原子语义；`topk`、`interpolate`、
  stride-aware elementwise/归约和 RNG 仍在后续节点。

## 5. 工作记录

| 步骤 | 操作 | 结果 |
|---|---|---|
| 1 | 审计 PyTorch 2.14 的 `gather` 与 `scatter.src` C++ 签名 | 确认 gather 的 sparse_grad 参数和 scatter 的 src overload |
| 2 | 增加统一 index 参数块及 gather/scatter kernel | 4 维内连续布局坐标映射完成 |
| 3 | 增加 int32/int64 索引、shape、layout 和 uint32 边界检查 | 非法 schema 在提交前拒绝 |
| 4 | 加入设备端越界标志和小范围同步校验 | 不回读整个 index，错误信息稳定 |
| 5 | 添加 CPU 对比、空 index、launch 计数和错误测试 | 8/8 通过 |
| 6 | 更新 README、覆盖矩阵与路线图 | gather/scatter 移入基础支持，index_add 保留为下一项 |
