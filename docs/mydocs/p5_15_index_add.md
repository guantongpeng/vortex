# P5.15 W3.2：index_add

实施日期：2026-09-22。承接 [p5_14_avg_pool2d.md](p5_14_avg_pool2d.md)。本轮
关闭 W3.2 的索引累加基础缺口，基线为 `build_dl64`、XLEN=64、simx、Python
3.10.21 / PyTorch 2.14.0。

## 1. 实现

注册 `aten::index_add`。输入要求是连续 FP32 self/source，连续 int32 或 int64
的一维 index，rank 不超过 4；source 除累加维外与 self 形状相同，累加维长度
等于 index 长度，dim 支持负索引，alpha 支持标量缩放。

硬件基础配置没有可依赖的 float atomicAdd（ZACAS 未启用，CAS fallback 在高
竞争时不能作为正确性基础），因此没有把重复 index 留给不确定的并行写入。新增
`index_add_kernel` 为每个输出元素执行确定性 index 扫描：匹配当前行的 index
逐个累加 source，再写出 `self + alpha * sum`。这保留了重复 index 的 PyTorch
语义和固定的求和顺序，代价是 O(output_elements × index_length)，适合作为
正确性路径，后续可在有 float atomic 或分桶归约后替换。

设备端错误标志沿用 gather/scatter 路径，负数或超出 self 累加维的 index 会在
写结果前标记并由 host 报告；空 index 不发射 kernel，只复制 self。

## 2. 验收用例

新增 `torch-vortex/tests/test_index_add.py`，覆盖：

- 重复 int64 index、alpha=0.5 与 CPU `torch.index_add` 对比；
- int32 index、负 dim 和默认 alpha；
- 空 index 的 self 复制和零 launch；
- 越界 index、非一维 index、source shape 不匹配、错误 dtype 的拒绝路径。

## 3. 结果

```text
build_dl64: ../configure --xlen=64 --tooldir=/data/vortex-tools -> PASS
build_dl64: make -s -C torch-vortex/kernels                -> PASS
build_dl64: pytest torch-vortex/tests/test_index_add.py    -> 4 passed
```

测试使用项目记录的正式组合（Python 3.10.21、PyTorch 2.14.0、simx），扩展
manifest 判定为 `supported`。

## 4. 当前边界与后续

- 仅实现非 in-place、非 out 的 `index_add`；`index_add_`、`index_add.out` 和
  其他 dtype 尚未接入。
- 当前 kernel 是确定性扫描，重复 index 正确但不适合大 index 或高吞吐模型；
  性能节点应在 float atomic/ZACAS 或分段归约可用后单独验收。
- topk/sort、interpolate、stride-aware elementwise/归约和 RNG 仍是后续缺口。

## 5. 工作记录

| 步骤 | 操作 | 结果 |
|---|---|---|
| 1 | 审计 A 扩展与 float atomic 能力 | 确认不能依赖 float atomic，选择确定性扫描 |
| 2 | 增加 `index_add_args_t` 与扫描 kernel | 支持重复 index 和 alpha 缩放 |
| 3 | 注册 `aten::index_add`，加入 rank/shape/dtype/范围校验 | 错误在非法写入前报告 |
| 4 | 添加重复 index、空 index、int32 和错误路径测试 | 4/4 通过 |
| 5 | 更新 README、W3 覆盖矩阵与路线图 | index_add 移入基础支持，topk 保留为下一项 |
