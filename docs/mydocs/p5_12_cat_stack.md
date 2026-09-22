# P5.12 W3.2：cat 与 stack

实施日期：2026-09-22。承接 [p5_11_bmm.md](p5_11_bmm.md)。本轮关闭
W3.2 数据拼接的基础缺口，基线为 `build_dl64`、XLEN=64、simx、Python
3.10.21 / PyTorch 2.14.0。

## 1. 实现

`aten::cat` 和 `aten::stack` 注册到 `cat_impl`、`stack_impl`。两条路径都只
接受连续的 FP32 Vortex tensor；输入列表必须非空，`cat` 要求除拼接维之外
形状相同，`stack` 要求所有输入形状相同。维度支持负索引，`stack` 还支持
在最后一维插入新轴。

host 侧先物化 PyTorch 2.14 的 `at::ITensorListRef`，完成 rank、shape、维度
和 uint32 边界检查，再分配连续输出。每个输入片段提交一次 `cat_kernel`，不
做 host 往返，也不依赖 CPU 拼接。

kernel 以连续输入的线性索引还原为 `(outer, source_dim, inner)` 坐标，再写到
输出的 `(outer, output_dim, inner)` 布局：

```text
inner_coord = i % inner
source_coord = (i / inner) % source_dim
outer_coord = i / (inner * source_dim)
dst = ((outer_coord * output_dim) + dst_offset + source_coord) * inner + inner_coord
```

`stack` 把 `source_dim` 固定为 1，并为第 `i` 个输入使用 `dst_offset=i`。
空输入片段不发射零网格，只增加 skipped-launch 统计；其余片段仍按正常顺序
写入同一个当前队列。

## 2. 验收用例

新增 `torch-vortex/tests/test_concat.py`，覆盖：

- `cat` 在首维、末维、负维和三维输入上的 CPU 对比；
- 拼接列表含空片段时的结果和 launch 数；
- `stack` 在每个轴及负维上的 CPU 对比；
- `stack` 每个输入一次 launch，以及空张量的 shape/数值语义；
- 形状不匹配、非连续输入和非 FP32 输入在 launch 前拒绝。

## 3. 结果

```text
build_dl64: ../configure --xlen=64 --tooldir=/data/vortex-tools -> PASS
build_dl64: make -s -C torch-vortex/kernels                  -> PASS
build_dl64: pytest torch-vortex/tests/test_concat.py         -> 11 passed
```

测试使用项目记录的正式组合（Python 3.10.21、PyTorch 2.14.0、simx），扩展
manifest 判定为 `supported`。首次编译暴露出 PyTorch 2.14 的 `ITensorListRef`
没有随机访问运算符，随后改为一次 `materialize()` 并通过
`reference_wrapper::get()` 访问；该兼容性修正和输出 uint32 边界检查已包含在
最终实现中。

## 4. 当前边界与后续

- 仅支持连续 FP32 输入；dtype promotion、非连续输入和 `cat.out`/
  `stack.out` 尚未实现。
- 输出和中间布局使用 uint32 元素索引；超过该范围的输入或输出在发射前拒绝。
- `index_add`、`topk` 仍是 W3 缺口；stride-aware
  elementwise/归约、`avg_pool2d` 与 RNG 也未在本节点处理。

## 5. 工作记录

| 步骤 | 操作 | 结果 |
|---|---|---|
| 1 | 审计 `aten::cat`/`aten::stack` schema 与 PyTorch 2.14 C++ 签名 | 确认 `cat` 使用 `at::ITensorListRef`，`stack` 使用 `at::TensorList` |
| 2 | 增加 `cat_args_t`、`cat_kernel` 和 host 拼接映射 | 统一覆盖 cat/stack，空片段不发射 |
| 3 | 注册 ATen 实现并加入 shape/dtype/layout/uint32 校验 | 错误在设备提交前报告 |
| 4 | 添加参数化 CPU 对比和边界测试 | 11 个用例覆盖轴、空维度、launch 计数和拒绝路径 |
| 5 | 重新 configure、编译并运行正式环境测试 | 11/11 通过 |
| 6 | 更新 README、W3 覆盖矩阵与路线图 | cat/stack 移入已支持，index_add 保留为下一缺口 |
