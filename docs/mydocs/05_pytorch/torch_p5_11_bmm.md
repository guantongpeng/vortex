# P5.11 W3.2/W3.4：bmm 批量矩阵乘法

实施日期：2026-09-22。承接 [torch_p5_10_argmax.md](torch_p5_10_argmax.md)。本轮关闭
W3.2/W3.4 的 batched GEMM 缺口，基线为 `build_dl64`、XLEN=64、simx、
Python 3.10.21 / PyTorch 2.14.0。

## 1. 实现

`aten::bmm` 注册到 `bmm_impl`。ATen 侧检查两个输入都是连续的 FP32 三维
张量，形状为 `[B,M,K]` 和 `[B,K,N]`，并明确报告 batch 或 contraction
不匹配。输出为连续的 `[B,M,N]`。

实现复用现有 `vx_blas_gemm`，按 batch 计算三个矩阵片段的字节步长，在同一个
当前队列上连续提交 B 次 GEMM。队列本身有序，因此批次之间不需要 host 同步，
也没有把中间结果搬回 CPU。`beta=0` 使 GEMM 不读取尚未初始化的输出。

空 batch 直接返回空输出；`K=0` 返回全零结果；其他零维度不发射非法的零网格。
每个实际 batch 对应一次 DL GEMM，统计计数器也能观察这一点。

## 2. 验收用例

`torch-vortex/tests/test_matmul.py` 新增：

- 三组非方阵、非整 tile 的 `[B,M,K] @ [B,K,N]`，与 CPU `torch.bmm` 对比；
- B=4 的 launch 计数，确认是四次同源 DL GEMM 而不是 host 计算；
- B=0 与 K=0 的边界；
- batch 不匹配、contract 不匹配仍在任何 launch 前拒绝。

现有 `mm`、`linear`、`addmm` 回归也在同一文件中运行。

## 3. 结果

```text
build_dl64: make -s -C torch-vortex/kernels       -> PASS
build_dl64: pytest torch-vortex/tests/test_matmul.py
           -> 25 passed in 9.03s
build_dl64: smoke 分拆运行（323 项）                  -> 317 passed, 6 skipped
```

smoke 分拆是因为一次性运行在约 66% 后出现无输出停滞；按测试文件重新运行后，
所有非 `slow` 用例均完成，没有失败。6 个跳过项来自 `test_resnet.py` 的 slow
模型用例，符合默认 smoke tier。

测试使用项目记录的正式组合（Python 3.10.21、PyTorch 2.14.0、simx），环境
manifest 判定为 `supported`。首次编译遇到 `/home/guantp/.ccache` 只读的环境
错误；按 AGENTS 规则以 `CCACHE_DISABLE=1` 重跑，随后编译和测试通过。该环境
问题不涉及实现代码。

## 4. 当前边界与后续

- 只支持 FP32、连续三维输入；broadcast、transpose view、FP16/BF16 和
  `bmm.out` 仍拒绝。
- 当前设计是 B 次 GEMM 提交，正确性优先；真正的 batched/tiled GEMM 可在
  后续 BLAS ABI 增加 batch stride 后合并发射。
- `interpolate`、stride-aware elementwise
  与归约、RNG 仍是 W3 缺口。

## 5. 工作记录

| 步骤 | 操作 | 结果 |
|---|---|---|
| 1 | 审计 `mm_launch`、`vx_blas_gemm` 和 PyTorch 注册表 | 确认已有 row-major GEMM 可按连续 batch slice 复用 |
| 2 | 在 `vortex_ext.cpp` 增加 `bmm_impl` 与 `aten::bmm` 注册 | 形状、dtype、连续性和 uint32 边界在发射前检查 |
| 3 | 重新 `configure` 并编译 torch kernel image | build-tree 使用最新源文件 |
| 4 | 增加 CPU 对比、launch 计数和空维度测试 | 覆盖数值、队列提交和边界行为 |
| 5 | 用 PyTorch 2.14 正式环境运行 matmul 回归 | 25/25 通过 |
| 6 | 更新 README、P5.5 覆盖矩阵与路线图 | bmm 从未支持列表移入已支持列表 |
