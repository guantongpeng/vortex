# P5.8 W3.2:softmax / log_softmax / logsumexp

实施日期:2026-09-18。承接 [torch_p5_07_layer_norm_rms_norm.md](torch_p5_07_layer_norm_rms_norm.md)。
本轮做 W3.2 的第 3 项。基线:`build_dl64`,XLEN=64 / simx。

## 1. 三个算子一个 kernel

softmax、log_softmax 与 logsumexp 共用两次扫描(行内 max、平移后的指数与其和),
只有 pass 3 写出的东西不同:归一化分布、它的对数、每行一个 log-sum-exp。
所以 DL 侧是**一个 kernel 加一个 op 选择子**,不是三份拷贝。

`vx_prim_rowargs_t` 因此加了 `op`(2 指针 + 4 u32),`PRIM_ROWARGS_SIZE`
24→32(rv64)/ 16→24(rv32)。宿主侧是三个具名入口
(`vx_prim_softmax` / `vx_prim_log_softmax` / `vx_prim_logsumexp`),共用一份
实现——调用方不该看见选择子。

**logsumexp 的 `out` 是每行一个 float,不是每行 cols 个**,所以 pass 2 对它
根本不写 per-element 的值:写了就会越过缓冲区末尾。

## 2. 一个由测试抓出来的缺陷

第一版把 logsumexp 写成 `log(sum) + max`,和 torch 在**行 max 是 ±inf 的行**上不一致:

| 行 | torch | 第一版 |
|---|---|---|
| `[inf, 0]` | `inf` | `nan` |
| `[inf, inf]` | `inf` | `nan` |
| `[-inf, -inf]` | `-inf` | `nan` |

原因是平移 `inf - inf` 本身是 NaN,NaN 污染了和,`log(nan) + inf` 还是 NaN。
而 torch 的**答案就是那个无穷**:行 max 是无穷时,每个元素都不超过它,
log-sum-exp 就是它。

修法是 `isinf(m) ? m : logf(s) + m`。这一条是**先写出测试、看它失败**才发现的
——`tests/dl/prim` 与 `test_softmax.py` 各有一条,回退即失败:

```
logsumexp of a +inf row: got nan want inf
logsumexp of an all -inf row: got nan want -inf
```

softmax 与 log_softmax **不需要**这个特例:它们在同样的行上 torch 也给 NaN,
我们的 NaN 与它一致(这也是 p5_06 §5 记的「不改变结果就不改」的同一类判断——
区别在于这次测了)。

## 3. ATen 侧

注册三个 schema:

```
aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
aten::logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
```

`F.softmax`/`F.log_softmax`/`nn.Softmax`/`nn.LogSoftmax` 都经由 `_softmax` 系列的
composite,所以**只注册被分解到的那一层**就够——与 norm 族同一条经验,同样是
实测而不是按 dispatch 优先级推的。测试调用**公开拼写**。

### 3.1 布局:softmax 需要「搬过去再搬回来」

归约的输出是别的形状,所以 `reduce_layout` 只要把数据搬成 (rows, cols) 就够了。
softmax 的输出**与输入同形状**,所以中间维要搬两次:

```
x.movedim(dim, -1).contiguous()   ->  (rows, cols),内核的输入
kernel writes                     ->  (rows, cols),独立缓冲
out.movedim(dim, -1).copy_(...)   ->  调用方的 layout
```

尾维是直接用(张量连续,它的行就是内核的行),没有这一趟。
回写走 `launch_copy_strided`,所以**中间维的情况上限是 4 维**——与归约族同一条边界,
在 README 与测试里都写明。

中间维的两次搬运是**设备上的拷贝**,不是 host 往返;`test_steady_state_is_device_only`
量的是这一点。

### 3.2 实测出来的边界

| 情形 | torch | 本实现 |
|---|---|---|
| 0 维输入 `softmax(tensor(3.), 0)` | 0 维 | 0 维(**第一版错成 `(1,1)`**,被测试抓到) |
| 0 维 + `dim=1` | 报错 | 拒绝,报「0-D tensor takes 0 or -1」 |
| `(0,4)` 在 dim=1 | `(0,4)` | 不 launch,返回空 |
| `logsumexp` 多个 dim | 支持 | **拒绝**,与 `sum`/`mean`/`amax` 同一条 v1 边界 |
| `half_to_float=True` | 半精度路径 | 拒绝 |
| 非连续输入 | 可算 | 拒绝(与归约族同一条边界) |

## 4. 验收

- **同源 kernel**:`test_dl_bridge.py` 新增 `vx_prim_softmax` / `vx_prim_log_softmax` /
  `vx_prim_logsumexp` 的**逐位相同**比对(尾维,避开中间维的搬运——那是另一个问题)。
- **数值**:`tests/dl/prim` 覆盖 log_softmax、logsumexp、±inf 行;
  `test_softmax.py` 覆盖每个 dim、每个特殊值、退化形状与拒绝。
- **等价性**:`exp(log_softmax) == softmax`(DL 侧)与
  `log_softmax(x) == x − logsumexp(x, keepdim=True)`(ATen 侧)各钉一条。

### 4.1 本轮结果(2026-09-18)

```
build_dl64: pytest torch-vortex/tests -q --tier=full  -> 284 passed, 0 skipped (20m09s)
build_dl64/tests/dl/prim: make run-simx               -> PASSED
```

284 = 上一轮的 259 + 本轮 25(`test_softmax.py` 22 + `test_dl_bridge.py` 3)。
`--tier=full` 含 MiniResNet 模型门槛。

### 4.2 一条测试在测什么、不在测什么

`test_log_softmax_is_not_log_of_softmax` 用 `[0, -100, -200]`:普通量级下
`log(softmax(x))` 与 `(x − max) − log(sum)` 数值上无从区分,只有到 exp 下溢时才分开。
写这一条是为了让「实现的是稳定式」成为**可检验的性质**,而不只是注释里的一句话。

## 5. 仍未完成

- ~~**`argmax` / `max(dim=)`**~~ —— **P5.10 已做**,见
  [torch_p5_10_argmax.md](torch_p5_10_argmax.md)。DL 一次归约同时写 uint32 索引和极值,
  torch 侧只做表示转换为 ATen 要求的 int64。
- **`topk`**。
- **`interpolate`**、stride-aware 的 elementwise 与归约、RNG(W3.5)。
- **`nll_loss_forward`**:`F.cross_entropy` 在 `log_softmax` 之后还会在这一步拒绝
  ——softmax 族的「动机消费者」不是 softmax 自己。
- 多 dim 归约、非尾维归约的 stride-aware 版本。
