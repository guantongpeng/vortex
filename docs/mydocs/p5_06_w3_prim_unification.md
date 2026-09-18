# P5.6 W3.1(续):一元与归约并入 sw/dl 的 prim 库

实施日期:2026-09-18。承接 [p5_05_w3_operator_coverage.md](p5_05_w3_operator_coverage.md)。
本轮关闭它的 §5.7 前两条,**W3.1 的四个算子(migration 主体)至此全部完成**。
基线:`build_dl64`(XLEN=64 / simx),解释器 `~/miniconda3/envs/vortex/bin/python`。

## 1. 起点:一份没有调用方的改动

工作区里留着一份未提交的改动:`sw/dl/src/prim_args.h` 把 `vx_prim_op_e` 从 4 个一元
算子扩到 13 个,`prim_kernels.hip` 实现它们并把 reduce 的 NaN 语义修好。

但它**不生效**:公开头 `prim.h` 的枚举没扩,`prim_host.cpp` 仍写着
`if (op > VX_PRIM_OP_NEG) return VX_PRIM_ERR_BAD_ARGS;`——新算子一个都到不了 kernel。
本轮把它接完,并把扩展的 `unary_op_kernel` 与 `reduce_rows_kernel` 迁过去。

## 2. 一并修:DL 路径不算「设备工作」(独立提交,先落)

`vortex_free` 用 `work_since_last_barrier()` 决定能否立即 `hipFree`,而这个 epoch
只在扩展的 `launch()` 里更新。W3.1 把 conv/pool/bn/mm 迁到 `sw/dl` 之后,那四个入口
在 queue 上直接 `vx_enqueue_launch`(`queue.cpp` 的 `enqueue` 入 FIFO 即返回,worker
线程稍后执行),**不经过 `launch()`**。四个调用点各自手写 `++g_stats.launches;`,epoch
一次都没动。

实测(其余工作全部 retire 后):

```
conv2d(x, w); del x; torch.empty(同尺寸)
  -> frees=1, immediate_frees=1, launches=0
```

`launches=0` 是同一个缺陷的另一面:计数器被手工维护了,**生命周期依赖的那一半没有**。
地址于是被立刻还回设备分配器的空闲链,而 DL kernel 还排在队列里等着读它。

修法:抽出 `note_device_work()`(`launches` + epoch),`launch()` 与 DL 调用点共用它。
四个 DL 调用点改为 `DL_LAUNCH(...)`(init 用不着记账,仍是 `DL_CHECK`)。

测试:`test_memory.py` 新增参数化用例,把 `a + b` 换成 DL 算子。去掉记账即失败
(`immediate_frees == frees`),加上即通过。**先确认它会失败再留下**。

## 3. prim 侧

### 3.1 一元算子

13 个:RELU / GELU_TANH / SILU / NEG / ABS / EXP / LOG / SQRT / RSQRT / SIGMOID /
TANH / RECIPROCAL / GELU_ERF。枚举 append-only——镜像与 host 可能不同时构建。

`ABS` 由 `x < 0.0f ? -x : x` 改为 `fabsf(x)`:前者对 `-0.0` 返回 `-0.0`,而
`torch.abs(-0.0)` 是 `+0.0`。这类「同一个错误在这棵树里犯过」的写法,relu 已经栽过一次
(`x > 0 ? x : 0`)。

### 3.2 reduce 改为按行

由「单 CTA 归约整条向量」改为 `rows x cols`、**一行一个 CTA**——与 `prim_softmax_kernel`
和两个 norm kernel 同形。整条向量的归约是 `rows = 1`,也就是旧形态能表达的全部,所以是
加字段而不是加第二个入口。补 `MEAN`。

旧形态与 ATen 需要的形状对不上,这正是 §5.7 说「两套实现」的由来:ATen 归约的是
`(rows, cols)` 视图的尾维。

### 3.3 两个 NaN 缺陷

- **max 用 `fmaxf`**,它返回非 NaN 的那个操作数。行里任何一个 NaN 于是归约成其余值的
  最大值。`torch.amax` 与 `torch.max` 都传播 NaN。
- **argmax 的合并没有 NaN 分支**,所以 NaN 会输给任何「比它大」的数——而 argmax 恰恰是
  值与索引必须对同一元素达成一致的地方。

现在 NaN 胜出,且 (value, index) 合并是**同一个谓词**,两者不可能不一致。`torch.argmax`
平局取首个索引,也一并钉住。

### 3.4 枚举的两份声明有了机制

op 号声明了两次:一次给调用方,一次给看不到公开头的设备编译器(prim_args.h),而
`vx_prim_reduce` 用减法取 kernel 的 op。两份漂移的后果是**静默换一个算子**。现在
`prim_host.cpp` 用 `static_assert` 逐条钉住。

**第一版断言本身就是错的**——写成了 `VX_PRIM_OP_SUM == VX_PRIM_RED_SUM`,而两者本来就
相差一个偏移;编译器当场报错。这正是 §5.6 记的「一个事实、多份手工维护的副本」缺的那个
机制,能在编写时就抓到作者本人。

## 4. 扩展侧

`unary_op_kernel`、`reduce_rows_kernel`、它们的参数块与表行,以及**死代码 `relu_kernel`**
(`relu__impl` 一直走 `launch_unary_op`,`h_relu_kernel` 没有调用点)全部删除。`TORCH_UNARY_*`
与 `TorchReduceOp` 这两份第二编号也没了——扩展直接说 `vx_prim_op`。

`load_ops` 增加 `vx_prim_init`,`vortex_at_exit` 增加 `vx_prim_finalize`(在
`hipDeviceReset()` 之前)。

### 4.1 DL 入口拒绝、而本地 kernel 默默吸收的两件事

迁移会撞上它们,两件都要显式表态:

| 情形 | 旧本地 kernel | DL 入口 | 现在的处理 |
|---|---|---|---|
| `n == 0`(空张量一元)、`rows == 0` | grid 为 0 → 跳过 | 返回 ERR_BAD_ARGS | 显式不 launch,结果同前 |
| 全归约 `cols == 0` | 循环零次,输出留在**种子值** | 返回 ERR_BAD_ARGS | sum→0、mean→NaN(一次 fill);max **拒绝** |

`cols == 0` 那一格值得单说:torch 的 `sum`/`mean` 有答案(`0` / `NaN`),而 `amax` 本身
就报错——"amax(): Expected reduction dim to be specified for input.numel() == 0"。
旧 kernel 留下的是 `-INFINITY` 种子,是 torch 永远不会返回的值。所以这里让 sum/mean 写
出正确的identity,**max 与 torch 一样拒绝**,而不是造一个值。

`torch.empty(3, 0).sum(dim=1)` 仍是**拒绝**的:布局归一化要除以行长,没有形状可建。
这是先前就存在的缺口(W3.2/F09 范围),本轮只在测试里钉住它,没有顺手改。

## 5. 验收

计划书 W3.1 的三条:

- **同源 kernel**:`test_dl_bridge.py` 新增一元与归约的比对,判据是**逐位相同**而不是
  容差。归约这条**不是同义反复**:kernel 的累加次序与它取代的「一行一线程串行累加」
  不同(实测:同一份数据串行求和与 pairwise 求和相差 1.8e-07),所以第二份实现不可能
  通过它。
- **stream 一致**:DL 的 launch 走调用方当前流的队列(`current_queue()`),与扩展自己的
  launch 同队列定序。
- **数值与 ABI**:`tests/dl/prim` 覆盖 13 个一元算子 × `[-inf,-1,-0,0,1,inf,NaN]`,
  reduce 的 NaN/全 `-inf`、MEAN、多行归约、NaN softmax。`PRIM_REDUCE_SIZE`
  24→32(rv64)/16→24(rv32),由 `test_dl_metadata_matches_measured_sizes` 重测比对。

**每条新测试都先确认能抓住原缺陷**:

| 回退什么 | 失败信息 |
|---|---|
| reduce 的 NaN 合并 | `reduce_max dropped the NaN: got -1` / `argmax picked 1, expected the NaN at 3` |
| `ABS` 的 `fabsf` | `abs(-0.0) came back as -0.0` |
| DL 的 epoch 记账 | `immediate_frees == frees` |
| 空张量的不 launch | DL 入口的 ERR_BAD_ARGS 抛出 |

### 一处**没有**做的改动

softmax 的 row-max 原本也用 `fmaxf`。把它改成 NaN-aware 之后测试仍然通过——**因为它不
改变任何结果**:NaN 是经由 pass 2 的指数与该 pass 的和走到输出的,row-max 是不是 NaN-aware
无关。既然拿不出一个能失败的用例,就没有改,而是把原因写进 kernel 注释(「实测,不是假设」),
并把「NaN 行输出全 NaN」留成契约测试而非回归测试。

## 6. 本轮验证结果(2026-09-18)

```
build_dl64: pytest torch-vortex/tests -q --tier=full   -> 238 passed, 0 skipped (12m32s)
build_dl64/tests/dl/prim: make run-simx                -> PASSED (23 项)
```

`--tier=full` 是必要的:精简档跳过 MiniResNet,而模型门槛是 W3.1 的验收条件之一。

`test_modules.py` 原本手工加载 `prim.vxbin` 作为两个「额外镜像」之一——现在那是重复加载
后端已持有的文件,`hipModuleLoad` 以地址范围冲突拒绝。改为加载后端**不持有**的镜像
(`rng`、`quant`),并断言整个集合共存。

## 7. 仍未完成

- **二元 elementwise 没有迁**:DL 的 prim 没有二元算子,所以 `binary_op_kernel`、
  `scalar_op_kernel`、`broadcast_op_kernel` 与 `TorchBinaryOp` 继续留在 torch 镜像里。
  要统一就得先给 prim 加二元入口。
- `linear` 每次调用物化 `w.t().contiguous()`;DL gemm 没有 `transb`。
- `mxfp8_args.h` 的尺寸未纳入漂移测试(头文件在 host 侧编不过)。
- 三处重复的状态枚举(`mxfp8`/`nvfp4`/`rng`)没有机制保证一致。
- `torch.empty(3, 0).sum(dim=1)` 被拒绝(见 §4.1)。
- `avg_pool2d` 未注册;`max(dim=)`/`argmax` 到 ATen 的桥接(prim 侧已有行式 argmax)。
