# P5.9 对 P5.7/P5.8 的对抗性复审,及其修复

实施日期:2026-09-18。对象是 [p5_07](p5_07_layer_norm_rms_norm.md) 与
[p5_08](p5_08_softmax_family.md) 的七个提交。基线:`build_dl64`。

## 0. 为什么单独写一篇

那两轮的每一处修复都配了「回退即失败」的用例,但**用例是自己写的**:它能证明
「这个改动有效」,不能证明「该找的地方都找过了」。所以另跑了一份 4 维复审 +
逐条对抗性证伪的工作流(27 个 agent,产出的 23 条里 **19 条存活**)。本文记录其中
属于本轮代码的缺陷与修复;**它推翻了 P5.6/P5.7 里我自己写下的一条结论**(见 §4)。

## 1. 一个 use-after-free,在新路径上被踩响

`copy_impl` 的 D2D 跨步分支末尾写着 `note_default_queue_barrier()`。这行在它
**以 `hipMemcpy` 结尾**的时候是对的:`hipMemcpy` 入队后自己等完成事件,FIFO 队列
意味着它之前的一切都已 retire,于是「队列已空」这个断言成立。

而 `launch_copy_strided` 只是**入队一个 kernel**。断言留在原地之后,分配器的
epoch 被清空,`work_since_last_barrier()` 返回 false,下一次释放就走立即
`hipFree` —— 而拷贝还排在队列里等着读那块 source。

实测(复审给出的复现,我在修复前后各跑一次):

```
dst.copy_(base.t());  del base      修复前 frees=1 immediate=1   修复后 immediate=0
F.softmax(x, 1)                     修复前 immediate=2          修复后 immediate=0
```

第二行是**本轮引入的调用方**:softmax 的中间维路径把 `moved` 与 `L.shaped` 两个
缓冲区交给 kernel,而它们的唯一持有者是 C++ 局部变量,函数返回即释放。
`test_memory.py` 新增用例把它钉住。

这行来自 3eb9873ea(上一轮的跨步拷贝),一直在那里,只是**当时没有调用方会释放
一个 kernel 的输入**。这与 P5.6 §2 是同一条规律的两个实例:迁移会把潜伏的
生命周期缺陷变成活的。

## 2. 两个错误的答案

### 2.1 `logsumexp` 的无穷捷径与 NaN

P5.8 给 logsumexp 加了「行 max 是 ±inf 时答案就是它」的捷径,用来对齐 torch。
但 pass 1 的 row max 当时是 **NaN-blind** 的(`fmaxf`),所以 `isinf(m)` 对
`[NaN, +inf]` 也成立,捷径返回 `+inf`,而 torch 返回 NaN。

修法不是给捷径打补丁,而是**让 row max 变 NaN-aware**:这样 `isinf(m)` 才真的
意味着「这一行确实是无穷的」。

**这推翻了 P5.6 §5 与 P5.7 §4 里我写下的结论**——那两处说「把 softmax 的 row max
改成 NaN-aware 不改变任何结果,所以不改」。对 softmax 与 log_softmax 仍然成立;
对 logsumexp 不成立,因为我给它加的捷径依赖 `isinf` 与 NaN 可区分。当时的结论
在当时的代码上是对的,在新增捷径之后就错了——这正是「不改变结果就不改」这条
判断的适用边界:它只在**没有新增依赖**时成立。

覆盖:`[NaN,+inf]`、`[+inf,NaN]`、`[NaN,-inf]`、`[NaN,0]` 四个组合,回退即失败。

### 2.2 LayerNorm 的方差:只改一遍不够

P5.7 把一遍式 `E[x²] − mean²` 换成两遍式,并用 `[1e6, 1e6+1, 1e6+2, 1e6+3]`
钉住。**两遍式还不够**:它从**取整后的 float32 均值**算偏差。

那四个数的精确均值是 `1e7+1.5`(推广到 1e7 一档时),在 ulp 为 1 处不可表示,
于是偏差算成 `[-2,-1,0,1]` 而不是 `[-1.5,-0.5,0.5,1.5]`,方差是 1.5 而不是 1.25,
rstd 是 0.8165 而不是 0.8944 —— **8% 的误差**,而行更长时更大(8e7、14 列时 34%)。

修法:**两遍都在「减去行首元素」的坐标里做**。`in[j] - in[0]` 是精确的(移位后的
值是小的),所以两遍的求和都被行的**离散程度**而不是**量级**所决定。均值输出是
`anchor + mean_shift`。

测试因此要两个量级:**1e6 抓一遍式,1e7 抓「不锚定的两遍式」**。只测 1e6 会让
修复看起来完整。

## 3. 三处「torch 能算而我们拒绝」

| 情形 | torch | 修复前 | 现在 |
|---|---|---|---|
| `F.layer_norm(x, [4], weight=w)`(bias 缺省) | 算,等价于 bias=0 | 拒绝 | 算,gamma/beta 各自可选 |
| `normalized_shape=[]` | 报错 | **接受**,且算出无意义的结果 | 拒绝 |
| 权重形状 `(2,2)` 配 `[4]` | 报错(元素数对、形状错) | 接受 | 拒绝 |

第一条需要 DL 侧配合:`vx_prim_layernorm` 原来要求 gamma/beta **同时**给或同时
不给(那是我 P5.7 自己加的检查),而 torch 允许只给一个。现在两者独立可选,
`gamma` 空表示 gamma=1、`beta` 空表示 beta=0。

顺带把 affine 的校验**移到分配之前**:原来三个输出张量先分配、再校验权重,
拒绝时留下三个设备块。

`logsumexp` 的空张量也一并修了:torch 返回 `-inf`(空和的 log),而内核拒绝零形状,
所以那个值由一次 fill 写出——与 `reduce_into` 为 sum/mean 写 identity 是同一个做法。

## 4. 四处「测试测不出东西」

这一类的价值不低于前几类:它们让上面那些缺陷**曾经可以存在**。

1. **`check()` 把 NaN 记成通过**。它算 `|got − ref|`,而 `|NaN − ref|` 是 NaN,
   `NaN > rtol` 是 **false**,于是 kernel 答 NaN、参考是有限值时**判为通过**。
   所有基于它的断言都是 NaN 盲的。现在 NaN 按类别比较。

2. **`exp(log_softmax) == softmax` 比较的是两个 host 参考**,与 kernel 无关:
   把 log_softmax 的 pass 3 换成常量,这条仍然报 ok。现在比较的是 **kernel 的
   两个输出**。

3. **`sin[0] = 50.0f // large value exercises the max-shift` 是假的**。
   50 远不到 `expf` 的上溢点(88.7),去掉 max shift 结果仍然精确到 1e-7,
   **测试照样通过**。改成 100 之后,去掉 shift 会得到 `inf/inf = NaN`,
   两条断言同时失败(见 §4 的 `max_rel=inf`)。

4. **rms_norm 的 eps 断言比较的是 device 与 CPU**,而两者的差是 4.8e-07,
   远在容差之内——把 `eps=None` 换成 `0` 仍然通过。现在比较 **device 对
   device**,逐位。

5. argmax 的 `[NaN,1,NaN,NaN] → 0` 这一条**修前修后同解**,不暴露任何东西;
   换成 17 宽、首尾都是 NaN 的行(线程 0 同时拥有下标 0 与 16),它才真的抓
   `|| isnan(x)` 那个逐 lane 缺陷。

## 5. 记录在案但未改的两处

- **`amax` 与 `torch.amax` 在零的符号上不一致**:`[+0.0,-0.0]` 的 `amax` 是
  `+0.0`,内核给 `-0.0`。**torch 自己的 `amax` 与 `max` 在这上面也不一致**
  (`max` 给 `-0.0`),而后端把两者都映射到 `VX_PRIM_OP_MAX`,所以**不可能同时
  对齐**。这是先前的边界,不是本轮引入的;要处理需要区分两个 op。
- **rv32 的 `args_size` 列基本没被检查**:`test_dl_metadata_32_bit_column` 只解析
  到 `ARGS_SIZE` 一个变量,34 行里 31 行被 `continue` 掉。**数值本身是对的**
  (与实测一致),但把它改错测试也不会响。这是测试的缺口,属 W7.2。

## 6. 本轮结果

```
build_dl64/tests/dl     make run-simx   -> 11/11 PASSED
build_dl64: pytest torch-vortex/tests -q --tier=full  -> 288 passed, 0 skipped (22m49s)
```

**每条修复都先确认能抓住原缺陷**:

| 回退什么 | 失败信息 |
|---|---|
| layernorm 的锚定(`anchor = in[0]` → `0.0f`) | `rstd for 1e+07-scale input: got 0.8164939 want 0.8944236` |
| softmax 的 row max 改回 `fmaxf` | `logsumexp of [NaN,+inf]: got inf`(×3) |
| softmax 的 max shift | `softmax[0] got=nan ref=1.000000`,`max_rel=inf` |
| argmax 的逐 lane 扫描 | `argmax of a 17-wide row: got 16 want 0` |
| 跨步拷贝后的 `note_default_queue_barrier` | `test_free_is_deferred_after_a_strided_copy` |

288 = 上一轮的 284 + 本轮 5 条新用例 − 1 条被替换掉的旧用例。`--tier=full` 含
MiniResNet 模型门槛。

## 7. 复审本身的一个教训

工作流给出的 23 条里有 4 条被证伪、19 条存活,而**存活的那 19 条我逐条自己复跑过**
才动手。第一次复审(P5.8 前)有两条关键论断我没验证就采信了方向——这次的经验是:
**复审的产出是候选缺陷,不是缺陷**,它的价值在于指出**去哪里看**,不在于代替看。

真正值钱的是它的**方法**:让 agent 去**编译并运行** kernel 而不是读代码推理。
本轮的三个 kernel 缺陷(§2.1、§2.2、以及 `check()` 的 NaN 盲区)都是这样抓到的,
其中 §2.2 的 8e7 一档是我自己没想到去试的量级。
