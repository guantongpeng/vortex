# P5.7 W3.2:layer_norm 与 rms_norm

实施日期:2026-09-18。承接 [p5_06_w3_prim_unification.md](p5_06_w3_prim_unification.md)。
本轮做 W3.2 的第 4 项(norm 族)。基线:`build_dl64`,XLEN=64 / simx。

## 0. 为什么先做 norm 而不是 softmax

p5_05 §3 的建议顺序把 softmax 排在 norm 前面。本轮改了顺序,理由是**审计在
`sw/dl` 的 norm kernel 里找到两个已实测的缺陷**,而 softmax 的 DL kernel 经查是好的:

| 缺陷 | 实测 |
|---|---|
| `prim_layernorm_kernel` 用一遍 `E[x²] − mean²` 求方差 | 输入 `[1e6, 1e6+1, 1e6+2, 1e6+3]` 时两项都是 ~1e12、差是 ~1,有效数字全丢:方差算出 0,rstd 算出 `1/sqrt(eps)=316.23`,而真值是 1.25 与 0.8944237 |
| `prim_reduce_kernel` 的 argmax 合并把两个 NaN 当成不等 | `[1, NaN, NaN, NaN]` 返回索引 2,torch 返回 1;21 宽、首个元素是 NaN 的行返回 16,torch 返回 0 |

**两个缺陷都测不出来,因为测试的参考实现自己也带着同一个错误**:`tests/dl/ref.h`
的 layernorm 用同一遍公式(double 也照样抵消),`tests/dl/prim` 的 argmax 用例没有 NaN。
这正是 p5_05 §5.1 记的「把缺陷写成 oracle 的测试永远通过」。

两个都是「先修共享 kernel,再迁移」这条 W3.1 规矩的直接实例,所以它们先落,并在
`tests/dl` 里各配一条**回退即失败**的用例。

## 1. `sw/dl` prim 侧的改动

### 1.1 layernorm 的方差改成两遍

一遍式在精确算术里等价、在 float32 里不等价。现在是「对均值的偏差的平方的平均」,
第二遍扫描。ATen 用 Welford(跨线程递推,这东西在这个 CTA 形状里没有),两遍的
数值与它一致到 float32。

同时删掉了原来的 `fmaxf(var, 0)` 钳位:它是针对抵消结果的补丁,抵消没了,
钳位只会把负方差藏起来。

### 1.2 可选 affine

`F.layer_norm(x, shape)` 与 `nn.LayerNorm(elementwise_affine=False)` 传 `None`,
而入口原来写死 `!gamma || !beta → BAD_ARGS`,这条路**根本不可达**。现在 gamma 为空
表示「不做 affine」,是模式不是调用错误;**只给一半**仍然是调用错误。

gamma/beta 一起检查,是因为 kernel 成对读它们:只给一个会静默忽略给了的那个。

### 1.3 mean 与 rstd 输出

`aten::native_layer_norm` 返回 `(out, mean, rstd)`,没有第二个入口会算它们,所以是
schema 要求而非便利。两者都是每行一个 float、可为 0(调用方不要)。约定是
`rstd = 1/sqrt(var + eps)`——**是倒数**,不是 var 也不是 1/var;这一条用手算的
1.25/0.8944237 钉住。

`vx_prim_norm_args_t` 因此从 4 指针变 6 指针 + 4 个 u32:`PRIM_NORM_SIZE`
48→64(rv64)/ 32→40(rv32),由 `test_dl_metadata_matches_measured_sizes` 重测比对。

### 1.4 argmax 的 NaN 平局

两处都要改,方向相反:

- **合并**(`argmax_better`):`NaN != NaN` 为真,所以 `if (v != bv) return v > bv`
  把两个 NaN 送进大于比较、得到 false、保留左操作数——赢家是树最后访问到的那个。
  现在两个 NaN 视作平局,按索引取小。
- **逐 lane 扫描**:`|| isnan(x)` 让一个已经持有 NaN 的 lane 把后面每个 NaN 都收进来,
  于是一行里保留的是**最后一个** NaN 而不是第一个。改成 `isnan(x) && !isnan(seen)`。

`tests/dl/prim` 的用例直接对着 torch 的答案(1 与 0),不是对着自己的另一种写法。

## 2. ATen 侧

注册两个 schema:

```
aten::native_layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight,
                        Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)
aten::rms_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight=None,
               float? eps=None) -> Tensor
```

**只注册 `native_layer_norm` 就够**了:`aten::layer_norm` 是 composite,会调它。
这一条是**实测**的,不是按 dispatch 优先级推的——上一轮的工作流里有个 agent 按
优先级论证「注册 layer_norm 才有效」,而实际跑下来 `F.layer_norm`、`nn.LayerNorm`
在只注册 `native_layer_norm` 时都工作。测试因此调用**公开拼写**,不调用被注册的那个。

分工与 conv 相同:**ATen 校验形状,DL 库计算**。`normalized_shape` 必须是输入的
尾维,权重必须恰好 `cols` 长;两者都给不满足时报出两个形状。

### 2.1 实测出来的三件事

- **`rms_norm` 的 `eps=None` 是机器 epsilon,不是 0**,float32 下 1.1920928955078125e-07。
  实测:eps=None 与 eps=finfo.eps 逐位相同,与 eps=0 不同。传 0 会让两者差一点点。
- **`native_layer_norm` 的 mean/rstd 形状是「输入形状、归一化维置 1」**,`(2,3,4)` 配
  `[4]` 得到 `(2,3,1)`。
- **两种零形状的答案不同**,这点会诱人写成同一个分支:

  | 输入 / 形状 | torch | 实现 |
  |---|---|---|
  | `(0,4)` / `[4]` | 三个输出都是空的 | 不 launch,统计量也是空的 |
  | `(2,0)` / `[0]` | out `(2,0)` 空,mean **0**,rstd **NaN** | 不 launch,但用一次 fill 写出 0 与 NaN |

  第二种不是「跳过」而是「有值要写」,所以它走 fill kernel 而不是直接返回。

### 2.2 边界

- 非连续输入**拒绝**:`aten::layer_norm` 的 composite 把跨步张量原样交下来,所以这与
  归约族是同一条边界(`check_vortex_f32` 要求连续),不是新的限制。
- 权重不是恰好 `cols` 长(例如只按尾维广播)**拒绝**,属 W3.2 后续。
- fp16/fp64 **拒绝**(分配可以,计算不行)。

## 3. 验收

计划书 W3.1 的三条,与 conv/pool/bn/mm 同一套:

- **同源 kernel**:`test_dl_bridge.py` 新增 `vx_prim_layernorm`(out/mean/rstd 三者分别)
  与 `vx_prim_rmsnorm` 的**逐位相同**比对。
- **stream 一致**:走 `current_queue()`,与扩展自己的 launch 同队列定序。
- **数值与 ABI**:`tests/dl/prim` 覆盖 4 个新用例;

**每条新测试都先确认能抓住原缺陷**:

| 回退什么 | 失败信息 |
|---|---|
| layernorm 第二遍扫描 | `rstd for 1e6-scale input: got 0.0097 want 0.8944236`(而 `layernorm`/`ln_rstd` 在普通数据上**仍然通过**——这就是它原来测不出来的原因) |
| argmax 的两个 NaN 平局 | `argmax of a 4-wide row: got 2 want 1` / `got 16 want 0` |
| kernel 的可选 affine 分支 | 两个 no-affine 用例读到空 gamma |

### 3.1 本轮结果(2026-09-18)

```
build_dl64: pytest torch-vortex/tests -q --tier=full  -> 259 passed, 0 skipped (14m15s)
build_dl64/tests/dl/prim: make run-simx               -> PASSED
```

`--tier=full` 含 MiniResNet 模型门槛。259 = 上一轮的 238 + 本轮 21 条
(`test_norm.py` 14 条 + `test_dl_bridge.py` 2 条 + `tests/dl/prim` 的 4 条在另一棵树)。

## 4. 一个**没有**做的改动

`vx_prim_softmax` 经查**不需要改**:它的 row-max 用 `fmaxf`(不传播 NaN),但 softmax
的 NaN 是经由 pass 2 的指数与该 pass 的和走到输出的,row-max 是不是 NaN-aware 无关。
p5_06 §5 已经量过这一点;本轮再次确认后没有动它。softmax 族留在下一轮,那时的
DL 侧工作只有 `log_softmax`/`logsumexp` 两个新入口。

## 5. 仍未完成

- **softmax / log_softmax / logsumexp**:`vx_prim_softmax` 就绪;`log_softmax` 与
  `logsumexp` 需要新的行入口(`vx_prim_rowargs_t` 要加 `op`,尺寸 24→32/16→20)。
  注意 `logsumexp` **不**经过 `_softmax`,它是 amax/exp/sum/log(实测)。
- **argmax / `max(dim=)`**:DL 的 row-wise ARGMAX 已修好且就绪,但写出的是 uint32,
  而 ATen 要 int64——需要一个 torch 侧的小 kernel 做加宽(不是 sw/dl 的改动)。
- **norms 的更多形态**:非连续输入、按尾维广播的权重、`group_norm`。
- `bmm`、`avg_pool2d`、`cat`/`gather`、stride-aware 的 elementwise 与归约、RNG。
- 二元 elementwise 仍在 torch 镜像(DL 的 prim 没有二元入口)。

## 6. 本轮工作流发现的、计划书未记的

用一份 8 agent 的 recon/design/critic 工作流做了范围界定,它产出的价值不在方案本身
(方案的若干论断被 critic 推翻、我又逐条实测复核),而在**它点出的边界**:

1. `aten::nll_loss_forward` 未注册,所以 `log_softmax` 做完之后 `F.cross_entropy`
   还会在下一步拒绝。softmax 族的「动机消费者」不是 softmax 自己。
2. `aten::_to_copy` 未注册,`torch.softmax(x, dim, dtype=...)` 的 dtype 分支会死在
   `copy_` 的类型检查上——错误信息该指向 `copy_`,不是 `_to_copy`。
3. W3.4 还要 embedding/gather、RoPE、dropout(`F.dropout` 也未注册),本轮不宣称接近
   W3.4。
4. `check_reduce_dims` 对 0 维输入的每个 dim 都判越界,而 torch 在 0 维上接受
   `argmax(t, 0)`/`argmax(t, -1)`/`argmax(t, None)` 都返回 0。argmax 落地时要单独处理。
