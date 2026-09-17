# P5.5 W3.2:算子覆盖(首轮)

实施日期:2026-09-17。承接 [p5_04_m1_w2_device_basics.md](p5_04_m1_w2_device_basics.md)。本轮做计划书 W3.2(通用 Tensor 与基础算子)里可用性收益最大的部分。**W3 未完成**,缺口列在 §3。

## 1. 做了什么

### 一元/二元的 op-code 化

elementwise 原本一个算子一个 kernel。改成**按 arity 的 op-code 驱动 kernel**:一个 kernel,操作由参数块里的字段选择,枚举定义在 `kernels/torch_kernel_args.h`(host 与 device 必须一致,不一致就是静默错值)。

`binary_args_t` 的 padding 字改成了 op code,`unary_args_t` 白捡一个,所以**参数块尺寸没变**。

新增:`sub`/`div`、`maximum`/`minimum`(传播 NaN,与 `fmaxf` 不同)、就地 `add_`/`sub_`/`mul_`/`div_`、`neg`/`abs`/`exp`/`log`/`sqrt`/`rsqrt`/`sigmoid`/`tanh`/`reciprocal`/`silu`/`gelu`(两种形式)。

**每一条都留在简单的 grid-stride 形状里**——这是刻意的,`tv_mm_kernel` 的注释记着 kernel 形状一旦不简单会发生什么。

### 标量算子

`x * 3.0` 之前被拒,报 "mul.b is on cpu rather than the vortex device"。原因是**PyTorch 把 Python 数字交给 tensor 重载**(`mul.Tensor` 配一个 0 维 CPU 张量),而不是 `mul.Scalar`;把它识别成标量是后端的事,CUDA 后端就是这么做的。

### 广播

`x + bias` 是模型真正会写的形状,之前是 `TORCH_CHECK`。elementwise 现在接 operand 的 stride 而不是假定连续:**广播维就是 stride 0**,和跨步拷贝用的是同一个机制。输出连续,所以只有 operand 带 stride。等形状走原来那个线性 kernel(已知对 VOLT 安全的形状)。

形状合法性交给 `at::infer_size`(ATen 自己的规则),所以被拒时报的是 ATen 的原文。就地版本里「会撑大 self 的广播」被拒——多出来的元素无处可放。

### as_strided 与跨步拷贝

`aten::as_strided` 之前没有实现,于是 **transpose / permute / t / 所有索引全部不可达**——报错说的是 `as_strided` 而不是用户写的那个操作,因为它们都分解到它。现在委托给 `at::native::as_strided_tensorimpl`,和 `view` 委托 `view_symint` 是一个路子。

跨步拷贝 kernel 让跨步张量**可用**:`hipMemcpy` 搬的是一段线性范围,转置视图不是。支持 ≤4 维的任意 stride、storage offset、广播(stride 0)。

这同时**兑现了 W1 里程碑里点名推迟的「跨步目标」**:`t.t().to("vortex")` 因为 `_to_copy` 保留源布局而要求跨步目标,现在能用了,而不是被拒。

### 归约

`sum`/`mean`/`amax`/`max`,全归约或单维,支持 `keepdim`。一个 kernel 归约连续 (rows, cols) 视图的**尾维**——一行一线程,无 LMEM、无树。其余情况(全归约、中间维)在 host 侧用 `movedim`+`contiguous` 归一化成那个形状。归一化的拷贝要付一趟;stride-aware 归约是显然的下一步,W7.1 会量它值不值。

累加器初值是 `-INFINITY` 而非有限哨兵,且 NaN 胜出——`amax` 传播 NaN,而 pool kernel 当年用 `-3.4e38f` 做种子会静默丢掉 `-inf` 输入。**这两种特殊值都有测试**,因为同一个错误在这棵树里已经犯过一次。

维度校验在**分配输出张量之前**——坏调用不该先算出结果尺寸。测试抓到的是反过来的顺序。

## 2. 可用性矩阵(实测,2026-09-17)

```
SUPPORTED                          UNSUPPORTED
  clone/contiguous                   bmm (batched)          aten::bmm.out
  t/transpose/permute                cat/stack              aten::cat.out
  slice/index                        softmax/log_softmax    aten::_softmax.out
  as_strided                         layer_norm             (BN training 分支)
  mm/mv/linear/addmm                 group_norm             aten::var_mean.correction
  broadcast add/mul                  avg_pool2d             (参数解析)
  elementwise(sub/div/min/max/…)     interpolate            aten::upsample_nearest2d.out
  unary(exp/log/sqrt/…/silu/gelu)    gather/scatter         aten::gather.out
  sum/mean/amax/max                  randn/rand             aten::normal_
  conv2d / BN(推理) / pool           topk/sort              aten::topk.values
  torch.Stream / torch.Event         cumsum                 aten::cumsum.out
```

**MiniResNet 全前向仍然通过**(conv/bn/relu/maxpool/add/avgpool/linear/view),这是本轮之后的模型级门槛。

## 3. W3 未完成的部分(按建议顺序)

1. **`bmm` / batched matmul** —— host 侧按 batch 循环现有 mm kernel,成本低,transformer 必需。
2. **`cat`/`stack`/`gather`/`scatter`/`index_add`** —— 数据搬运,各自一个简单 kernel。
3. **`softmax`/`log_softmax`/`logsumexp`** —— 归约已就位,再加一个 exp 归一化 pass。
4. **`layer_norm`/`group_norm`** —— 需要 `var_mean`(归约 + 平方),然后 affine。
5. **stride-aware 的 elementwise 与归约** —— 现在 `x.t() + 1` 与 `amax(dim=0)` 都要付一次拷贝。
6. **`avg_pool2d`**、`interpolate`、`ceil_mode` —— W3.3。
7. **`randn`/`rand`** —— 属 W3.5,需要 `c10::GeneratorImpl`;`sw/dl` 的 Philox kernel 已经存在且有测试,缺的是接到 PyTorch 的生成器接口。
8. **W3.1 与 `sw/dl` 统一** —— 已开始,见 §5。`conv2d` 与 pooling 已迁到 DL kernel 并删除重复实现;`batch norm` 与 `mm` 尚未。`sw/dl` 的 `prim_reduce`/`prim_unary` 与本轮的归约/一元 kernel 仍是**重复实现**。
9. **`max(dim=)`/`argmax`/`topk`** —— 需要索引归约。

## 4. 本轮的一个实现教训

跨步拷贝的第一版把 **CPU 指针传给了 device kernel**,于是 `strided_vortex.cpu()` 往设备写零、host buffer 原样不动。拷贝 kernel 寻址的是设备内存,所以**两侧都必须在设备上**:host 源要抬上去,host 目标要先汇集到连续的设备缓冲再搬下来。测试同时覆盖两个方向才抓到它——跨步源是读模式,跨步目标是写模式,只测一边会掩盖另一边。

`gelu` 的第一版只实现了 tanh 形式并把它当成默认值。实际上 torch 的默认是 erf 形式,`approximate="tanh"` 是**另一个函数**而不是同一个的另一种拼法。schema 不匹配(我们的 kernel 收一个参数,而 schema 有两个)会直接导致 import 失败,这才暴露出来。


## 5. W3.1:与 `sw/dl` 统一(进行中)

### 5.1 审计发现:直接迁移会倒退三个修复

W3.1 原文警告「先对照测试再迁移 mm/conv/norm;同步修复共享 kernel 中的对应问题,避免将已知错误从一层搬到另一层」。**这条警告是准的**,审计在 `sw/dl` 里找到了三个 W1 已修、而 DL 一直没修的问题:

| W1 的修复 | `sw/dl` 的状态(实测) |
|---|---|
| F03 BN 通道索引 `plane = total/c`(= N\*H\*W) | **未修**。N=2 时 16 个输出错 8 个 |
| F10 pool 用有限哨兵 `-3.4e38f` | **未修**。`-inf` 与 NaN 都变成该哨兵 |
| F01 relu `x > 0 ? x : 0` | **未修**。NaN 归零、`-0.0` 丢符号 |

两个 DL 测试之所以测不出来,是因为**测试的参考实现自己就写了同一个缺陷**:BN 测试跑 `N = 1`(`total/c` 恰好等于 `H*W`),pool 测试的参考初值也用了 `-3.4e38f`。把缺陷写成 oracle 的测试永远通过。

所以先修 DL 的 kernel(含新增的 N>1 与 `-inf`/NaN 用例,并确认这两条用例在旧 kernel 下会失败),再迁移。

顺带修的:BN 原本收预计算的 `rstd`(迫使调用方做一次 host 往返算 sqrt,即 F11 的形状),现在收 `var`+`eps`,rstd 在 kernel 内算;`vx_dnn_bn_affine` 的 grid 按 4 线程算而共享 launch 助手硬编码 16 线程(过度 4 倍 CTA)。

### 5.2 桥接

DL 库说的是 vortex2.h,需要**本进程的** device 和 queue——自己开第二个 device 就是第二个 context,会让 DL 的 launch 失去与 HIP 侧的定序关系。所以 `sw/hip` 把两个句柄交出来(`hipGetVxDevice`、`hipStreamGetQueue`),扩展用它们初始化 DL 模块。两个镜像集现在同进程共存——这正是 W2.4 多模块工作解锁的;在此之前,这正是 ops 与 dnn kernel 被合并成单一镜像的原因。

### 5.3 已迁移的算子与验收

`conv2d`、`max_pool2d`/`adaptive_avg_pool2d` 已走 DL kernel,扩展里对应的 kernel、参数结构体与 launch 助手已删除。

**验收按计划书的三条**:

- **同源 kernel**:`torch.nn.functional.conv2d` 与直接调 `vx_dnn_conv2d` 的结果**逐位相同**。这里刻意不用容差——两个独立实现会在累加次序内一致,容差恰好会掩盖「其实是两份实现」;同一个 kernel 则不可能有差异。对比 CPU 参考时才用容差,因为那**确实是**另一个实现。
- **stream 一致**:DL 的 launch 走调用方当前流的队列(`current_queue()`),与扩展自己的 launch 同队列定序。
- **数值与 ABI**:DL 侧新增的 N>1/特殊值用例通过;扩展侧 ABI 三方比对(边车 / 扩展编译期 / 现场 `sizeof`)保持通过。

分工由此明确:**ATen 校验,DL 计算**。边界检查(`window_out`、范围检查)留在扩展——它们会指名被拒的参数,而且是这条路径上唯一的带符号算术;DL 的形状算术是无符号无检查的。

### 5.4 尚未完成

- **`batch norm`**:DL 的 BN 要求 weight/bias 指针都非空,而 ATen 侧支持 affine 可选。要么给 DL 加 `has_affine`(torch-vortex 的 kernel 已经有),要么在缺省时传 1/0 缓冲。
- **`mm`**:DL 的 gemm **保留了内联的 alpha/beta epilogue 分支**——正是我在 `tv_mm_kernel` 里因为 VOLT 误编译而移出的那种形状。迁移前必须先用 `test_mm_partial_tiles` 那套形状测它,否则可能把误编译带回来。审计明确建议先测。
- **`prim`**:`vx_prim_unary`/`vx_prim_reduce` 与本轮的归约/一元 kernel 仍是两套实现;`prim_reduce` 的 max 用 `fmaxf`(丢 NaN),迁移前要一并修。
- **`sw/dl` 的 init 与 metadata**(审计发现,未修):init 第二次调用是**静默 no-op**,不看 device 也不看路径;部分初始化失败会留下 `module` 已设而某个 kernel 槽为 nullptr 的状态,而 runtime 把 nullptr kernel 当「legacy escape hatch」——于是会**以 PC 0 成功入队**。`args_size` 也仍是手填且漂移(conv 声明 96 实际 88)。这些是 `sw/dl` 自身的账,不是迁移引入的,但会随迁移一起被继承。
