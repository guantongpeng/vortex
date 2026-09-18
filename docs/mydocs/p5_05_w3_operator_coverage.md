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
3. ~~**`softmax`/`log_softmax`/`logsumexp`**~~ —— **P5.8 已完成**,见
   [p5_08_softmax_family.md](p5_08_softmax_family.md)。三个算子共用一个 DL kernel。
4. ~~**`layer_norm`/`group_norm`**~~ —— `layer_norm`/`rms_norm` **P5.7 已完成**,见
   [p5_07_layer_norm_rms_norm.md](p5_07_layer_norm_rms_norm.md);`group_norm` 仍未做。
5. **stride-aware 的 elementwise 与归约** —— 现在 `x.t() + 1` 与 `amax(dim=0)` 都要付一次拷贝。
6. **`avg_pool2d`**、`interpolate`、`ceil_mode` —— W3.3。
7. **`randn`/`rand`** —— 属 W3.5,需要 `c10::GeneratorImpl`;`sw/dl` 的 Philox kernel 已经存在且有测试,缺的是接到 PyTorch 的生成器接口。
8. ~~**W3.1 与 `sw/dl` 统一**~~ —— **已完成**,见 §5 与 [p5_06](p5_06_w3_prim_unification.md)。`conv2d`、pooling、`mm/linear/addmm`、`batch norm`、一元与归约全部走 DL kernel,重复实现已删除。**例外**:二元 elementwise(`binary_op_kernel` 等)仍留在 torch 镜像,因为 DL 的 prim 只有一元;要统一需先给 prim 加二元入口。
9. **`max(dim=)`/`argmax`/`topk`** —— DL 的 row-wise ARGMAX 已修好(P5.7),差一个
   uint32→int64 的加宽 kernel(torch 侧,不是 sw/dl)。

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

### 5.4 已完成的四个算子

`conv2d`、`pooling`、`matmul`(mm/linear/addmm)、`batch norm` 全部走 DL kernel,扩展里对应的
kernel、参数结构体与 launch 助手已删除。`batch norm` 是最后也是最关键的一个:审计发现的
F03 就在它的 DL 实现里,所以「ATen 与 DL 直调逐位相同」这条同时证明了用的是**修好的那份**,
而不是它的副本。

障碍是可选 affine:ATen 允许 weight/bias 缺失,而 DL 入口要求两者都非空。解法不是传 1/0 缓冲
(那能work,但是对参数含义的谎),而是给 kernel 加 `has_affine`、host 从指针推导;只给其一按
调用方错误拒绝。

### 5.5 迁移带出的 `sw/dl` 自身欠账(已修)

- **init 参数检查**:每个模块的 init 第二次调用都直接返回 OK,**不看 device 也不看路径**。第二个
  调用方拿着不同镜像会静默拿到第一个调用方的 kernel。现在同参数才幂等,不同则返回
  `ERR_ALREADY_INITIALIZED`(已修 `blas`、`dnn`;其余 9 个模块同样的问题仍在)。
- **部分初始化失败**:kernel 查找失败会留下 `module` 已设的状态,之后每次 init 都返回 OK 而某个
  slot 是 nullptr——而 nullptr kernel 是 runtime 的 escape hatch,于是 launch 会**以 PC 0 成功入队**。
  现在失败即 finalize。
- **`args_size` 手填且漂移**:用 kern 源码里的单参数类型取出 kernel→struct 映射,用 host 编译器
  实测两套 XLEN(host 与 rv64 同为 LP64;`-DVX_CFG_XLEN=32` 且不定义 `__VORTEX__` 走头文件的
  `uint32_t` 指针分支,即 rv32 布局),**改掉 13 个数字**。其中 rv64 分支 3 个且都是危险方向
  (声明小于实际):`quant_gemm_w8a8` 48→56、`llm_embedding`/`llm_kv_append` 24→32;另加此前
  已修的 dnn 三个。新增测试对每个可测镜像重测并比对,把值改回去即失败。

### 5.6 init 检查已覆盖全部 11 个模块

`blas`/`dnn` 之后把其余 9 个也补上了。脚本化做这件事本身暴露了两点:

- **三个模块持有公开状态枚举的第二份副本**(`mxfp8` 在 `mxfp8_args.h`,`nvfp4`/`rng` 在自己的
  host 文件里),其中两个还重复声明了公开函数。只改公共头**够不到它们**——第一次尝试把头文件
  改对了,却在 .cpp 里报「新枚举未声明」。这与手填 `args_size` 是同一类缺陷:一个事实、多份手工
  维护的副本,副本之间会漂移。现在是自洽的,但没有任何机制维持它。
- **脚本改写必须逐文件改、编译、失败即回退**:把「按原文算出的下标」拼进「已被前一步改动的
  字符串」会写坏文件——第一次尝试就是这么让 9 个文件全部报 `expected unqualified-id` 的。
  逐文件的代价是一个模块而不是整库;`mxfp8` 的 init 与 `rng` 的两处枚举最后仍需手工处理。

### 5.7 仍未完成

> 前两条已在 P5.6 关闭,见 [p5_06_w3_prim_unification.md](p5_06_w3_prim_unification.md):
> 一元与归约已迁到 DL 的 prim kernel（`unary_op_kernel`/`reduce_rows_kernel` 及参数块已删除）,
> `prim_reduce` 的 NaN 语义与按行形态都已修。W3.1 的算子迁移至此完成;二元 elementwise
> 仍留在 torch 镜像,因为 DL 的 prim 没有二元入口。

- ~~`prim`:`vx_prim_unary`/`vx_prim_reduce` 与本轮的归约/一元 kernel 仍是两套实现;`prim_reduce`
  的 max 用 `fmaxf`(丢 NaN),迁移前要一并修。~~(P5.6 已关闭)
- 二元 elementwise(`binary_op_kernel`/`scalar_op_kernel`/`broadcast_op_kernel`)仍是 torch
  镜像里的实现:DL 的 prim 只有一元。要统一需先给 prim 加二元入口。
- `linear` 每次调用物化 `w.t().contiguous()`——DL gemm 没有 `transb`,这次拷贝是「一个 GEMM
  而不是两个」的代价。给 `vx_blas_gemm` 加 `transb` 是后续项。
- `mxfp8_args.h` 的尺寸未纳入漂移测试:它的头文件在 host 侧编不过(声明了设备侧辅助函数),
  探针无法包含它。
- 三处重复的状态枚举(见 §5.6)没有机制保证一致。
- `avg_pool2d` 未注册(`count_include_pad` 默认语义与 DL 的 in-bounds count 不同,属 W3.3)。
