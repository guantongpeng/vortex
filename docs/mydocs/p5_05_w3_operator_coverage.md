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
8. **W3.1 与 `sw/dl` 统一** —— 本轮没有做。`sw/dl` 的 `prim_reduce`/`prim_unary`/`prim_softmax`/`prim_layernorm` 都是现成的,本轮的归约与一元 kernel 是**重复实现**;W3.1 要求的「kernel 算法只维护一份」仍未满足,需要把扩展链到 `libvortex_dl` 并统一 device/context 生命周期。
9. **`max(dim=)`/`argmax`/`topk`** —— 需要索引归约。

## 4. 本轮的一个实现教训

跨步拷贝的第一版把 **CPU 指针传给了 device kernel**,于是 `strided_vortex.cpu()` 往设备写零、host buffer 原样不动。拷贝 kernel 寻址的是设备内存,所以**两侧都必须在设备上**:host 源要抬上去,host 目标要先汇集到连续的设备缓冲再搬下来。测试同时覆盖两个方向才抓到它——跨步源是读模式,跨步目标是写模式,只测一边会掩盖另一边。

`gelu` 的第一版只实现了 tanh 形式并把它当成默认值。实际上 torch 的默认是 erf 形式,`approximate="tanh"` 是**另一个函数**而不是同一个的另一种拼法。schema 不匹配(我们的 kernel 收一个参数,而 schema 有两个)会直接导致 import 失败,这才暴露出来。
