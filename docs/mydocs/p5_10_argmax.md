# P5.10 W3.2:argmax 与 max(dim=),以及一个 VOLT 误编译的复现

实施日期:2026-09-22。承接 [p5_09_review_findings.md](p5_09_review_findings.md)。
本轮做 W3.2 的索引归约族。基线:`build_dl64`,XLEN=64 / simx。

## 1. 一个 kernel,一对结果

`aten::argmax` 要的是索引,`aten::max(dim=)` 要的是 **(值, 索引)**。而 DL 的
`prim_reduce_kernel` 在找索引的同时**本来就把极值握在手里**(`seen`)——分开两次归约
是同一份数据的第二遍扫描。

所以给 `vx_prim_reduce_args_t` 加了一个 `values` 指针(arg 两个算子专用,可为 0),
一次 launch 同时写出两者。索引仍是 uint32(`vx_prim_reduce` 的既有契约)。

**uint32→int64 的加宽不是 sw/dl 的事**:那是 ATen 的表示,不是计算。新增一个
`widen_u32_i64_kernel` 放在 torch 镜像里,O(rows) 的纯转换,不是第二遍归约。

## 2. tie 规则:值与索引必须指向同一个元素

`torch.max(dim=)` 的契约不只是「值是最大值」,而是**「值等于 `x[indices]`」**。
tie 时这一条才有内容:一个把平局判给**较晚**元素的归约会给出一个正确的最大值,
但值/索引这一对是错的。

`nan_max` 原来写 `a > b ? a : b`,平局保留右操作数(合并树里是较晚的那个)。
改成 `a >= b ? a : b`,平局保留左操作数,与索引「取最小索引」的规则一致。

实测(`amax`,ATen 侧):

| 行 | torch | 改前 | 改后 |
|---|---|---|---|
| `[0.0, -0.0]` | `+0.0` | `-0.0` | `+0.0` |
| `[-0.0, 0.0]` | `-0.0` | `+0.0` | `-0.0` |

**只有 ±0 的平局能看出这个差别**,而 `assert_close` 看不见零的符号。两棵树里各加了
一条用例:`tests/dl/prim` 的 tie 块(回退即 `max of a tie gave -0, want 0`)与
`test_reductions.py::test_amax_of_a_signed_zero_tie`。

`max(dim=)` 的**值**走的是 arg 那条路(`idx_better` 按索引决策),所以它本来就不受
`nan_max` 影响——两条路径现在都给「最早索引处的值」。

### 2.1 ATen 侧实测出来的边界

| 情形 | torch | 本实现 |
|---|---|---|
| `argmax(t)` 无 dim | 0 维 int64 | 0 维 |
| `argmax(empty, dim)` rows=0 | 空结果 | 不 launch,返回空 |
| `argmax(empty)` 无 dim | 报错「requires a reduction dim」 | 拒绝,同义 |
| `argmax(t, dim=d)` 且 size(d)=0 | 报错「dim d to have non-zero size」 | 拒绝(`reduce_layout`) |
| 0 维输入 + dim=0 | 0 | 支持(`row_dim`/`reduce_layout` 的 0 维分支) |

## 3. 本轮的主要产出其实是一个工具链缺陷的复现

原计划把 **min/argmin/amin** 一并做掉(kernel 里加两个 op code,索引比较加一个方向)。
写完之后 `tests/dl/prim` 全线崩溃:**所有归约都返回 `-inf`**。

定位过程(每一步都是实测,不是读代码推断):

1. 直接写探针调 `vx_prim_reduce(SUM)`,`sum([1,2,3,4])` 得到垃圾 → 排除 torch 侧。
2. 探针改成让 kernel 把 `arg->cols`/`arg->op`/`arg->rows`/`arg->in` 写进输出 →
   **四个字段全对**(cols=4, op=0, rows=1)。
3. 再让它写各线程的**累加器** → `1,2,3,4,0,...` 全对。
4. 再让它写 **warp 归约后**的值 → `10,12,14,16,...` 全对。
5. 再让它写 **LMEM 里的值** → 全 `0`。**写入没有落进 LMEM。**

然后二分:把 kernel 还原到 HEAD,逐条加回改动,每次编译+运行。

**结论:单独把累加器的种子改成依赖 `arg->op` 就足以让整个 kernel 坏掉**——
不管写成嵌套三元还是 `if/else if` 语句。改成依赖 op 的种子之前,HEAD 的
`(arg->op == VX_PRIM_RED_MAX) ? -INFINITY : 0.0f` 是好的。

这正是 kernel 文件开头记的那条:

> Arg fields are read directly, never cached into locals (the VOLT -O3 miscompile workaround).

以及 `blas_kernels.hip` 里更具体的版本(2026-09-14 实测):

> do NOT cache arg->M/N/K/alpha/beta into local const variables under -O3 — VOLT
> miscompiles the cached loads (wrong results across CTAs; exact mechanism TBD,
> likely load hoisting around the staged argument block).

**本轮把「read arg-> directly」这条规避也推翻了**:这里的代码**逐个使用点都直接读
`arg->op`**,没有把它存进 local,照样坏。所以已知规避不是「不缓存」,而是某种
更窄的形状约束,而现有文档没有说出那个形状是什么。

### 3.1 本轮怎么处理

**把 min/argmin 从本轮摘出去**,kernel 的 op 集合保持与 HEAD 一致(SUM/MEAN/MAX/ARGMAX),
只留下两处**已验证可用**的改动(`>=` 的平局规则、`values` 输出)。
`argmin`/`min(dim=)`/`amin` 写进 README 的「尚未支持」,并注明原因是内核还没有
min 归约。

这是**权宜**:计划书 W5.4 要求「对 VOLT 历史 -O3 误编译建立最小复现并修工具链根因;
若确需临时补丁,按仓库规则标记并建立后续修复」。本轮交出了**可复现的触发条件**
(种子依赖 `arg->op` → LMEM 写入丢失),但没有定位到根因,也没有修工具链。
最小复现的脚本与探针留在 `/tmp/probe/`(会话级,不入库),复现步骤见 §3 的 1–5 步。

## 4. 验收

- **同源 kernel**:`test_dl_bridge.py` 新增一条,把 ATen 的 `max(dim=)` 与直接
  `vx_prim_index_reduce` 的索引和值**逐位**比对。
- **不变量**:`test_argmax.py::test_the_value_is_the_index` 用 `gather` 检查
  「值 = `x[indices]`」,这是 tie 时唯一有内容的契约。
- **实测边界**:§2.1 的表格逐条有用例。

### 4.1 结果(2026-09-22)

```
build_dl64/tests/dl          make run-simx          -> 11/11 PASSED
build_dl64: pytest test_argmax.py + DL bridge         -> 28 passed (临时本地环境门槛)
```

**每条修复都先确认能抓住原缺陷**:把 `nan_max` 改回 `a > b`,`tests/dl/prim` 报
`max of a tie gave -0, want 0`(两次),`test_reductions.py` 报 `amax` 的零符号反了。

### 4.2 结果

本轮在 `build_dl64` 重新 `configure` 后重新编译了 DL 库、全部 DL 测试和 torch kernel image。
`make -C tests/dl run-simx` 的 11 个子套件全部通过；其中 `tests/dl/prim` 新增的
argmax 值/索引配对、±0 tie 和多行归约均通过。

torch-vortex 的当前容器是 PyTorch 2.4.1 + Python 3.8。扩展原先按 2.14 API 编写，
在这个基线下会因为 `CachingDeviceAllocator`、`DeviceCapability` 和若干 SymInt/schema
差异无法重新编译。本轮把这些差异收敛在 `vortex_ext.cpp` 的版本分支里：旧版本使用基础
`c10::Allocator`、旧的 `c10::string_view`/view API，并跳过只在新 schema 存在的 norm 和
adaptive-pool 注册；2.14 路径保留原来的 DeviceAllocator 统计和 SymInt 实现。扩展随后在
2.4.1 下成功编译、加载，手工 oracle 覆盖 2D/3D/4D 各维、`keepdim`、全量归约、NaN、
±0 tie、重复最大值及空行，全部与 CPU 一致。

pytest 默认不能在该环境直接声称通过：测试夹具依据 `torch_vortex/env/supported.json`
将 Python 3.8/torch 2.4 标为未支持组合，所以会在 fixture 阶段拒绝；这不是算子失败。
为验证实现本身，临时只修改了 build-tree 的支持表（源树文件已恢复），移除该环境门槛后
运行 `test_argmax.py` 加新增的 DL bridge 用例，结果为 **28 passed**。这次结果是实现级
验证，不把 2.4/3.8 变成正式支持环境；正式验收仍需在支持表中的 PyTorch 2.14/Python
3.10 环境运行同一测试和完整 suite。

附加的静态检查：`make -s -C torch-vortex/kernels`、`git diff --check` 和 Python 测试文件
编译均通过。

## 5. 仍未完成

- **min / argmin / amin**:曾尝试增加 kernel op,但被 VOLT 误编译挡住；当前 kernel
  仍只包含 SUM/MEAN/MAX/ARGMAX,见 §3。
- **`argmax` 的 `out=` 变体**、`topk`。
- **`nll_loss_forward`**:`F.cross_entropy` 在 `log_softmax` 之后仍会在这一步拒绝。
- **`avg_pool2d`**、`index_add`、stride-aware 的 elementwise 与归约、RNG。
- **`bmm` 已在后续 P5.11 完成**，见 [p5_11_bmm.md](p5_11_bmm.md)。
- **VOLT 误编译的根因**:§3 给了触发条件,没给机制。

## 6. 工作记录

| 步骤 | 操作 | 结果 |
|---|---|---|
| 1 | 检查工作树、历史提交和 P5 文档 | 确认本轮改动集中在 `sw/dl`、`torch-vortex`、`tests/dl/prim`；发现 argmax 实现尚未提交，`min/argmin` 仍被 VOLT 触发条件阻塞 |
| 2 | `build_dl64/../configure --xlen=64 --tooldir=/data/vortex-tools` | 重新生成 build-tree 副本，确保 `torch-vortex` 和 `VX_config.h` 不陈旧 |
| 3 | 重编 `sw/dl` 与 torch kernel image | `prim_reduce_args_t` 的 40-byte RV64 ABI 与 image metadata 一致 |
| 4 | `make -C tests/dl clean && make -C tests/dl run-simx` | BLAS、prim、quant、DNN、MXFP8、NVFP4、2:4、attention、LLM、Mamba、RNG 共 11 套件全部 PASS |
| 5 | 追踪 torch pytest 首次无输出 | 定位为中断构建留下的 `_ext/lock`；清理锁后暴露 PyTorch 2.4 缺少新 allocator/schema API 的编译差异 |
| 6 | 为 `vortex_ext.cpp` 加 PyTorch 版本分支 | 2.4 用基础 allocator、`c10::string_view` 和 int view；2.14 保留 DeviceAllocator、DeviceCapability、SymInt 与新 norm/pool schema |
| 7 | 重新编译并加载 torch extension | 当前 2.4.1 环境成功编译、注册并加载 `argmax`/`max.dim` |
| 8 | 手工 CPU oracle 脚本 | 2D/3D/4D、所有维度、keepdim、整 tensor、NaN、±0 tie、重复最大值和空行全部通过 |
| 9 | 临时放宽 build-tree 环境门槛，运行 `test_argmax.py` 与 DL bridge 用例 | 28 passed；源树 `supported.json` 未修改，结果不升级正式支持声明 |
| 10 | `make -s -C torch-vortex/kernels`、`git diff --check`、`py_compile` | 全部通过 |

这份记录保留了两个容易误判的失败：旧 build 中未重编 `sw/dl` 会造成新旧 arg-block
不匹配并表现为 kernel 不返回；而当前 Python 3.8/torch 2.4 不是支持表中的 pytest
环境，fixture 拒绝属于环境门槛，不能记作算子回归失败。
