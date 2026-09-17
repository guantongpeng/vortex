# P5.3 M0+W1:可信的 FP32 eager 基线

实施日期:2026-09-17。基线 commit `d8458f058` + 本次改动。执行 `docs/mydocs/pytorch_plan.md` 的第一轮(M0 + W1),目标是把已有的 eager 子集从"能跑"变成"可信"。

## 1. 做了什么

按计划书 §5.1 建议的顺序交付:W0.1 版本锁定 → W1.6 工厂 dispatch → W0.2 干净构建与 2.14 迁移 → W1.7 参数 ABI → W1.1/W1.5/W1.4/W1.2/W1.3/W1.8 算子语义。

### W0.1 环境与版本锁定

- `torch_vortex/env/{manifest,supported.json,check}.py`:`collect()` 记录 python/torch/triton/C++ ABI、build(XLEN/TOOLDDIR/驱动/工具链版本)、source(git commit/dirty)、artifacts(vxbin 与扩展 `.so` 的 sha256),每次 pytest session 写入 `<build>/torch-vortex/env_manifest.json` 并打印。
- `supported.json` 是手写可评审的窄表:supported = {torch 2.14.0, py 3.10, triton 3.8.0, xlen 64, simx}。`check.py` 比对后 **unsupported 非零退出、untested 大声告警**。
- 解释器路径只出现在 `TORCH_VORTEX_PYTHON` 与 `ci/testcases/torch.yaml`;库代码里零硬编码。

### W0.2 干净 out-of-tree 构建 + torch 2.14 迁移

- `configure` 的 `SUBDIRS` 增加 `"!torch-vortex*"`,构建树里的 Python 包、C++ 源、kernel Makefile **全部由 configure 生成**,不再手工拷贝(此前 `build_dl64/torch-vortex` 是手工副本,新 configure 出来的树跑不了测试)。`Makefile.in` 增加 `torch:` 目标(与 `hip:` 一样按需,不进 `all`)。
- **扩展在 torch 2.14 下直接编译通过**,但踩到两点:
  1. **torch 2.14 要求 C++20**。显式传 `-std=c++17` 会让 ATen 头以 `"C++20 or later compatible compiler is required"` 失败。不要覆盖 `cpp_extension` 的默认标准。
  2. **`rename_privateuse1_backend` 单独不再生成 `.vortex()`/`.is_vortex`**,必须调用 `torch.utils.generate_methods_for_privateuse1_backend()`(2.5 时代的 `generate_tensor_methods_*` 已不存在)。P5-01 记录的该条踩坑在 2.14 上是**反的**。
- **Python device module 是硬前提**:缺少 `torch.vortex` 时 `torch.zeros(device="vortex")` 会在 dispatch **之前**抛 `ModuleNotFoundError`。`_register_device_module` 因此移到 `load_ops` 之前。
- `PrivateUse1HooksInterface` 在 2.14 上**不是**加载前提(缺它只是 `TORCH_CHECK_NOT_IMPLEMENTED`),但补了一个最小实现。
- 2.4 踩坑清单逐条复核,结论已并入 P5-01 文档的修订说明。

### W1 算子语义(全部 P0)

| 编号 | 修复 |
|---|---|
| F01 | `relu` 拆成 `relu`/`relu_`:非原地版本分配输出,输入逐元素不变、`data_ptr` 不同、版本计数器不动。kernel 改为 `x < 0.0f ? 0.0f : x`(NaN 传播、**`-0.0` 保持符号**) |
| F02 | `addmm` 不再委托 `linear`。三者分离:未知 epilogue 的缩放走独立 in-place kernel,严格实现 `beta*self + alpha*(mat1@mat2)`,`beta=0` **不读** `self`(NaN/Inf 被忽略) |
| F03 | BN 通道索引:参数改为显式传 `hw = H*W`,kernel 用 `(idx/hw)%C`。原来的 `total/c = N*H*W` 只在 N=1 时正确 |
| F04 | `copy_` 先限定安全 fast path(同 dtype/同 shape/连续目标/offset 0/非别名),其余**指名拒绝**;非连续 CPU 源先 `contiguous()` |
| F05 | `view` 改调 `at::native::view_symint`,直接复用 ATen 的 `-1` 推导、numel 校验、`computeStride`、storage_offset 与版本计数器 |
| F06 | **删除** `aten, BackendSelect` 的全局注册(它覆盖了 ATen 自己的 backend-select kernel,把 `torch.empty(3)`/`zeros`/`arange`/`device='meta'` 全部改道 CPU);PrivateUse1 单独注册已足够 |
| F07 | 新增 `kernels/torch_kernel_args.h`,host 与 device **共用一份**参数布局;`gen_metadata.cpp` 用实测 `sizeof` 生成 VXKMDATA;launch 改用 `HIP_LAUNCH_PARAM_BUFFER_SIZE` 取 host 的 `sizeof`,metadata 不再参与字节数计算;init 时与 `<vxbin>.meta.json` 边车交叉校验 |
| F08 | BN 删掉整个 host 往返与 scratch buffer(**这同时消除了 use-after-free 的触发点**);rstd 在 kernel 内计算 |
| F09 | 形状助手改用 `int64_t` 并命名报错参数(stride≤0、padding<0、窗口非法);launch helper 对 grid 为 0 的输入**不 launch** |
| F10 | max pool 初值 `-INFINITY`,更新条件 `v > acc \|\| isnan(v)` |
| F11 | 见 F08:稳态 BN 的 H2D/D2H 字节数为 0 |
| F12 | 保留 `AutogradPrivateUse1` fallthrough(属 W8.1),但补测试钉住"backward 必须抛错" |

另修:新增抛出型 `PrivateUse1` catch-all,未注册算子**报出算子名并拒绝**,取代 README 里那个不存在的 `fallback_counter`;`torch.vortex.is_available` 从 `@property` 改为**方法**(2.14 的 FakeTensor 会调 `is_available()`,property 直接 `TypeError`);设备模块补 `_is_in_bad_fork`/`manual_seed_all`;`device="vortex:1"` 不再被静默接受为 `vortex:0`。

### 观测性

扩展内统计计数器,`torch_vortex.stats()` 暴露 `{launches, skipped_launches, h2d/d2h/d2d_bytes, allocations, frees, blocking_syncs, host_numeric_ops}`。这让"设备上真的跑了"从断言变成可检验的性质:BN 稳态 0 字节传输、空输入 0 launch、`relu` 两消费者 `launches == 2`(CPU 回退会是 0)。计数器是进程级、非线程安全,是诊断不是 profiler(W7.1 属真正的)。

## 2. 两个实现过程中发现、计划书未记录的问题

### 2.1 VOLT 误编译:matmul kernel 的 epilogue(重要)

**把 addmm 的 `alpha/beta/self` 分支放进 `tv_mm_kernel` 的 epilogue,会让 VOLT 对任何 `n` 不是 16 的倍数的形状累加错误。** 现象:输出是残留内存或错误值,`(16,16,16)`/`(32,32,32)` 正确而 `(4,4,4)`/`(17,17,17)`/`(1,16,4)` 全错。

定位过程(可作为最小复现的基础):
1. 设备端探针确认 `m/n/k/transb/alpha` 与 `blockDim/gridDim` **全部正确**,`out` 指针正确 —— 排除 ABI、launch 形式与运行时。
2. 保留新结构体布局、把 epilogue 还原成原样 → **正确** —— 排除结构体布局。
3. 保留原 epilogue 一行式、把分支加回去(**分支体替换为 `v += arg->beta;`**)→ **复现** —— 触发条件是分支本身,不是它的语义。
4. 同一段代码放进独立的 elementwise kernel → **正确**。

规避:`tv_mm_kernel` 回到"纯 `out = a@b`"的已知正确形态;`alpha/beta/self` 的缩放移到新的 `tv_mm_epilogue_kernel`(与既有 `tv_bias_add_kernel` 同构)。`tests/test_matmul.py::test_matmul_kernel_carries_no_epilogue` 用 args_size 把这个拆分钉住,`test_mm_partial_tiles` 覆盖回归形状。

工具链根因未修(属计划书 W5.4「对 VOLT 历史 -O3 误编译建立最小复现并修工具链根因」),本次只做到可复现的最小形态与规避。

### 2.2 `hipDeviceSynchronize` 是空操作(**未修,本轮范围外**)

`torch.vortex.synchronize()` 目前返回成功但什么都没做:`hipDeviceSynchronize`/`hipStreamSynchronize` 都映射到 `vx_queue_flush`,而 `Queue::flush`(`sw/runtime/common/queue.cpp:155-160`)只 notify 一个 condvar。真正的屏障是 `vx_queue_finish(q, VX_TIMEOUT_INFINITE)`。

**陷阱**:`vx_queue_finish(q, 0)` **不是**永久等待 —— `Event::wait_value` 对 `timeout_ns=0` 走 `wait_for(0ns)` 立刻返回 `VX_ERR_TIMEOUT`。

测试之所以现在能过,是因为 `hipMemcpy` 自身会等待其完成事件;launch 则确实异步。属计划书 W2.2/W2.1,本轮不动 `sw/runtime` 故未修。

## 3. 验证记录

全部在 `build_dl64`(XLEN=64, TOOLDIR=/data/vortex-tools, simx)上,解释器 `~/miniconda3/envs/vortex/bin/python`(Python 3.10.21 / torch 2.14.0+cpu)。

```
$ python -m pytest torch-vortex/tests -q                 # 快档
86 passed, 4 skipped in 76s

$ python -m pytest torch-vortex/tests -q --tier=full     # 含模型门槛
6 passed (test_resnet.py)  —— 另有快档 86 项

$ python -m torch_vortex.env.check
status: supported        (改坏 torch 版本后 -> unsupported, exit 1)

$ python3 ci/testcase.py lint | grep -i torch
(无错误)
$ python3 ci/testcase.py matrix --tier full | grep torch
{"category": "torch", "driver": "host", "xlen": 64, "needs": []}
```

模型门槛(`test_resnet.py`,batch 1/2/3 + 随机化 BN + 中间层逐层对拍)全部与 CPU 一致,容差 rtol=atol=1e-3。

**这些用例是能发现原 bug 的**:`test_batch_norm_matches_cpu[2]`/`[3]`(随机化逐通道参数 + batch>1)在修复前必然失败;`test_mm_partial_tiles` 覆盖 epilogue 回归的全部形状;`test_factories.py` 用子进程比对 import 前后的 CPU/Meta 工厂行为。

已知未解:测试进程退出时会 core dump(计划书 F23 记录的退出析构竞态),测试本身通过、退出码为 0,归 W2.3。

## 4. 已知边界与刻意收窄

以下全部**大声拒绝并指名归属工作项**,没有静默回退(README 有对照表):

- **dtype 转换**(`copy_` 的 float32↔float16/int64):压到 W3.2/W4.1。这是本次唯一一处**收窄计划书 W1.4 书面验收**的地方 —— 理由是已核查目标模型(MiniResNet,全 FP32 连续)不触发 cast,该转换集在本轮为空;计划书 W1.4 的正向转换用例因此改为断言明确报错的负例。
- **标量算子、`add_`/`mul_`、`as_strided`/`.t()`**:W3.2。`x + 1.0` 会以 0 维 CPU 张量进入 `add.Tensor`,报错信息现在指名"标量算子"而非"必须是 vortex 张量"。
- **跨步目标**:`t.t().to("vortex")` 会**响亮拒绝**(ATen 的 `_to_copy` 请求跨步目标)。比原来的静默错拷好,但确实还不可用,属 W3.2。
- **设备 RNG**:`torch.randn(device="vortex")` 需 Generator,属 W3.5。
- **`avg_pool2d`/`ceil_mode`/分组卷积**:W3.3。

## 5. 下一步

计划书 §5.1 的第 7 项之后:**W2.4 同进程多模块共存**是 DL 库统一、Triton 与 ATen 共用的共同阻塞;`synchronize()` 的诚实化与延迟释放属 W2.1/W2.2。工具链侧的 VOLT 误编译建议随 W5.4 一并处理(本轮已给出最小形态)。
