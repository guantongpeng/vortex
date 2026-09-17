# P5.4 M1 完成 + W2:内存、stream 与设备基础

实施日期:2026-09-17。承接 [p5_03_m0_m1_baseline.md](p5_03_m0_m1_baseline.md)。本轮把计划书的 **W2 全部**做完,`pytorch_plan.md` §5.1 的第 7、8 项,以及第 3 项的收尾。

M1 的门槛是「W1 + W2 的内存/设备基础」,现已满足;W2 的完成也让 M2 只剩 W3(算子覆盖)与发布/CI。

## 1. W2.1 异步内存生命周期

**审计实测到的窗口**(不是推断):`x = torch.relu(x)` 之后,输入张量的地址在 launch 返回 **44 µs** 后就被回收,而 kernel 还要跑 **11.6 秒**。c10 的 deleter 调 `hipFree`,它当场把地址还回设备分配器的空闲链,完全不管还在读它的 kernel。计划书点名的 BN scratch 早已不存在,但这条路径**从普通算子链就能到达**。

修法:`hipFreeAsync`(在默认队列上,当前所有算子都在它上面)按队列顺序释放,地址要等队列排空才回来。

**但不能每次都延迟**:不做快速路径时整个套件从 25s 涨到 48s(没有地址被复用)。所以加了 epoch 对:队列是 FIFO 的,任何 host 侧屏障(`hipMemcpy` 等自己的完成事件、`hipDeviceSynchronize` 完成所有队列)都会retire它之前的全部命令,于是「上次屏障之后没有 launch」就等价于「没有在途 kernel 可能读这个 buffer」。典型循环下 20/20 次释放走快速路径。

`hipMemcpy` 只排空默认队列,这点单独处理:凡是在显式流上 launch 过,就置一个 flag,只有设备级屏障能清它。

## 2. W2.2 stream / event / guard

原 guard 是**装饰性的**:当前设备与当前流是进程级 `static inline`(不是 thread_local);`getStream()` 无视自己存的流、总返回默认流;`getNewStream()` 返回默认流(所以两个 `torch.Stream` 相等);`exchangeStream()` 写进去没人读;而**每个 launch 都传 nullptr**,所有设备工作都进同一个队列。HIP 层的流和事件本来就是真的(`hip_native/stream_overlap` 一直通过),缺的只在 guard。

现在:当前设备/流是 thread_local;`getNewStream` 建真实队列并注册新的 c10 stream id;launch 解析当前流(这条是让 stream context 有意义的关键,把它改回去测试就会失败);事件方法全部实现,转发到 HIP 层已有的 `hipEvent_t`,语义只有一份。

`queryStream`/`queryEvent` **刻意不实现**:运行时的 `vx_queue_finish` 每次都会入队一个屏障,拿它做 poll 会累积命令;本轮不许改 `sw/runtime`。基类抛 "Backend doesn't support querying streams." 好过给一个假答案,测试钉住了这个响亮拒绝。

## 3. W2.3 设备模块与退出

**退出 core dump 的真正原因**(计划书 F23 记的是「退出析构竞态」,没说到点上),两个独立成因:

1. **我们的注册活得比 libtorch 长**。`TORCH_LIBRARY_IMPL` 造出的静态 `RegistrationHandleRAII` 析构时会回调 Dispatcher 去注销,而本 `.so` 的静态对象在 libtorch 自己之后析构,那次调用落进已拆掉的 Dispatcher。backtrace 终点是 `RegistrationHandleRAII::~RegistrationHandleRAII` → `deregisterFallback_` → `computeDispatchTableEntry`。注册是进程生命周期的,泄漏句柄正是 torch 的本意(这也是 `torch::Library` 可堆构造的原因),所以改成堆上的 Library 永不销毁。allocator 同理:`c10::SetAllocator` 是非拥有指针,静态 allocator 先析构会让退出时的任何释放以 "pure virtual method called" 中止。

2. **模拟器线程还在跑**。第一层修掉后才看得见:simx 的 worker 线程还在 `Processor::run()` → `SimPlatform::tick()` 里,进程就在拆它要走的东西。没有任何地方调用 `hipDeviceReset`,所以 `vx_device_release`——以及随之而来的「等模拟器 future」的析构——从未发生。`load_ops` 现在注册 `atexit`:先同步、再 reset;atexit 早于静态析构,那时 runtime 还活着。

最小复现:**import、launch 一个 kernel、不读回就退出**。读回会掩盖它,这就是旧套件只在部分运行里 core dump 的原因。

顺带修的设备模块问题:`memory_allocated()`/`get_memory_info()` 会以 `device_allocator INTERNAL ASSERT FAILED ... is not a DeviceAllocator` 硬崩——torch 的 `getDeviceAllocator()` 要 dynamic_cast;allocator 现在派生 `c10::DeviceAllocator`。`get_device_capability()` 原本返回默认值,**宣称支持所有标量类型**;现在如实报 `{float32}`。`is_available` 曾是 `@property`,而 2.14 的 FakeTensor 把它当方法调(`TypeError`)。重导入曾直接抛错(reload 会重置模块全局,守卫必须钉在 torch 上)。

## 4. W2.4 同进程多模块

计划书把根因写成「编译/链接及公共 runtime」,**runtime 那一半不成立**:加载器按镜像头里的 `min_vma`/`max_vma` 各自 reserve(`module.cpp`),launch PC 取自 `kernel->module()->base_address()`(`queue.cpp`)——本来就是通用的。固定的只有 `ci/hipcc_vortex.py` 的一个常量。

动手前先实测:两个镜像在各自基址下**同进程加载成功,且两个 kernel 都执行正确**。

于是 hipcc 增加 `--image-base`/`--image-slot`,`sw/common/module_slots.mk` 是发号表。未在表中的镜像**构建失败**而不是静默回落到 slot 0——那个回落正是让冲突隐形的元凶。重复槽位在测试里另查(表是手写的)。

注意**不能**用追加 `-Wl,--defsym` 覆盖:hipcc 自己先加那份,先生效,覆盖是静默无效的。

## 5. 验证

全部在 `build_dl64`(XLEN=64, simx),解释器 `~/miniconda3/envs/vortex/bin/python`(py3.10 / torch 2.14)。

```
pytest torch-vortex/tests -q --tier=full   → 117 passed(4m14s,含模型门槛)
tests/dl/{blas,prim,dnn,rng}  run-simx     → 全部 PASS(它们的镜像换了基址)
tests/hip_native/*  run-simx               → 全部 PASS
```

新增测试都在**确认能抓住原 bug** 之后才留下:

- `test_memory.py`——延迟释放与快速路径,去掉延迟即失败。
- `test_streams.py`——「流上下文里的算子真的落在那个流上」,把 launch 改回 nullptr 即失败。
- `test_device.py`——`test_process_exits_cleanly_with_work_in_flight` 在子进程里测,去掉 teardown 钩子即 `returncode -11`。
- `test_modules.py`——把 rng 指回 slot 0 即 `hipErrorInvalidValue`。

## 6. 新发现,计划书未记

- **`hipFreeAsync` 每次调用都泄漏**:它把句柄交给队列(队列自己 retain 了一次),但从不释放调用方那一份,引用计数停在 1。一行之差。`tests/hip_native/stream_overlap` 原本只断言「重新分配到的指针非空」,所以泄漏能过 CI;现在断言地址真的被复用。**这个测试此前编译的是构建树里的陈旧副本**——那三个 `tests/hip_native/*/Makefile` 用的是相对 `main.cpp`,所以改源码等于没改;已改为像 `tests/hip/*` 那样用 `$(VORTEX_HOME)`。
- `tests/dl/*/Makefile` 有同样的相对路径问题,未改。
- `hipDeviceSynchronize`/`hipStreamSynchronize`/`hipStreamDestroy` 原本都是 `vx_queue_flush`(只唤醒 worker,不等)。**陷阱**:`vx_queue_finish(q, 0)` 不是「永久等待」,`Event::wait_value` 对 0 走 `wait_for(0ns)` 立刻返回超时,必须传 `VX_TIMEOUT_INFINITE`。
- `hipDeviceSynchronize` 只完成默认队列;现在登记所有分发的队列并全部完成(实测:显式流上的 launch 之后,设备级同步等 4.04s)。

## 7. 后续进展(本轮之后)

W3.2 的 elementwise 与跨步支持在同一分支上继续推进(elementwise op code 化、标量算子、一元数学、`as_strided` 与跨步拷贝、broadcasting、silu/gelu)。其中 **W1 里点名推迟的「跨步目标」已关闭**:`t.t().to("vortex")` 现在可用,跨步拷贝 kernel 同时服务跨步源、跨步目标和 storage offset。

一个未解释的偶发:`tests/test_factories.py::test_cpu_and_meta_factories_are_untouched` 在 3 次全量运行中失败 1 次,失败点是子进程返回码非 0(即子进程崩了)。隔离复现 45 次(25 次单测循环 + 20 次父子进程并发持设备)全部通过,已排除「父子进程同时打开 simx 设备」这一猜测。断言消息本身包含子进程 stderr,下次出现时是可诊断的。它测的是 W1.6 的工厂隔离,与 W2 无关。

## 8. 已知边界

- `queryStream`/`queryEvent` 不可用(需 `sw/runtime` 的非阻塞查询原语)。
- `getNewStream` 建的流无销毁钩子(该 torch 版本没有 `destroyStream`),随进程存活。
- 无 caching allocator:每次 `hipMalloc`,释放按流序延迟。计划书要求「性能优化前先证明生命周期正确」,顺序如此。
- 无 pinned memory、无 `hipMemGetInfo`(内存信息由我们的计数器拼出)。
- `tests/dl/*/Makefile` 的源码路径问题与 `sw/dl` 手填 `args_size` 未处理。
