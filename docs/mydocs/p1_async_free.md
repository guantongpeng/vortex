# P1.1 queue-ordered free 开发记录

## 目标

为 HIP `hipFreeAsync` 和 PyTorch caching allocator 提供一个不会提前释放设备缓冲区的 Vortex 原语。调用者提交 free 后可以立即释放自己的 handle；runtime 保留一份引用，直到 queue FIFO 和显式 wait-list 都完成，才执行最终的 `Buffer` 析构和 `Device::mem_free`。

## 实现

- [`sw/runtime/include/vortex2.h`](../../sw/runtime/include/vortex2.h) 新增 `vx_enqueue_free`。
- [`sw/runtime/common/vortex2_internal.h`](../../sw/runtime/common/vortex2_internal.h) 和 [`queue.cpp`](../../sw/runtime/common/queue.cpp) 将 free 表达为一个普通 `Queue::Command`，复用现有事件、依赖和 worker 生命周期。
- [`tests/runtime/test_async_free.cpp`](../../tests/runtime/test_async_free.cpp) 在 worker 运行前释放调用者引用，验证 retained reference 能让异步 free 正常完成。
- [`tests/runtime/Makefile`](../../tests/runtime/Makefile) 将该测试加入 SimX/XRT runtime 测试集合。

该节点不实现内存 caching pool、`hipMallocAsync` 或跨设备 peer memory；这些需要 allocator policy 和更多 capability 字段，将在后续 P1 节点完成。

## 语义

`vx_enqueue_free(q, buf, waits, out_event)` 的释放点满足：

```
q 中此前已入队命令完成
并且 waits 中每个 event >= 1
             ↓
       Buffer::release()
             ↓
       最后引用才 Device::mem_free
```

如果 wait-list 非法或 command 入队失败，保留引用立即回滚，不泄漏 buffer。free command 本身返回一个普通 completion event，失败状态通过该 event 传播。

## 验证

修改 runtime 源码或 Makefile 后，从独立 build 目录重新 configure：

```bash
cd build_dl64
../configure --xlen=64 --tooldir=/data/vortex-tools
make -C sw/runtime/stub
make -C tests/runtime test_async_free
VORTEX_DRIVER=simx LD_LIBRARY_PATH=$PWD/sw/runtime tests/runtime/test_async_free
```

源码和链接验证：

```bash
cd /home/guantp/aichip/vortex
g++ -std=c++17 -Wall -Wextra -Werror -fsyntax-only \
  -Isw/runtime/include tests/runtime/test_async_free.cpp
```

完整 SimX 运行还依赖 Ramulator 的 `yaml-cpp`，当前环境的外部下载不可达；因此提交时应分别记录“host 编译通过”和“后端运行待依赖解除”，不能把未运行标成 PASS。运行恢复后必须再测：同 queue FIFO、跨 queue wait-list、free 后地址不可重用到旧命令、超时/错误传播。
