# P1.4 launch 参数大小与 metadata 提交前校验

## 目标

把 kernel 参数块从“运行时尽量分配并复制”改成明确的 ABI 合同：编译器 metadata 发布非零 `args_size` 时，主机必须提供完全相同大小的参数块；任何超过固定 scratch 上限的参数必须在入队前失败。错误不能延迟到设备执行，也不能通过静默截断或隐藏的备用分配路径继续运行。

## 实现

- `VX_KERNEL_ARGS_MAX_BYTES` 进入公共 `vortex2.h`，主机 launcher、runtime 和测试共享同一个 4096 字节上限。
- `vx_enqueue_launch` 在复制参数和创建队列命令前检查参数大小及 kernel 的 `vx_kernel_info_t.args_size`。
- `vx_enqueue_commands` 在构建批量命令记录时执行同样的检查，避免单次提交和批量提交的语义分叉。
- `Device::args_slot_acquire` 拒绝超过上限的请求；不再为超限参数创建一次性 device allocation。
- metadata 为旧镜像缺失或 `args_size=0` 时保留兼容路径；有非零 metadata 时不允许传入 0 或其他大小。

## 验证

在 `build_dl64` 重新执行 `configure` 后，关闭不可写 ccache 并构建 SimX runtime 和 runtime 测试：

```text
env -u DEBUG CCACHE_DISABLE=1 make -s -C sw/runtime simx
env -u DEBUG CCACHE_DISABLE=1 make -s -C tests/runtime test_module_kernel
VORTEX_DRIVER=simx LD_LIBRARY_PATH=$PWD/sw/runtime \
  VX_TEST_VXBIN=$PWD/tests/regression/basic/kernel.vxbin \
  ./tests/runtime/test_module_kernel
```

新增的 `test_module_kernel` 用例验证：

1. metadata 声明 64 字节而主机传 32 字节时，`vx_enqueue_launch` 立即返回 `VX_ERR_INVALID_VALUE`；
2. 主机传入 4097 字节时立即返回 `VX_ERR_INVALID_VALUE`；
3. 两种失败路径都不创建 completion event，也不启动设备 kernel；
4. 无 metadata 的旧 `kernel.vxbin` 仍能完成原有 launch。

## 边界和后续

本节点只完成 `args_size` 和参数上限。`alignof/offsetof` 单一来源、XLEN/ISA/TCU/DXA、block/LMEM 资源校验以及编译器 sidecar 自动生成仍属于 P1.3 后续节点。超过 4 KiB 的参数目前是明确拒绝；如果后续需要大参数，应先提交 ABI/proposal，不能重新引入静默的一次性分配。
