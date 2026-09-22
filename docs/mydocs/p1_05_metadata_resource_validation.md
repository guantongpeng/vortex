# P1.3 metadata 资源约束提交前校验

## 目标

`VXKMDATA` 的参数大小已经在入队前校验，但只检查 `args_size` 仍会让一个明显超出 kernel 资源合同的 launch 进入队列。本节点把可由当前公共 ABI 表达的资源约束集中到 runtime：block 形状、静态/动态 local memory 和 ISA 要求必须在提交前满足设备能力。

## 实现

`sw/runtime/common/queue.cpp` 新增统一的 `validate_kernel_resources` 路径，单次 `vx_enqueue_launch` 和 `vx_enqueue_commands` 都调用它：

- `ndim > 0` 时，grid 与 block 每个有效维度都必须非零；block 乘积不能超过设备的 `NUM_THREADS × NUM_WARPS`，也不能溢出 32 位 KMU 字段。
- metadata 发布非零 `max_block[i]` 时，实际 block 维度不能超过相应上限。
- metadata 的 `required_isa` 必须是 `VX_CAPS_ISA_FLAGS` 的子集；缺失的标准或 Vortex 扩展直接返回 `VX_ERR_INVALID_VALUE`。
- `static_lmem_bytes + launch.lmem_size` 不能超过 `VX_CAPS_LOCAL_MEM_SIZE`。
- 所有失败都发生在 retain/复制参数或创建 completion event 之前；批量命令在保留 kernel 后若失败，会统一释放已保留的 kernel。

metadata 为零的字段继续表示“未声明该约束”，由现有设备默认 block 和运行时容量检查负责。`required_features`、XLEN、编译配置和 ABI `offsetof/alignof` 仍没有足够的公共描述，需要后续扩展 metadata 合同后再实现，不能把任意 bit 解释成当前 ISA。

## 测试

`tests/runtime/test_module_kernel.cpp` 的 synthetic VXKMDATA 回归新增：

- metadata `max_block[0] = 1` 而提交 block 2；
- metadata 静态 LMEM 等于设备总量而请求 1 字节动态 LMEM；
- metadata 要求当前设备没有的 ISA 扩展；
- 每个场景都断言返回 `VX_ERR_INVALID_VALUE` 且不产生 event。

其中 block 上限同时通过单次 `vx_enqueue_launch` 和批量 `vx_enqueue_commands` 验证，防止两条提交路径重新分叉。

原有参数大小 mismatch/4 KiB limit 测试继续覆盖 args 合同。所有合成镜像只用于提交前错误路径，不执行真实 kernel。

## 验证记录

RV64 `build_dl64`：

```text
env -u DEBUG CCACHE_DISABLE=1 make -s -C sw/runtime stub
env -u DEBUG CCACHE_DISABLE=1 make -s -C tests/runtime run-simx
```

`test_module_kernel`、runtime 全套测试均通过。后续若修改 VXKMDATA 记录布局，必须同时更新 parser、公共 `vx_kernel_info_t`、单次/批量提交和这些 synthetic metadata 测试。

## 未完成项

- 让编译器/sidecar 发布并验证 XLEN、ABI 对齐、配置签名、TCU/DXA capability 和 `required_features`；
- 为 metadata 约束增加 RV32/rv64 镜像交叉解析测试；
- 将同一资源校验接入 graphics draw descriptor 及其它直接构造 launch descriptor 的入口；
- 评估寄存器数量、occupancy 和多 CTA cluster 约束，形成稳定的公共字段后再加 hard check。
