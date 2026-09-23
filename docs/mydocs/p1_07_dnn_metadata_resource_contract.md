# P1.1：DL kernel 资源元数据与 host launch 合同

实施日期：2026-09-23。验证基线为 `build_dl64`、XLEN=64、SimX、Python
3.10.21、PyTorch 2.14.0。

## 问题

runtime 现在会在 `vx_enqueue_launch` 前检查 kernel metadata 的 `args_size`、
`max_block` 和 LMEM 约束。DL host 的 `launch3` 固定使用 `block_dim[0]=16`。
卷积和 pool 元数据已经声明 `[16, 1, 1]`，但 BN 记录只声明了 `args_size`。
VOLT/加载器对缺失字段提供了保守的默认 block 上限 `[4, 4, 1]`，因此 BN 实际
提交会在队列入环前返回 `VX_ERR_INVALID_VALUE`。

## 修复

`sw/dl/Makefile` 的 `DNN_META` 为 `dnn_bn_affine_kernel` 增加
`"max_block": [16, 1, 1]`，使三条 DNN kernel 的 metadata 与 `dnn_host.cpp`
统一。这个字段不是放宽 runtime 检查：它记录 host 真实提交的资源合同，仍会
拒绝更大的 block。

## 验证

修改 Makefile 后先从 build 目录重新执行 `../configure`，再清理并重建 DL image，
避免 build tree 中的旧 Makefile 和旧 `dnn_meta.json` 掩盖修改：

```text
../configure --xlen=64 --tooldir=/data/vortex-tools                         PASS
make -s -C sw/dl clean                                                       PASS
CCACHE_DISABLE=1 make -s -C sw/dl                                             PASS
CCACHE_DISABLE=1 python -m pytest torch-vortex/tests/test_batch_norm.py -q     PASS (9 passed)
```

修复前，BN 的 7 个数值/稳态测试均在提交阶段失败；修复后 BN 的 9 个测试全部
执行到设备并通过。卷积、pool 和 BLAS 回归也保持通过。

