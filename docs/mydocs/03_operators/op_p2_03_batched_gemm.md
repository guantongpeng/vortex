# P2.2：grid-z batch-stride GEMM

实施日期：2026-09-23。验证基线为 `build_dl64`、XLEN=64、SimX、Python
3.10.21、PyTorch 2.14.0。

## 变更

BLAS GEMM 参数块扩展为 batch metadata：`stride_a`、`stride_b`、`stride_c`
使用字节距离，`batch` 指定 grid-z 的 batch 数。kernel 用 `blockIdx.z` 计算
三个矩阵的基址，因此每个 CTA 仍负责一个 16×16 tile，但所有 batch 在同一个
queue launch 中完成。普通 `vx_blas_gemm_ex` 通过 `batch=1` 和零 stride 保持
原语义。

新增 `vx_blas_batched_gemm`，PyTorch `bmm` 使用输入/输出的真实
`stride(0) * sizeof(float)`，不再在 host 侧循环调用 batch 次单矩阵 GEMM。
`beta=0` 仍保证不读取未初始化输出；空 batch、M/N/K=0 继续在 host 侧跳过或
返回零结果，不提交无效 grid。

## 验收

`test_bmm_queues_one_batched_gemm` 把原先的“每 batch 一次 launch”回归改为严格
检查一次 launch；`test_matmul_3d_reaches_bmm` 也检查一次 launch。矩形 batch、
奇数 tile、零 batch、K=0、非连续输入拒绝和所有 mm/addmm/linear 用例保持覆盖。

```text
../configure --xlen=64 --tooldir=/data/vortex-tools                         PASS
CCACHE_DISABLE=1 make -s -C sw/dl blas.vxbin libvortex_dl.so                  PASS
CCACHE_DISABLE=1 make -s -C torch-vortex/kernels                              PASS
CCACHE_DISABLE=1 python -m pytest torch-vortex/tests/test_matmul.py -q        PASS (26 passed)
```

## ABI 说明与边界

RV64 参数块从 48 bytes 扩展为 80 bytes，`blas_meta.json` 与 host/device 共享
结构同步更新；RV32 的 host layout 记录为 72 bytes，后续 RV32 parity 会单独
复核设备编译器的 `uint64_t` 对齐。当前 bmm 仍要求连续 FP32，broadcast batch
和低精度 batched GEMM 尚未接入；这避免用错误 stride 冒充完整布局支持。

