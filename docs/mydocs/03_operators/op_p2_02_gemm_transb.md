# P2.2：GEMM 的 `transb` 直读路径

实施日期：2026-09-23。验证基线为 `build_dl64`、XLEN=64、SimX、Python
3.10.21、PyTorch 2.14.0。

## 变更

BLAS 参数结构原本最后一个字段只是 padding。现在它是 `transb`：

- `0`：B 按 `[K][N]` 读取；
- `1`：B 按 `[N][K]` 读取，相当于逻辑上的 `B^T`。

`gemm_tile_body` 在加载 B tile 时按该字段选择地址公式，FP32、FP16 和 BF16
三条 kernel 共用同一行为。`vx_blas_gemm_ex` 暴露带 `transb` 的 API，原有
`vx_blas_gemm` 保留为 `transb=0` 的兼容入口，因此其他 DL 调用方无需改变。

PyTorch `mm_launch` 不再构造 `b_in.t().contiguous()` 临时 Tensor，直接把
linear 的 `[out_features, in_features]` weight 和 `transb=1` 传给 BLAS。shape、
dtype、连续性检查仍在 host 侧完成，GEMM 的 alpha/beta epilogue 语义不变。

## 验收

`test_matmul.py` 的现有矩形、非方阵、partial tile、linear、addmm 和 bmm 用例
继续通过；新增 `test_linear_transb_does_not_materialise_weight`，清零统计后
确认 linear 只有 zeros 初始化、GEMM、bias 三次设备 launch。若先转置再连续化，
会多出一次 copy kernel launch，测试会失败。

```text
../configure --xlen=64 --tooldir=/data/vortex-tools                         PASS
CCACHE_DISABLE=1 make -s -C sw/dl blas.vxbin libvortex_dl.so                  PASS
CCACHE_DISABLE=1 make -s -C torch-vortex/kernels                              PASS
CCACHE_DISABLE=1 python -m pytest torch-vortex/tests/test_matmul.py -q        PASS (26 passed)
```

## 边界

本节点只消除转置 B 的临时拷贝；`bmm` 仍按 batch 循环提交单个 GEMM，真正的
batch-stride/grid-z kernel 是后续节点。非连续 weight 仍按既有 matmul 合同拒绝，
不会暗中复制成连续布局。

