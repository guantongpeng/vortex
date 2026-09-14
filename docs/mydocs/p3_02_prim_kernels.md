# P3.2 prim/norm kernel 与统一 reference harness

## 目标

交付计划 §7.2 第一层算子的核心子集:activation、reduction/argmax、softmax、layernorm/rmsnorm,并建立所有 `tests/dl` 共享的 double-precision CPU reference harness。

## 交付物

- [`sw/dl/include/vortex/prim.h`](../../sw/dl/include/vortex/prim.h):主机 C API(`vx_prim_init/finalize/unary/reduce/softmax/layernorm/rmsnorm`)。
- [`sw/dl/src/prim_args.h`](../../sw/dl/src/prim_args.h):每算子参数块(设备宽度指针,同 blas 约定)。
- [`sw/dl/src/prim_kernels.hip`](../../sw/dl/src/prim_kernels.hip):KMU 镜像,5 个 kernel:
  - `prim_unary_kernel`:ReLU/GELU(tanh 近似)/SiLU,grid-stride 单 warp CTA(P2-03 约束);
  - `prim_reduce_kernel`:sum/max/argmax,单 16 线程多 warp CTA + LMEM 分块归约;argmax 以 (value, index) 对 shuffle,平局取最小索引(测试显式验证);
  - `prim_softmax_kernel`:每 CTA 一行,三遍(max→exp+sum→scale),LMEM 广播槽;
  - `prim_layernorm_kernel`/`prim_rmsnorm_kernel`:每 CTA 一行,warp 归约 + LMEM 广播,LN 用 E[x²]−mean² 方差。
- [`sw/dl/src/prim_host.cpp`](../../sw/dl/src/prim_host.cpp):dispatch(与 blas 同一 `libvortex_dl.so`);未用维度一律填 1(P3-01 发现)。
- [`tests/dl/ref.h`](../../tests/dl/ref.h):共享 CPU reference(gelu/silu/softmax/layernorm/rmsnorm,double 精度)——P3-02 起所有 dl 测试复用。
- [`tests/dl/prim/`](../../tests/dl/prim/):777 元素(非整块)、softmax 65 列、LN/RMS 33 列、大值边界、argmax 平局(500 与 123 同值取 123)。

## 验证记录(2026-09-14)

| 算子 | simx rv64 | rtlsim rv64 | simx rv32 |
|---|---|---|---|
| relu | 0.00e+00 | 同 | 同 |
| gelu | 7.67e-07 | 同 | 同 |
| silu | 1.25e-07 | 同 | 同 |
| reduce_sum/max | 0.00e+00 | 同 | 同 |
| argmax(平局) | 123 ✓ | 同 | 同 |
| softmax | 1.53e-07 | 同 | 同 |
| layernorm | 8.18e-07 | 同 | 同 |
| rmsnorm | 1.40e-07 | 同 | 同 |

三后端/双 XLEN 数值完全一致(errors=0)。

## 未覆盖(后续节点)

第一层其余算子:cast 矩阵形态、embedding、transpose/reshape/concat/slice、Philox RNG;第二层 conv/pooling/batchnorm;attention 族(P3 后续)。`__expf/__logf` 快速近似入口已在 hip-vortex 头文件,但本节点用 libm 精确版本(近似版需 MPM 精度对比后再切换)。
