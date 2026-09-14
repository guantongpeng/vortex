# P3.1 BLAS dispatch 与 FP16/BF16 GEMM 种子

## 目标

建立计划 §7.1 的 `sw/dl` 结构,交付第一个算子库节点: tiled GEMM 的主机 dispatch（`libvortex_dl.so`）+ FP32/FP16/BF16 三种 dtype 的 KMU kernel 镜像,并按 §7.2 要求用 CPU double reference 验收。

## 交付物

- [`sw/dl/include/vortex/dtypes.h`](../../sw/dl/include/vortex/dtypes.h): `vx_fp16_t`/`vx_bf16_t` 存储类型 + 位级转换(fp16→f32 精确;f32→fp16 RNE;bf16 截断)。**不用 `_Float16`/compiler-rt half 辅助函数——它们在当前 simx 上挂起**(实测),自实现与 GPU half API 同构(存储类型 + 显式转换)。
- [`sw/dl/include/vortex/blas.h`](../../sw/dl/include/vortex/blas.h): 主机 C API(`vx_blas_init/finalize/gemm/kernel_name`),只依赖 `vortex2.h`。
- [`sw/dl/src/blas_args.h`](../../sw/dl/src/blas_args.h): kernel 参数块共享布局,指针按设备宽度(device 用 `uintptr_t`,host 用构建树 `VX_CFG_XLEN` 决定的 `uint32_t/uint64_t`)——P2-02 发现的 RV32 宽度问题的制度化。
- [`sw/dl/src/blas_kernels.hip`](../../sw/dl/src/blas_kernels.hip): 模板化 tiled GEMM(16×16 tile,16 线程 = 4 warp CTA,K 以 8 分块,LMEM staging + `__syncthreads`——KMU 路径的多 warp CTA 形态)。FP16/BF16 在加载时转换、FP32 累加(与 W4A16/TCU 路径相同的正确性形状);非整 tile 维度零填充 + 写回保护。
- [`sw/dl/src/blas_host.cpp`](../../sw/dl/src/blas_host.cpp): `libvortex_dl.so`——module 按名解析 kernel(`__vx_kentry_*`),组装 `vx_launch_info_t` 提交;`vx_blas_kernel_name` 报告 dispatch 变体(未来 TCU vs fallback 的 perf 归因入口)。
- [`tests/dl/blas/`](../../tests/dl/blas/): M=48/N=20/K=13(非整 tile),alpha=1.5/beta=0.5(β 路径),固定 seed;低精度 dtype 的 reference 用**同量化输入**计算(隔离 kernel 正确性与输入量化)。
- `ci/hipcc_vortex.py`: 设备编译加入 `-I sw/dl/include`。

## 验证记录(2026-09-14)

| 后端 | 结果 |
|---|---|
| simx rv64 | f32 max_rel=2.33e-05、f16 8.34e-06、bf16 2.09e-07,errors=0,PASSED |
| rtlsim rv64 | 同上数值一致(周期级模型 parity),PASSED |
| simx rv32 | 同上,PASSED(设备宽度参数块 36B 验证) |

## 两个新发现(均已写入源码注释)

1. **VOLT -O3 miscompile:缓存 kernel 参数块字段到局部变量产生错误结果**。`const uint32_t M = arg->M, K = arg->K; ...` 后所有 CTA 结果错误(值接近但不精确,几乎全部元素超差);直接读 `arg->M/arg->K` 则 bit 级正确。二分定位:同一 kernel 源码仅去掉缓存语句即通过。疑似加载被提升/重排出 staged 参数块的生命周期。**workaround 已注明,待向 VOLT 报 bug 并回归**。
2. **`vx_launch_info_t` 未用维度必须填 1 而非 0**:block_dim[1..2]=0 时 KMU 推导的 CTA 形状坍缩,kernel 不写回任何结果(数值保持 C 初值)。`vx_launch_info_t` 文档建议补充此约束。

## 与计划的差距(如实)

- TCU kernel 变体未实现:`vortex2.h` 尚无 TCU dtype capability ID(P0.2 预留),dispatch 先返回 FPU 变体名;`sgemm_tcu_wg` 等 regression 测试是现成的 TCU 数值基线,TCU 接入需要独立的 A+TCU 配置构建树。
- batched GEMM/matmul epilogue/conv 等第二层算子属后续节点。

## 后续

P3-02 将在同一 harness 下增加 prim/norm/activation kernel;`tests/dl` 的 CPU reference 代码会抽成共享的 `ref.h`。
