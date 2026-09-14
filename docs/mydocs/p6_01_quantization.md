# P6.1 量化:W4A16 与 INT8 W8A8

## 目标

完成计划 P6-01:W4A16 weight-only 与 INT8 W8A8 的 pack/dequant/matmul 与精度报告(计划 Q1+Q2 的 kernel 侧)。FP8/MX/NVFP4/2:4 稀疏(Q3-Q5)按计划需先有 TCU dtype capability 与独立 golden,不在本节点。

## 交付物

- [`sw/dl/include/vortex/quant.h`](../../sw/dl/include/vortex/quant.h):`vx_quant_init/pack_w4/unpack_w4/gemm_w4a16/gemm_w8a8/finalize`。
- [`sw/dl/src/quant_args.h`](../../sw/dl/src/quant_args.h):参数块与**元数据约定文档化**——W4 为 signed nibble(低半字节=偶数 k,单字节两值)、per-(行,K 组)FP32 scale = amax/7(全零组=1.0);W8A8 为 int8 激活(per-tensor)+ int8 权重(per 行)。
- [`sw/dl/src/quant_kernels.hip`](../../sw/dl/src/quant_kernels.hip):
  - `quant_scales4_kernel`/`quant_pack4_kernel`:两遍打包(**单线程独占整字节**——初版对 nibble 做跨线程读改写,实测丢失 1531/1650 字节,已修复并在源码注明);
  - `quant_unpack4_kernel`:解包回 FP32;
  - `quant_gemm_w4a16_kernel`:16×16 tile/4-warp CTA,**解包在 LMEM staging 阶段在线完成**(nibble×scale→FP32),内积与 FP32 GEMM 相同——即计划 Q1 的 "unpack + FPU FP16 accumulate" 形状;
  - `quant_gemm_w8a8_kernel`:int8 输入 int32 累加,epilogue 一次乘 act_scale×w_scale。
- [`tests/dl/quant/`](../../tests/dl/quant/):CPU reference 以 **float 逐位模拟**(相同运算顺序)支撑 bit-exact 断言;dequant 用按组绝对误差界(|Δ|≤scale,单元素相对误差对 |q|=1 数学上无界——测试注释说明);含零权重组边界。

## 验证记录(2026-09-14)

| 检查 | simx rv64 | rtlsim rv64 | simx rv32 |
|---|---|---|---|
| pack_w4 nibble/scale | **bit-exact**(0/1650, 0/132) | 同 | 同 |
| unpack_w4 | bit-exact + abs 界内(mean 3.25e-2) | 同 | 同 |
| gemm_w4a16(M40×N33×K100,g32) | max_rel=6.33e-05 | 同 | 同 |
| gemm_w8a8 | max_rel=1.09e-07 | 同 | 同 |

(矩阵含非整 tile 维度与 K=100 非整组尾部。)

## 计划要求的其余检查(本节点结论)

- "pack→unpack bit-exact" ✓;reference dequant 误差 ✓(绝对界);GEMM 误差 ✓。
- 奇数组尾(K%group≠0)、零组(scale=1 分支)✓;饱和 clamp(±7/±8)在 pack 内建 ✓。
- 端到端模型精度(top-1/mAP)与性能报告:需要 P7 模型 harness,后续节点。
- 与 torchao 对接:需 P5 Tier A 算子扩展后以自定义 op 暴露,不在本节点。

## 后续(Q3-Q5 前置)

FP8/INT4 TCU/MX/NVFP4/2:4 稀疏全部依赖 `vortex2.h` 的 TCU dtype capability ID(尚不存在)与 TCU 配置构建树;先补 capability 位再开独立 golden。
