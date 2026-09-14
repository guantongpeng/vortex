# P3.3/P6.2/P4.2 工作流批次:attention/LLM/Mamba/RNG + 量化收尾 + Triton codegen

## 方法

9 个模块经并行 agent 实现(每模块独立 scratch、独立 SimX 验证)→ 每模块独立对抗审计(重跑测试 + 数学/ABI/容差/边界四维核查)→ 审计修复后由主线集成进 `sw/dl`。审计层抓到了实现层 5 处问题(attn rv32 尺寸、llm u32 溢出、nvfp4/rng 的 arg 缓存违规、以及**已提交代码 `quant_fp8.h` 的两个真实编码 bug**)——后者的发现与修复见独立提交。

## 交付与验证(全部 simx;attn/mamba 加验 rtlsim)

| 模块 | 关键结果 |
|---|---|
| attention(SDP,full+causal) | 奇数 L=33、H4/L64 大形状 max_rel≤4e-06;rv32 args_size 审计修正为 40 |
| llm(RoPE/SwiGLU/embedding/KV-append) | 全部 bit/1e-5 级;kv_append 溢出检查升级 u64 |
| mamba selective scan | B2C5T32N8 与长 T 两种形态 max_abs≤8e-07;状态寄存器展开限 d_state≤8(已注明生产版需 LMEM 分块) |
| Philox 4x32-10 RNG | 前 16 输出与独立 host 实现精确一致;mean/min/max 统计界内 |
| MXFP8(E8M0 组尺度+e4m3) | pack bit-exact;可证界 \|err\|≤16s(E4M3 半 ULP 数学上限,文档论证);GEMM 8.5e-08 |
| NVFP4(E2M1+e4m3 组尺度) | 双 tensor_scale 档 pack bit-exact;GEMM 9.8e-07;arg 缓存违规已修 |
| 2:4 sparse | 剪枝 bit-exact(确定性平局);**毒化重跑证明 GEMM 内循环真读 metadata**;稀疏率恰 50% |
| Triton TTIR→C→vxbin | 官方 vecadd 形状 **max_abs=0.000e+00**;rowmax+完整 softmax(reduce 类)1.5e-08;fp16 指针/2-D broadcast/tl.dot 响亮 NotImplementedError(负例测试) |

`libvortex_dl.so` 现含 11 个 host 层;`tests/dl` 11 套件全绿,旧 4 套无回归。

## 已知问题(如实)

- pytest 单文件路径的**退出时**析构竞态(simx 后端清理;测试全过、组合运行干净退出)—— 记录于 p5_02,后续在 simx 退出路径修。
- 同进程多设备模块受 simx 固定基址限制(torch_ops+triton vxbin 不能共存于一进程)—— e2e 脚本按 kernel 分进程,已有记录。
