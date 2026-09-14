# P7.1 模型验证报告(第一份:ResNet 族)

按计划 §11 的验收框架(算子覆盖表、CPU fallback 次数、最大内存、数值),首份模型级报告来自 P5-02 的 MiniResNet 端到端验证。后续模型(YOLO/SAM/DINOv3/Mamba/LLM/VLM)依赖 conv/attention/SSM 算子扩展与 torch.compile 路径,状态在文末如实列出。

## 模型 1:MiniResNet(ResNet 族结构验证)

**结构**(`torch-vortex/tests/test_resnet.py`):stem(conv3x3+BN+ReLU+MaxPool2)→ BasicBlock×1(残差)→ BasicBlock(stride 2 下采样 + 1×1 短路)→ 全局平均池化 → Flatten → Linear。通道 8/16,输入 1×3×16×16,分类数 4 —— ResNet 的全部结构要素(conv/bn/relu/maxpool/残差加/下采样短路/global pool/fc)。

**配置**:SimX 后端(rv64,默认 1 core/4 warps/4 lanes),固定 seed 权重,CPU 同权重模型为参考。

| 验收项 | 结果 |
|---|---|
| 前向执行 | **100% 设备算子**(未注册 op 会 NotImplementedError;测试断言全链无异常) |
| CPU fallback 次数 | **0** |
| 数值 vs CPU | logits `[-0.0451, -0.0726, 0.1493, 0.1896]`,逐元素差 ≤ 1e-7(rtol 1e-3 门限,实际富余 4 个量级) |
| 算子覆盖 | convolution / native_batch_norm / relu / max_pool2d / add / adaptive_avg_pool2d / view / linear(= mm+transb+bias) |
| 前向时长 | ~90 s(SimX 功能仿真;1 core @ 4 lanes;性能非本节点目标) |
| 设备内存 | ≈1.7 MB(激活 + 权重;hipMalloc 直通 vx_buffer_create) |
| 测试 | pytest 9/9(含 2 项 ResNet) |

**结论**:ResNet 族网络在 Vortex 软件栈上的 eager 推理路径**端到端成立**;数值正确性与算子纪律(零静默回退)已验证。

## 后续模型的状态与阻塞(如实)

| 模型 | 状态 | 阻塞 |
|---|---|---|
| ResNet-18 完整版 | 结构要素已覆盖 | simx 时长(224² 输入);rtlsim/FPGA 性能路径 |
| YOLO | 未开始 | 检测头/NMS/动态 batch 算子 |
| SAM/DINOv3 | 未开始 | ViT 长序列 attention(P3 attention 层) |
| Mamba | 未开始 | selective scan 专用 kernel |
| Llama/Qwen | 未开始 | RMSNorm✓/RoPE/GQA/KV cache;embedding/gather |
| VLM | 未开始 | 依赖上述全部 |

共同前置:P4 codegen(compile 路径)、TCU dtype 构建树(低精度性能)、FPGA/XRT(P7-02)。
