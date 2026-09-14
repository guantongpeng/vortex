# P5.2 ResNet eager on device 与 P7.1 首份模型报告

## 目标

完成计划 P5-02:ResNet 族网络在 vortex 设备上 eager 执行 —— 不只是单算子,而是完整前向(stem/残差块/下采样/全局池化/分类头),数值与 CPU 一致。这同时构成 P7-01 的第一份模型级验证报告。

## 交付物

- torch-vortex op 集扩展(全部设备 kernel,经 aten dispatch):
  - `convolution`(NCHW、dilation 1、groups 1)、`native_batch_norm`(推理,host 预计算 rstd)、`max_pool2d`、`adaptive_avg_pool2d`((1,1))、`mm`(16×16 tile + transb)、`linear`(x@Wᵀ+bias)、`addmm`(bias 向量形式)、`relu`、`view`(连续张量 metadata 重排,Flatten 所需)。
  - `kernels/dnn.hip`:对应 KMU kernel(conv 每 CTA 一行 + LMEM 权重 staging;pool 同形;bn 逐元素;mm 与 sw/dl 同 tile 结构 + transb;bias 行广播加)。
  - **ops 与 dnn 合并为单镜像 `torch_all.vxbin`**:simx 模块固定加载基址 0x80000000,双镜像地址重叠(实测第二模块加载失败)——hipcc 链接模式直接吃多个 .o。
- [`tests/test_resnet.py`](../../torch-vortex/tests/test_resnet.py):MiniResNet(ch=8)全前向,CPU 同权重参考。

## 验证记录(2026-09-15,SimX)

```
$ pytest tests/ -q
9 passed                       # 7 基础 + 2 ResNet
mini-resnet logits device: [[-0.0451, -0.0726, 0.1493, 0.1896]]
mini-resnet logits cpu   : [[-0.0451, -0.0726, 0.1493, 0.1896]]
(逐元素一致到 ~1e-7;整套 pytest 82s,其中 ResNet 前向 ~90s@1x3x16x16)
```

模型级结论(ResNet 报告条目):**算子覆盖** conv2d/bn/relu/maxpool/add/avgpool/linear/view 全设备、**CPU fallback 次数 0**(未注册 op 会直接 NotImplementedError,响亮失败)、**最大内存** ~1.7MB 设备缓冲(1x3x16x16 激活 + 权重)。

## torch dispatch 新踩坑(追加 P5-01 清单)

8. composite 层把 `bias=None` 作为 **undefined tensor 装进 optional**(仍 `has_value`)—— 必须检查 `defined()`。
9. `aten::max_pool2d` 参数是 `int[2]`(非 symint);`native_batch_norm` 的 float schema 对应 C++ **double**;optional 参数 canonical 签名带 `const&`;`addmm` 标量是 `c10::Scalar const&`。
10. Autograd 键需 namespace 级 fallthrough:`TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m)`(单 namespace 的 fallback 不被允许)。
11. **simx 模块固定加载基址** —— 多镜像需合并链接(见上)。
12. VXKMDATA `args_size` 必须等于真实 C struct sizeof(rv64 的 8 字节对齐会放大尺寸:mm 实为 40 非 32、conv 104 非 96;错尺寸使尾部字段读到垃圾,表现为数值错或挂起)。

## 未覆盖(如实)

- `torch.compile`/Inductor 路径:依赖 P4 codegen(开放);`torch.export` 图导出可在 CPU 侧做,设备执行仍走 eager。
- torchvision resnet18 全模型:op 集已大体覆盖但需要 7x7 stem conv(LMEM 界内)、ceil_mode 池化等边角 + simx 时长不可行(224² 输入)。
- fp16 推理、训练、动态 batch:后续节点。
