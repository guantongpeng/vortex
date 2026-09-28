# 阶段 2.1：factory、Meta 和 FakeTensor 隔离

## 合同

PrivateUse1 factory 只处理 `device="vortex"`：CPU、Meta、其他 dtype 和
`empty_strided` 仍由 ATen 自己的 dispatch 路径处理，不再通过 BackendSelect
全局替换。Vortex factory 负责创建设备 storage，但不会修改 CPU/Meta 的
device、stride 或数值行为。

FakeTensorMode 下，`torch.empty(..., device="vortex")` 和 view 只构造 fake
metadata：不会调用 `hipMalloc`、不会加载真实 device buffer，也不会发射
kernel。这样后续 Meta/FakeTensor shape propagation 可以独立于 SimX 运行，
而真实 eager tensor 仍走已有的 allocator 和 queue 语义。

## 验证

`test_factories.py` 在 build tree 的 torch-vortex 工作目录执行为 **7 passed**，
包括导入前后 CPU/Meta factory probe、Vortex empty/zeros、FakeTensor 零分配零
launch、非法 device index、未实现算子拒绝和 inference-only backward 边界。

## 边界

FakeTensor 目前只承诺 factory/view 的 metadata 路径；未实现 ATen schema 仍
通过 backend 的明确拒绝报告，不能把 FakeTensor 通过误认为真实 kernel 已支持。
