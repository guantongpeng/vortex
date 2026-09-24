# 阶段 2.3：dropout inference 语义

`aten::dropout` 在 `train=False` 时是 identity；bridge 直接返回原 Vortex Tensor，
不分配、不复制、不发射 kernel，并检查 `p∈[0,1]`。`aten::native_dropout` 同样
返回输入和全 1 的 device bool mask，保证 eval 模块能继续走标准 schema。

训练路径通过 `native_dropout` 明确报错，避免伪造随机数、mask 或梯度。新增测试
验证 eval 的 `data_ptr`/launch 计数和 training refusal；正式主线结果为 **2 passed**。
Philox Generator、seed/offset 和训练 dropout 仍属于阶段 2.5/RNG 节点。
