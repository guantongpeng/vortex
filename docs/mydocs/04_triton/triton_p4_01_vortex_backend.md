# P4.1 triton-vortex:out-of-tree Triton backend(v0.1)

## 目标

按计划 §8.1 建立 `triton-vortex/` out-of-tree backend,交付 P4-01 的可验证第一里程碑:**backend 注册、driver 层设备执行、interpreter 参考数值**,并把 llir→vxbin codegen 路线文档化(明确未完成,不冒充)。

## 环境

- Triton 3.8.0 + Python 3.10(用户级 miniconda,`~/miniconda3/envs/vortex`)—— 系统 Python 3.8 装不了现代 Triton;Ubuntu 20.04 的 `python3.10` 是 qgis 附带包,Anaconda 默认频道要交互式接受 ToS,最终用 **conda-forge** 频道建环境。
- **Triton 3.8 的 out-of-tree 发现机制是 `triton.backends` entry points**(不再是旧的 `TRITON_PLUGIN_DIRS` 目录扫描)——计划文档该处需随版本更新。

## 交付物

- [`triton-vortex/pyproject.toml`](../../../triton-vortex/pyproject.toml):pip 包 `triton-vortex`,entry point `vortex = "triton_vortex"`。
- [`triton-vortex/triton_vortex/driver.py`](../../../triton-vortex/triton_vortex/driver.py):`VortexDriver(DriverBase)` —— `GPUTarget("vortex", arch=64, warp_size=4)`(warp 尺寸如实为 4,不写 32)、benchmarker、设备/流占位;`VortexUtils.load_binary`(module+function 解析)。
- [`triton-vortex/triton_vortex/hip.py`](../../../triton-vortex/triton_vortex/hip.py):libhip_vortex 的 ctypes 完整绑定(init/malloc/free/memcpy/module/launch/sync),**`pack_args` 按设备宽度打包参数块**(自然对齐,rv32 4 字节指针);构建树自动发现(`VORTEX_BUILD` > 祖先 > 仓库根下 build*,优先 rv64)。
- [`triton-vortex/triton_vortex/compiler.py`](../../../triton-vortex/triton_vortex/compiler.py):`VortexBackend(BaseBackend)` —— 注册 ttir/ttgir/llir 通用阶段 + `bin` 阶段**显式 `NotImplementedError`**(指明缺失件:program-id→CSR lowering、VOLT `-x ir` 链接、VXKMDATA 参数元数据)。缓存 hash 含 repo HEAD + 构建树。
- [`triton-vortex/tests/test_interpreter.py`](../../../triton-vortex/tests/test_interpreter.py):官方教程形状的 vecadd/softmax/layernorm 走 interpreter(参考数值,后续 codegen 验收以此对照)。
- [`triton-vortex/tests/test_driver_launch.py`](../../../triton-vortex/tests/test_driver_launch.py) + [`kernels/vecadd.hip`](../../../triton-vortex/kernels/vecadd.hip):hipcc-vortex 编译的 **Triton 形状 kernel**(同 add_kernel 签名)经 driver 层在设备上运行,结果与 interpreter 一致。

## 验证记录(2026-09-14,SimX 后端)

```
$ python -c "import triton; from triton.backends import backends; print(sorted(backends.keys()))"
['amd', 'nvidia', 'vortex']

$ TRITON_INTERPRET=1 pytest tests/ -q      # 3 interpreter + 1 driver-launch
4 passed
```

driver 目标确认:`GPUTarget(backend='vortex', arch=64, warp_size=4)`;is_active=True。

## 踩坑记录

1. **插件模块不得冷启动 Triton**:`import triton_vortex.driver` 触发 `triton.__init__` → discovery 再 import 部分初始化的同名模块 → "0 concrete subclasses"。必须先 `import triton`(README 已注明;`triton_vortex/__init__` 安全)。
2. out-of-tree driver 必须真正继承 `DriverBase`(初版漏写基类,MRO 只有 object,discovery 静默视为非具体类)。
3. `typing.ModuleType` 不存在(在 `types`);3.8 无 `CubeLanguageTranslationError` 导出。

## 2026-09-15 codegen 探索结论(实证)

用 `triton.compile(ASTSource, target=GPUTarget("vortex",64,4))` 驱动真实管线:

- **AST→TTIR→TTGIR 的通用 pass 在我们的 backend 上实际可跑**(make_ttir 用 NVIDIA 同款通用 pass 序列:inliner/canonicalizer/combine/reorder-broadcast/cse/symbol-dce/loop-unroll;stage 契约为 module→module,由 `compile()` 的 `make_ir` 起头,`get_codegen_implementation` 返回最小 hooks 即可);
- **卡点精确定位在 ttgir→llvm-ir**:Triton 3.8 中该转换是每个后端自带的 C++ MLIR pass(`nvidia.passes.ttgpuir.add_to_llvm_ir` / AMD 同构),**没有目标无关入口** —— Vortex 版 pass 需以 MLIR C++ 编写并链入 backend,与 NVIDIA/AMD 后端同量级;
- 因此 `make_llir` 现在在正确边界上抛出带路线图的 `NotImplementedError`(而非假装有 llir)。管线产物 ttir/ttgir 可经 `TRITON_KERNEL_DUMP` 导出,作为编写该 pass 的输入。

## 后续(codegen 里程碑,计划 §8.2 映射表为蓝本)

1. **Vortex 版 `add_to_llvmir` MLIR pass**(program-id/lane → vx_spawn2 CSR;布局决策);
2. llir → VOLT clang `-x ir` → 两遍 vx_start.S 链接 → `vxbin.py`(复用 hipcc_vortex 链接管线);
3. 从 Triton signature 生成 VXKMDATA 参数布局(接 P1-3 ABI);
4. `tl.dot` → FPU tile 先行,TCU dtype 后置(capability 已就绪);
5. autotune 计数器接 MPM(禁 wall-clock-only)。
