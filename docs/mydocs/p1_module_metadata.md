# P1.3 内核与模块 ABI 元数据

## 目标

为 `.vxbin` 增加可选、版本化的内核编译元数据，使 HIP/PyTorch/Triton 适配层可以在提交 launch 前获得自然 block、静态 LMEM、寄存器占用、ISA/扩展需求和参数 blob 大小。旧镜像没有元数据时必须继续工作。

## 实现

- `vx_kernel_info_t` 和 `vx_kernel_get_info` 加入公共 `vortex2.h`。调用者先设置 `struct_size`；返回结构包含入口 PC、内核名、`max_block[3]`、静态 LMEM、寄存器数、`required_isa`、`required_features`、`args_size` 和 flags。
- `.vxbin` 在现有 `VXSYMTAB` 之后可以追加 `VXKMDATA` 尾部。尾部格式为：字符串区、每条 52 字节固定记录、`n_records:u32`、8 字节 magic。记录字段全部按 little-endian 编码，避免 C++ 结构体布局进入文件 ABI。
- 加载器先剥离并校验 `VXKMDATA`，再按原逻辑解析 `VXSYMTAB`。元数据按内核名关联；未提供的 block 信息回退到设备默认值。
- `sw/kernel/scripts/vxbin.py` 增加 `build_metadata_footer(records)`。设置 `VX_KERNEL_METADATA=/path/file.json` 时，转换器读取 JSON 列表并把元数据附加到镜像；默认行为与以前完全一致。

## JSON 示例

```json
[
  {
    "name": "main",
    "max_block": [8, 2, 1],
    "static_lmem_bytes": 1536,
    "registers": 17,
    "required_isa": 257,
    "required_features": 512,
    "args_size": 64,
    "flags": 3
  }
]
```

编译器后续应在生成 ELF 时计算这些字段，并在构建 `.vxbin` 的同一 job 中生成 sidecar；运行时不猜测寄存器或资源需求。

## 验证

```bash
../configure --xlen=64 --tooldir=/data/vortex-tools   # 在 build_dl64 中
make -C sw/runtime/stub -s
make -C tests/runtime test_module_kernel -s
python3 -m py_compile sw/kernel/scripts/vxbin.py
```

`test_module_kernel` 新增合成 `VXKMDATA` 镜像，验证版本、入口 PC、block 覆盖、资源字段和扩展需求；原有无 footer 和 `VXSYMTAB` 测试继续覆盖回退路径。实际 simx/rtlsim 运行需要 Ramulator 的 `yaml-cpp` FetchContent；当前环境无法访问 GitHub，因此仅完成编译级验证，依赖恢复后应运行：

```bash
./ci/blackbox.sh --driver=simx --app=test_module_kernel
```

## 2026-09-11 补充验证

网络恢复后 SimX 后端构建成功。`make -C tests/regression/basic kernel.vxbin` 生成真实镜像后，`make -C tests/runtime run-simx` 中的 `test_module_kernel` 全部子用例通过：`test_module_load_file`、`test_module_load_bytes`、refcount，以及此前被跳过的 `test_launch_via_kernel_handle`（通过 `VX_TEST_VXBIN` 指向真实 `kernel.vxbin` 完成实际 launch）。元数据尾部解析与无 footer 回退路径均在实际设备路径上验证。

## 后续工作

1. 在 LLVM/Vortex kernel 编译器中产出 sidecar 字段并加入能力校验（设备 ISA、TCU/DXA、LMEM 上限）。
2. 将 `args_size` 与 launch 参数布局校验接入 `vx_enqueue_launch`，错误在提交前返回。
3. 为 HIP `hipFuncGetAttribute` 和 Triton autotune 提供字段映射与端到端模型样例。
