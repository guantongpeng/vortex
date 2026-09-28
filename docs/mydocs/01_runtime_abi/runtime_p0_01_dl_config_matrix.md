# P0.1 DL 配置矩阵节点开发记录

## 目标

为后续 HIP、Triton、PyTorch 和量化工作建立唯一的 DL 配置输入。此节点不改变 `VX_config.toml` 的默认最小验证档，也不声称 `dl_rtl` 或 `dl_fpga` 已经通过硬件验证；它只把计划中的三档参数结构化，并提供一个不依赖第三方 Python 包的校验/导出工具。

## 实现

- [`ci/dl_config_matrix.json`](../../../ci/dl_config_matrix.json) 定义 `dl_functional`、`dl_rtl`、`dl_fpga` 三个 profile。
- [`ci/dl_config.py`](../../../ci/dl_config.py) 校验 schema、profile 唯一性、XLEN、shape、布尔开关和已声明的 `VX_CFG_*` 配置名，并导出 blackbox flags 与 `CONFIGS`。
- [`ci/test_dl_config.py`](../../../ci/test_dl_config.py) 覆盖仓库矩阵、XLEN 不覆盖、未知 profile 和重复 profile 等关键约束。
- profile 的 `shape` 只使用 [`ci/blackbox.sh`](../../../ci/blackbox.sh) 已支持的 knobs；TCU、DXA、A extension 等通过 `CONFIGS` 传入，避免引入第二套参数解析。XLEN 只用于 profile 适用性检查和输出，实际 XLEN 必须由 `../configure --xlen` 设置。

## 使用方式

在仓库根目录执行：

```bash
python3 ci/dl_config.py --list
python3 ci/dl_config.py --name dl_functional --xlen 64
python3 ci/dl_config.py --name dl_rtl --xlen 32 --format json
```

输出的 `XLEN`、`BLACKBOX_FLAGS` 和 `CONFIGS` 是后续测试脚本的输入；`XLEN` 应用于选择已经用同一 XLEN 执行过 `configure` 的 build 目录，不能转换成 `-DVX_CFG_XLEN` 覆盖项。运行真正的 Vortex 测试仍必须在已配置的 `build/` 目录进行，并在修改 TOML/Makefile 后先执行 `../configure`。

## 验收标准

1. `--list` 输出三个且仅三个 profile；
2. 三个 profile 在 XLEN 允许值上分别通过校验；
3. 不存在的 profile、非法 XLEN、重复 profile、非正 shape 或非 `-D` 配置返回非零退出码；
4. JSON 输出可被脚本直接解析，shell 输出不包含未引用的空格；
5. 本节点不影响默认配置和已有测试入口。

## 验证记录

验证命令和结果应在提交说明中保持可复现：

```bash
python3 ci/dl_config.py --list
python3 ci/dl_config.py --name dl_functional --xlen 32
python3 ci/dl_config.py --name dl_rtl --xlen 64 --format json
python3 ci/dl_config.py --name missing --xlen 64   # 预期失败
python3 -m unittest discover -s ci -p 'test_dl_config.py'
```

后续 P0.2 会在这些 profile 上运行 capability dump、现有 regression、TCU/DXA 和 HIP chipStar 基线，并把真实运行结果补入本记录或独立的带日期日志；在此之前不得把 profile 标记为“硬件已验证”。
