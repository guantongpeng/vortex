# P2.3 native HIP kernel 与并发验收

## 目标

完成计划 P2-03：native HIP 的 reduction、FP32 GEMM、atomic 与 stream/event overlap 测试，并把 P2-01/P2-02 的入口在 rtlsim（周期级 RTL）上补验。

## 交付物

- [`tests/hip_native/reduction/`](../../tests/hip_native/reduction/)：`__shfl_down` 树状归约，256 个单 warp block，grid-stride 累加。
- [`tests/hip_native/sgemm/`](../../tests/hip_native/sgemm/)：FP32 通用 FPU kernel（32×32×32，2D grid/block），设备端 reference 逐元素校验（max_rel=0）。FP16/BF16 GEMM 属 P3-01。
- [`tests/hip_native/atomics/`](../../tests/hip_native/atomics/)：int `atomicAdd` 直方图（512 元素、8 bin、32×16 线程），编译期 `VX_CFG_EXT_A_ENABLED` 门控。
- [`tests/hip_native/stream_overlap/`](../../tests/hip_native/stream_overlap/)：libhip_vortex 双流 + 跨流 `hipStreamWaitEvent` + `hipMemcpyAsync`，KMU 镜像。
- `libhip_vortex` 新增：`hipMemcpyAsync`/`hipMemsetAsync`/`hipStreamWaitEvent`（`vx_enqueue_wait_value`）。
- `tests/hip_native/Makefile`：atomics 拆分为 `run-simx-a` 目标（需 A 扩展构建树）。

## 两个重要实测发现（影响上层设计）

### 1. 老spawn 模型的 `__syncthreads` 对多 warp block 不可靠

诊断路径（`/tmp/hipdiag` 系列，从 64×16 全组合逐项拆到单因素）：

- 块调度、grid-stride 循环、warp shuffle、LMEM 基址（各 warp 一致，0x1ffff0000）单独全部正确；
- 但"多 warp block + `__syncthreads` + LMEM staging"组合的结局随时序变化：**死锁**（单 block 也挂）、**丢失 LMEM 更新**（单 block 读到 256/1024）、或**侥幸通过**（64 block 连跑全对）。

根因：`vx_spawn.c` 中每个 warp 独立循环处理多个 group（`for group_id += stride`，无组间同步），`__syncthreads` 用 `local_group_id` 做 barrier id——block 跨多 warp 时它恒为 0，且 spawn 自身的 join 也用同一 barrier；不同 group 的 barrier 到达混合计数。仓库现有证据一致：**所有使用 `__syncthreads` 的 regression 测试都跑在 KMU 路径（`KERNEL_LIB := vortex2`）**，老模型仅 sgemm2_v1 例外。

结论与约束（已写入 `third_party/hip-vortex/include/hip/hip_runtime.h` 头注释）：hostless 模式只保证单 warp block；多 warp CTA 语义必须走 KMU（`hipcc --kernel-lib=vortex2` + 主机 launch）。这不是本测试的规避，而是老模型的边界。

### 2. float atomicAdd 无 ZACAS 时同地址活锁

float 没有 AMO，头文件用 CAS 循环实现。实测 4 个 lane 对同一地址累加即**永久活锁**：无 ZACAS 时 compare-exchange 降级为 lr/sc，各 lane 的 store 条件互相踢掉对方的 reservation。int `atomicAdd` 直达 AMO 硬件无此问题（128/128 正确，含 16 线程多 warp block）。

处置：atomics 测试只覆盖 int 路径；头文件 `atomicAdd(float*)` 注释明确"仅限单 lane 每地址或 ZACAS 硬件"。对 DL 上层（softmax 在线归一、scatter-add）意味着：**应避免 float 原子聚合，或要求 ZACAS capability**。

## 验证记录（2026-09-11）

SimX（build_dl64 默认档 / build_dl64a A 扩展档）：

| 测试 | simx | rtlsim |
|---|---|---|
| vecadd（P2-01） | PASSED | PASSED |
| module_api（P2-02） | PASSED | PASSED（elapsed 171ms，真实 RTL 周期） |
| reduction | PASSED（11997.750 精确） | PASSED |
| sgemm FP32 | PASSED（max_rel=0，1024 元素） | PASSED |
| atomics（`CONFIGS=-DVX_CFG_EXT_A_ENABLE`，build_dl64a） | PASSED | 未跑（需 A 扩展 rtlsim，后续） |
| stream_overlap | PASSED | PASSED |

命令入口：`make -C tests/hip_native run-simx`（默认档）；`make -C tests/hip_native/atomics run-simx CONFIGS=-DVX_CFG_EXT_A_ENABLE`（A 档，在 build_dl64a）。

未做（如实记录）：chipStar 同源对照（P2.3 第 7 项，chipStar 兼容路径另行维护）；hiprtc 明确未实现（无 stub）；rtlsim 的 A 扩展档；SimX↔rtlsim retired-instruction 精确相等校验（`ci/trace_csv.py` 流程属 CI parity 节点）。

## 后续

1. KMU 模式下的多 warp block + `__syncthreads` + LMEM 测试（P3 reference harness 顺带覆盖）；
2. ZACAS 档的 float atomicAdd 验证；
3. `ci/testcases/hip_native.yaml`：把上述测试纳入 pytest 分类（functional/parity 分开）。
