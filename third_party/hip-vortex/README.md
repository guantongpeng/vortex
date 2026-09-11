# hip-vortex — native HIP device headers for Vortex

First milestone of the native HIPVortex path (plan §6, P2). This directory
provides the HIP surface that `ci/hipcc_vortex.py` compiles against; it does
not contain a host runtime (that is `libhip_vortex`, a later node) and does
not patch VOLT clang.

## Compilation model

HIP sources are compiled as RISC-V C++ for the Vortex device and linked into
a `.vxbin` (the repository's hostless execution model: `main()` runs on the
device itself, so the HIP host-side APIs are device-local shims). VOLT
clang's `-x hip` mode injects AMDGPU-only driver options that are invalid for
RISC-V targets, so the wrapper forces `-x c++` and this header set maps the
HIP surface onto the Vortex kernel APIs:

| HIP | Vortex |
|---|---|
| `__global__` | `extern "C"` + `annotate("vortex.kernel")` → `__vx_kentry_<name>` alias (by-name launch in `.vxbin` symbol tables) |
| `threadIdx/blockIdx/blockDim/gridDim`, `__syncthreads` | `vx_spawn.h` TLS variables and barrier |
| `__shfl_up/down/xor`, `__ballot`, `__all/__any`, `__syncwarp` | `vx_shfl_*`, `vx_vote_*`, `vx_wsync` |
| `atomicAdd/Max/Exch/CAS` | RVA `__atomic` builtins (float CAS loop) |
| `hipLaunchKernelGGL` | arg pack + per-signature trampoline → `vx_spawn_threads` |
| `hipMalloc/hipFree` | device-local bump allocator over `__hip_heap` |
| `hipMemcpy/hipMemset/hipDeviceSynchronize` | `memcpy`/`memset`/`vx_fence` (single address space) |
| `warpSize` | `vx_num_threads()` — queried, never assumed 32 |
| `printf` | `vx_printf` (tinyprintf; picolibc's stdio allocates via `_sbrk`, which is fatal on device) |

## Usage

```bash
python3 ci/hipcc_vortex.py --tooldir $TOOLDIR --build-dir <configured-build> \
  --offload-arch=vortex64 kernel.hip -c -o kernel.o
python3 ci/hipcc_vortex.py --tooldir $TOOLDIR --build-dir <configured-build> \
  --offload-arch=vortex64 kernel.o -o kernel.vxbin
```

or the test entry: `make -C tests/hip_native run-simx` from a configured
build directory.

## Not supported in this milestone

- triple-chevron `kernel<<<grid,block>>>(...)` syntax — requires a compiler
  driver that rewrites it; use `hipLaunchKernelGGL` (the portable HIP API).
- static `__shared__` declarations — the legacy spawn model sizes local
  memory per kernel, not per launch. Use `__local_mem(bytes)` with a
  compile-time tile size.
- `hipStream`/`hipEvent` semantics — typedefs exist for source
  compatibility; the hostless model has a single implicit stream.
- streams/events/hiprtc/module host APIs — `libhip_vortex` (P2-02) maps
  them onto `vortex2.h`.

`HIP_VORTEX_INCLUDE` can point at an alternative header installation; the
repository copy under `include/` is the default.
