# torch-vortex — PyTorch PrivateUse1 backend for Vortex

Registers `vortex` as a PrivateUse1 device so PyTorch eager programs allocate
in Vortex device memory and dispatch registered ops to device kernels.

## Architecture

```
torch eager
   │  PrivateUse1 dispatch key ("vortex" via rename_privateuse1_backend)
   ▼
torch_vortex (C++ extension)
   │  allocator / device guard / copy_ / ops
   ▼
libhip_vortex  ──  hipMalloc/hipMemcpy/hipModuleLaunchKernel
   ▼
vortex2.h ABI  ──  simx / rtlsim / xrt backends
```

- Data really lives in Vortex device memory (`vx_buffer_create` under
  `hipMalloc`); registered ops launch prebuilt `.vxbin` kernels through
  `hipModuleLaunchKernel`.
- Unregistered ops **raise**, naming the operator. There is a catch-all
  `PrivateUse1` fallback in `src/vortex_ext.cpp` whose only job is to refuse —
  the previous README advertised a `torch_vortex.fallback_counter` that never
  existed while every non-vortex factory call was being routed to the CPU.
- Kernel argument blocks are defined once in `kernels/torch_kernel_args.h`,
  shared verbatim by the host extension and the device compiler. `args_size` in
  the image metadata is generated from measured `sizeof` by `gen_metadata`, and
  the extension cross-checks it against the sidecar at init.

## Layout

- `torch_vortex/__init__.py` — backend rename, generated tensor methods, device
  module, extension build/load
- `torch_vortex/_paths.py` — source-tree / build-tree / kernel-image discovery
- `torch_vortex/env/` — environment manifest and supported-version check (W0.1)
- `src/vortex_ext.cpp` — allocator, DeviceGuardImpl, copy_, the aten ops
- `kernels/torch_kernel_args.h` — the single arg-block definition
- `kernels/gen_metadata.cpp` — generates the VXKMDATA `args_size` values
- `kernels/{ops,dnn}.hip` — device kernels
- `tests/` — pytest suite run against `VORTEX_DRIVER=simx`

## Usage

```python
import torch, torch_vortex
a = torch.randn(4).to("vortex")      # randn(device="vortex") needs RNG (W3.5)
b = torch.randn(4).to("vortex")
c = a + b                            # runs the vortex add kernel
c.cpu()                              # d2h copy
```

## Running the tests

From a configured build tree (see docs/mydocs/pytorch_plan.md §7):

```bash
cd build_torch64
../configure --xlen=64 --tooldir=/data/vortex-tools
make -s -C torch-vortex/kernels
export VORTEX_BUILD="$PWD" VORTEX_DRIVER=simx
export PYTHONPATH="$PWD/torch-vortex"
python -m pytest torch-vortex/tests -q                # fast tier
python -m pytest torch-vortex/tests -q --tier=full    # adds the model tests
python -m torch_vortex.env.check                      # environment manifest
```

`torch.utils.rename_privateuse1_backend` is called at import, so `torch` must
not have been asked for a different PrivateUse1 device first.

## Boundaries in this version

Everything below is refused loudly, with the work item that owns it named in
the message. None of it falls back to the CPU.

| Not supported yet | Owner |
|---|---|
| dtypes other than float32 (allocation is fine, compute is not) | W3.2 / W4.1 |
| scalar operands (`x + 1.0`), in-place `add_`/`mul_`, `as_strided`/`.t()` | W3.2 |
| strided destinations in `copy_`, dtype conversion in `copy_` | W3.2 |
| `torch.randn(device="vortex")`, `torch.manual_seed` on the device | W3.5 |
| grouped/dilated/transposed conv, `avg_pool2d`, ceil_mode | W3.3 |
| training: batch-norm training mode, autograd | W8.1 |
| multiple streams, `non_blocking` actually overlapping | W2.2 |
| warm restart / multiple kernel images in one process | W2.4 |
