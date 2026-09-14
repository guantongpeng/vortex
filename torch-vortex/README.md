# torch-vortex — PyTorch PrivateUse1 backend for Vortex

P5-01 of the fullstack plan: registers `vortex` as a PrivateUse1 device so
PyTorch eager programs allocate on Vortex buffers and dispatch registered
ops to device kernels.

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

- Data really lives in Vortex device memory (vx_buffer_create under
  hipMalloc); registered ops launch prebuilt .vxbin kernels through
  hipModuleLaunchKernel. Unregistered ops fall back to CPU **loudly**
  (torch_vortex.fallback_counter counts them; tests assert on silent
  fallback).
- The op kernels are KMU images compiled by ci/hipcc_vortex.py
  (--kernel-lib=vortex2), one per op family under kernels/.

## Layout

- torch_vortex/__init__.py — backend rename + device module + op registration
- torch_vortex/_ext.py — builds/loads the C++ extension
- src/vortex_ext.cpp — allocator, DeviceGuardImpl, copy_/fill/add/mul
- kernels/*.hip — device kernels for the registered ops
- tests/ — pytest suite run against VORTEX_DRIVER=simx

## Usage

```python
import torch, torch_vortex
a = torch.randn(4, device="vortex")
b = torch.randn(4, device="vortex")
c = a + b          # runs the vortex add kernel
c.cpu()            # d2h copy
```
