# E2E acceptance for the P4-02 TTIR->C->vxbin codegen: the official
# tutorial vecadd kernel compiled with triton.compile(target=vortex),
# executed on SimX through triton_vortex.hip, compared against the torch
# CPU reference (which the interpreter reproduces bit-exactly for this
# kernel: adds of integral fp32 values).
import os
import sys

import triton  # MUST be imported before any triton_vortex module
import triton.language as tl
import torch
import ctypes
from triton.compiler import ASTSource, compile as tri_compile
from triton.backends.compiler import GPUTarget


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def main():
    sys.path.insert(0, "/home/guantp/aichip/vortex/triton-vortex")
    from triton_vortex import hip

    src = ASTSource(
        fn=add_kernel,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32",
                   "n": "i32", "BLOCK": "constexpr"},
        constexprs={"BLOCK": 16})
    k = tri_compile(src, target=GPUTarget("vortex", 64, 4))

    md = k.metadata
    print("compiled kernel:", md.name)
    print("  vxbin_path:", md.vxbin_path)
    print("  sig:", md.sig, " fmt:", "".join(f for _, f in md.sig))
    print("  args_size:", md.args_size, " shared(lmem):", md.shared,
          " num_warps:", md.num_warps)
    print("  cache dir artifacts:",
          sorted(os.path.basename(p) for p in k.metadata_group.values()))

    n = 100
    x = torch.arange(n, dtype=torch.float32)
    y = torch.arange(n, dtype=torch.float32) * 2
    expected = x + y  # identical to the interpreter result for this kernel

    dx, dy, dout = hip.malloc(n * 4), hip.malloc(n * 4), hip.malloc(n * 4)
    hip.memcpy(dx, x.numpy().ctypes.data, n * 4, hip.HIP_MEMCPY_H2D)
    hip.memcpy(dy, y.numpy().ctypes.data, n * 4, hip.HIP_MEMCPY_H2D)

    mod = hip.load_module(md.vxbin_path)
    fn = hip.get_function(mod, md.name)
    fmt = "".join(f for _, f in md.sig)
    blob = hip.pack_args(fmt, [dx, dy, dout, n])
    assert len(blob) == md.args_size, (len(blob), md.args_size)
    grid = ((n + 15) // 16, 1, 1)
    block = (md.num_warps * 4, 1, 1)  # warp_size = 4 on vortex
    hip.launch(fn, grid, block, blob, shared=md.shared)
    hip.synchronize()

    out = (ctypes.c_char * (n * 4))()
    hip.memcpy(ctypes.addressof(out), dout, n * 4, hip.HIP_MEMCPY_D2H)
    got = torch.frombuffer(bytearray(out), dtype=torch.float32)

    max_abs = (got - expected).abs().max().item()
    bad = (~torch.isclose(got, expected, rtol=1e-6, atol=1e-6)).sum().item()
    print(f"vecadd: n={n} grid={grid} block={block} max_abs={max_abs:.3e} "
          f"bad={bad}")
    hip.free(dx)
    hip.free(dy)
    hip.free(dout)

    if bad == 0 and torch.allclose(got, expected, rtol=1e-6, atol=1e-6):
        print("PASSED")
        return 0
    print("FAILED")
    print("got     :", got[:16].tolist())
    print("expected:", expected[:16].tolist())
    return 1


if __name__ == "__main__":
    sys.exit(main())
