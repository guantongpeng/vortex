# Driver-layer acceptance (plan P4-01): loads the hipcc-compiled
# Triton-shaped kernel and runs it through triton_vortex.hip, comparing
# against the interpreter result of the same logical kernel.
import os
import sys

import pytest
import torch


@pytest.fixture(scope="module")
def vxbin():
    root = os.environ.get("TRITON_VORTEX_KERNEL",
                          os.path.join(os.path.dirname(__file__), "..",
                                       "kernels", "vecadd.vxbin"))
    if not os.path.exists(root):
        pytest.skip("vecadd.vxbin not built (make -C kernels)")
    return root


def test_driver_launch(vxbin):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from triton_vortex import hip

    n = 100
    x = torch.arange(n, dtype=torch.float32)
    y = torch.arange(n, dtype=torch.float32) * 2
    expected = x + y  # same as the interpreter's add_kernel result

    dx, dy, dout = hip.malloc(n * 4), hip.malloc(n * 4), hip.malloc(n * 4)
    hip.memcpy(dx, x.numpy().tobytes().__sizeof__() and x.numpy().ctypes.data,
               n * 4, hip.HIP_MEMCPY_H2D)
    hip.memcpy(dy, y.numpy().ctypes.data, n * 4, hip.HIP_MEMCPY_H2D)

    mod = hip.load_module(vxbin)
    fn = hip.get_function(mod, "add_kernel")
    blob = hip.pack_args("pppI", [dx, dy, dout, n])
    hip.launch(fn, ((n + 15) // 16, 1, 1), (16, 1, 1), blob)
    hip.synchronize()

    import ctypes
    out = (ctypes.c_char * (n * 4))()
    hip.memcpy(ctypes.addressof(out), dout, n * 4, hip.HIP_MEMCPY_D2H)
    got = torch.frombuffer(bytearray(out), dtype=torch.float32)
    torch.testing.assert_close(got, expected)

    hip.free(dx)
    hip.free(dy)
    hip.free(dout)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
