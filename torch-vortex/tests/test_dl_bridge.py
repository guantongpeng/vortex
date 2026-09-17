# Copyright © 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ATen and the device DL library are the same kernel (W3.1).

The acceptance criterion is not "both give the right answer" -- two independent
implementations of conv2d would do that, agreeing only to within accumulation
order. It is that the ATen path and a direct vx_dnn_conv2d call land on the
*same* kernel, which is what makes "one algorithm maintained in one place"
true rather than aspirational. So these compare bit patterns, not tolerances:
a tolerance is precisely what would hide a second implementation.

What makes it work: the DL library speaks vortex2.h and needs this process's
device and queue, not ones of its own. A second device context would break
ordering with everything launched through HIP, so sw/hip hands out the handles
it owns (hipGetVxDevice, hipStreamGetQueue) and the extension initializes the
DL modules against them.
"""

import ctypes
import os
import struct

import pytest
import torch
import torch.nn.functional as F

from torch_vortex import _paths

CONV_SIG = [ctypes.c_void_p] + [ctypes.c_uint64] * 4 + [ctypes.c_uint32] * 11


def _hip():
    build = _paths.find_build()
    lib = ctypes.CDLL(os.path.join(build, "sw", "hip", "libhip_vortex.so"))
    ctypes.CDLL(os.path.join(build, "sw", "runtime", "libvortex.so"),
                mode=ctypes.RTLD_GLOBAL)
    sig = {
        "hipMalloc": (ctypes.c_int, [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]),
        "hipMemcpy": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_size_t, ctypes.c_int]),
        "hipGetVxDevice": (ctypes.c_int, [ctypes.POINTER(ctypes.c_void_p)]),
        "hipStreamGetQueue": (ctypes.c_int, [ctypes.c_void_p,
                                             ctypes.POINTER(ctypes.c_void_p)]),
        "hipDeviceSynchronize": (ctypes.c_int, []),
    }
    for name, (res, args) in sig.items():
        fn = getattr(lib, name)
        fn.restype = res
        fn.argtypes = args
    return lib


@pytest.fixture(scope="module")
def dl():
    """The HIP layer's handles plus the DL library's conv entry point."""
    build = _paths.find_build()
    hip = _hip()
    libdl = ctypes.CDLL(os.path.join(build, "sw", "dl", "libvortex_dl.so"))
    conv = libdl.vx_dnn_conv2d
    conv.restype = ctypes.c_int
    conv.argtypes = CONV_SIG
    return hip, conv


def _upload(hip, data):
    p = ctypes.c_void_p()
    assert hip.hipMalloc(ctypes.byref(p), len(data)) == 0
    assert hip.hipMemcpy(p, data, len(data), 1) == 0        # H2D
    return p


def _download(hip, p, size):
    buf = ctypes.create_string_buffer(size)
    assert hip.hipMemcpy(buf, p, size, 2) == 0              # D2H
    return buf.raw


def test_the_handles_come_from_this_process(dl):
    """The DL library is given the HIP layer's device and queue, not new ones.

    Opening a second device would be a second context, which is what W3.1
    forbids: DL work would stop being ordered with everything else.
    """
    hip, _ = dl
    dev, queue = ctypes.c_void_p(), ctypes.c_void_p()
    assert hip.hipGetVxDevice(ctypes.byref(dev)) == 0
    assert dev.value, "no device handle"
    assert hip.hipStreamGetQueue(None, ctypes.byref(queue)) == 0
    assert queue.value, "no default queue"


def test_aten_conv_is_the_dl_kernel(backend, dl):
    """Same operands, bit-identical results: one kernel, two entry points."""
    hip, conv = dl
    torch.manual_seed(11)
    x = torch.randn(1, 3, 8, 8)
    w = torch.randn(4, 3, 3, 3)
    b = torch.randn(4)

    aten = F.conv2d(x.to("vortex"), w.to("vortex"), b.to("vortex"),
                    padding=1).cpu()

    queue = ctypes.c_void_p()
    assert hip.hipStreamGetQueue(None, ctypes.byref(queue)) == 0

    def flat(t):
        return t.detach().contiguous().numpy().tobytes()

    dx = _upload(hip, flat(x))
    dw = _upload(hip, flat(w))
    db = _upload(hip, flat(b))
    out_shape = (1, 4, 8, 8)
    dout = _upload(hip, b"\0" * (4 * 4 * 8 * 8))
    rc = conv(queue, dx.value, dw.value, db.value, dout.value,
              1, 3, 8, 8, 4, 3, 3, 1, 1, 1, 1)
    assert rc == 0, "vx_dnn_conv2d returned %d" % rc
    assert hip.hipDeviceSynchronize() == 0

    got = torch.frombuffer(bytearray(_download(hip, dout, 4 * 4 * 8 * 8)),
                           dtype=torch.float32).reshape(out_shape)

    assert torch.equal(got, aten), (
        "the ATen path and a direct vx_dnn_conv2d call disagree; they are not "
        "the same kernel.\n  max |diff| = %g"
        % (got - aten).abs().max().item())


@pytest.mark.parametrize("kwargs", [
    {}, {"padding": 1}, {"stride": 2, "padding": 1}, {"bias": False},
])
def test_the_same_kernel_for_several_shapes(backend, dl, kwargs):
    """The agreement is not a coincidence of one argument combination."""
    hip, conv = dl
    torch.manual_seed(5)
    x = torch.randn(2, 3, 9, 7)
    w = torch.randn(4, 3, 3, 3)
    bias = None if kwargs.get("bias") is False else torch.randn(4)
    pad = kwargs.get("padding", 0)
    stride = kwargs.get("stride", 1)

    want = F.conv2d(x, w, bias, stride=stride, padding=pad)
    got = F.conv2d(x.to("vortex"), w.to("vortex"),
                   None if bias is None else bias.to("vortex"),
                   stride=stride, padding=pad).cpu()
    # Against the CPU reference this is a *different* implementation, so it
    # agrees only to within float32 accumulation order. Bit-equality is the
    # evidence of same-source, and it is only meaningful further down, between
    # the ATen path and a direct DL call.
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)

    def flat(t):
        return t.detach().contiguous().numpy().tobytes()

    queue = ctypes.c_void_p()
    assert hip.hipStreamGetQueue(None, ctypes.byref(queue)) == 0
    n, ci, hi, wi = x.shape
    co, _, kh, kw = w.shape
    ho = (hi + 2 * pad - kh) // stride + 1
    wo = (wi + 2 * pad - kw) // stride + 1
    dx = _upload(hip, flat(x))
    dw = _upload(hip, flat(w))
    db = _upload(hip, flat(bias)) if bias is not None else ctypes.c_void_p(0)
    dout = _upload(hip, b"\0" * (n * co * ho * wo * 4))
    rc = conv(queue, dx.value, dw.value, db.value or 0, dout.value,
              n, ci, hi, wi, co, kh, kw, pad, pad, stride, stride)
    assert rc == 0, "vx_dnn_conv2d returned %d" % rc
    assert hip.hipDeviceSynchronize() == 0
    direct = torch.frombuffer(bytearray(_download(hip, dout, n * co * ho * wo * 4)),
                              dtype=torch.float32).reshape(n, co, ho, wo)
    assert torch.equal(direct, got)
