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
import glob
import json
import os
import re
import struct
import subprocess
import textwrap

import pytest
import torch
import torch.nn.functional as F

from torch_vortex import _paths

CONV_SIG = [ctypes.c_void_p] + [ctypes.c_uint64] * 4 + [ctypes.c_uint32] * 11
POOL_SIG = [ctypes.c_void_p] + [ctypes.c_uint64] * 2 + [ctypes.c_uint32] * 11


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
    pool = libdl.vx_dnn_pool2d
    pool.restype = ctypes.c_int
    pool.argtypes = POOL_SIG
    gemm = libdl.vx_blas_gemm
    gemm.restype = ctypes.c_int
    gemm.argtypes = [ctypes.c_void_p, ctypes.c_int] + [ctypes.c_uint32] * 3 + \
                    [ctypes.c_float, ctypes.c_float] + [ctypes.c_uint64] * 3
    bn = libdl.vx_dnn_bn_affine
    bn.restype = ctypes.c_int
    bn.argtypes = [ctypes.c_void_p] + [ctypes.c_uint64] * 6 + \
                  [ctypes.c_uint32] * 3 + [ctypes.c_float]
    return hip, conv, pool, gemm, bn


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
    hip = dl[0]
    dev, queue = ctypes.c_void_p(), ctypes.c_void_p()
    assert hip.hipGetVxDevice(ctypes.byref(dev)) == 0
    assert dev.value, "no device handle"
    assert hip.hipStreamGetQueue(None, ctypes.byref(queue)) == 0
    assert queue.value, "no default queue"


def test_aten_conv_is_the_dl_kernel(backend, dl):
    """Same operands, bit-identical results: one kernel, two entry points."""
    hip, conv = dl[0], dl[1]
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
    hip, conv = dl[0], dl[1]
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


def test_aten_max_pool_is_the_dl_kernel(backend, dl):
    """Same comparison for pooling: bit patterns, not tolerances."""
    hip, _, pool = dl[0], dl[1], dl[2]
    torch.manual_seed(3)
    x = torch.randn(2, 3, 9, 9)
    k, st, pad = 3, 2, 1

    want = F.max_pool2d(x, k, st, pad)
    got = F.max_pool2d(x.to("vortex"), k, st, pad).cpu()
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)

    queue = ctypes.c_void_p()
    assert hip.hipStreamGetQueue(None, ctypes.byref(queue)) == 0
    n, c, hi, wi = x.shape
    ho = (hi + 2 * pad - k) // st + 1
    wo = (wi + 2 * pad - k) // st + 1
    dx = _upload(hip, x.detach().contiguous().numpy().tobytes())
    dout = _upload(hip, b"\0" * (n * c * ho * wo * 4))
    rc = pool(queue, dx.value, dout.value, n, c, hi, wi, k, k, pad, pad, st, st, 0)
    assert rc == 0, "vx_dnn_pool2d returned %d" % rc
    assert hip.hipDeviceSynchronize() == 0
    direct = torch.frombuffer(bytearray(_download(hip, dout, n * c * ho * wo * 4)),
                              dtype=torch.float32).reshape(n, c, ho, wo)
    assert torch.equal(direct, got), (
        "the ATen pooling path and a direct vx_dnn_pool2d call disagree")


def test_aten_adaptive_avg_pool_is_the_dl_kernel(backend, dl):
    """The avg path, through the op that is registered.

    avg_pool2d itself is not registered yet, and its count_include_pad default
    (divide by the full kernel area, padding included) is not the same thing as
    the DL kernel's `count` either -- that is W3.3. adaptive_avg_pool2d(1) has
    no padding, so both agree and the comparison is meaningful.
    """
    hip, _, pool = dl[0], dl[1], dl[2]
    torch.manual_seed(4)
    x = torch.randn(2, 3, 5, 7)

    want = F.adaptive_avg_pool2d(x, 1)
    got = F.adaptive_avg_pool2d(x.to("vortex"), 1).cpu()
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)

    queue = ctypes.c_void_p()
    assert hip.hipStreamGetQueue(None, ctypes.byref(queue)) == 0
    n, c, hi, wi = x.shape
    dx = _upload(hip, x.detach().contiguous().numpy().tobytes())
    dout = _upload(hip, b"\0" * (n * c * 4))
    rc = pool(queue, dx.value, dout.value, n, c, hi, wi, hi, wi, 0, 0, hi, wi, 1)
    assert rc == 0, "vx_dnn_pool2d returned %d" % rc
    assert hip.hipDeviceSynchronize() == 0
    direct = torch.frombuffer(bytearray(_download(hip, dout, n * c * 4)),
                              dtype=torch.float32).reshape(n, c, 1, 1)
    assert torch.equal(direct, got)


def test_pool_special_values_are_the_dl_kernels(backend, dl):
    """-inf and NaN survive the trip through the shared kernel."""
    hip, _, pool = dl[0], dl[1], dl[2]
    x = torch.tensor([[[[-float("inf"), -float("inf")],
                        [-float("inf"), -float("inf")]]],
                      [[[float("nan"), 0.0], [3.0, 4.0]]]])
    got = F.max_pool2d(x.to("vortex"), 2).cpu()
    assert got[0, 0, 0, 0] == float("-inf")
    assert torch.isnan(got[1, 0, 0, 0])
    torch.testing.assert_close(got, F.max_pool2d(x, 2), rtol=0, atol=0,
                               equal_nan=True)


@pytest.mark.parametrize("m,k,n", [(4, 4, 4), (16, 16, 4), (17, 17, 17), (2, 3, 5),
                                   (1, 16, 4), (8, 8, 8), (48, 20, 13)])
def test_aten_mm_is_the_dl_gemm(backend, dl, m, k, n):
    """Including the shapes that a local kernel got wrong under VOLT.

    torch.mm on vortex tensors now calls vx_blas_gemm, so the two must agree
    bit-for-bit. The direct call mirrors what the ATen path does -- the same
    pre-zeroed C and the same alpha/beta -- so a difference would be a real
    difference rather than a difference in how the comparison was set up.
    """
    hip, _, _, gemm = dl[0], dl[1], dl[2], dl[3]
    torch.manual_seed(m * 100 + k * 10 + n)
    a = torch.randn(m, k)
    b = torch.randn(k, n)

    aten = torch.mm(a.to("vortex"), b.to("vortex")).cpu()

    queue = ctypes.c_void_p()
    assert hip.hipStreamGetQueue(None, ctypes.byref(queue)) == 0
    da = _upload(hip, a.detach().contiguous().numpy().tobytes())
    db = _upload(hip, b.detach().contiguous().numpy().tobytes())
    dc = _upload(hip, b"\0" * (m * n * 4))
    rc = gemm(queue, 0, m, n, k, 1.0, 1.0, da.value, db.value, dc.value)
    assert rc == 0, "vx_blas_gemm returned %d" % rc
    assert hip.hipDeviceSynchronize() == 0
    direct = torch.frombuffer(bytearray(_download(hip, dc, m * n * 4)),
                              dtype=torch.float32).reshape(m, n)

    assert torch.equal(direct, aten), (
        "the ATen matmul and a direct vx_blas_gemm call disagree; they are not "
        "the same kernel.\n  max |diff| = %g" % (direct - aten).abs().max().item())


def test_aten_batch_norm_is_the_dl_kernel(backend, dl):
    """The third op W3.1 names, and the one that had the channel bug.

    The DL kernel derived its per-channel span as total/c until this increment
    fixed it, so this comparison is also what says the fix is the one being
    used rather than a copy of it.
    """
    hip, _, _, _, bn = dl
    torch.manual_seed(23)
    n, c, h, w = 3, 4, 3, 5           # batch > 1 on purpose
    x = torch.randn(n, c, h, w)
    mean = torch.randn(c)
    var = torch.rand(c) + 0.5
    weight = torch.randn(c)
    bias = torch.randn(c)
    eps = 1e-5

    want = F.batch_norm(x, mean, var, weight, bias, training=False, eps=eps)
    got = F.batch_norm(x.to("vortex"), mean.to("vortex"), var.to("vortex"),
                       weight.to("vortex"), bias.to("vortex"),
                       training=False, eps=eps).cpu()
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)

    queue = ctypes.c_void_p()
    assert hip.hipStreamGetQueue(None, ctypes.byref(queue)) == 0
    dx = _upload(hip, x.detach().contiguous().numpy().tobytes())
    dm = _upload(hip, mean.detach().contiguous().numpy().tobytes())
    dv = _upload(hip, var.detach().contiguous().numpy().tobytes())
    dw = _upload(hip, weight.detach().contiguous().numpy().tobytes())
    db = _upload(hip, bias.detach().contiguous().numpy().tobytes())
    dout = _upload(hip, b"\0" * (n * c * h * w * 4))
    rc = bn(queue, dx.value, dm.value, dv.value, dw.value, db.value,
            dout.value, n, c, h * w, ctypes.c_float(eps))
    assert rc == 0, "vx_dnn_bn_affine returned %d" % rc
    assert hip.hipDeviceSynchronize() == 0
    direct = torch.frombuffer(bytearray(_download(hip, dout, n * c * h * w * 4)),
                              dtype=torch.float32).reshape(n, c, h, w)

    assert torch.equal(direct, got), (
        "the ATen batch-norm path and a direct vx_dnn_bn_affine call disagree; "
        "they are not the same kernel.\n  max |diff| = %g"
        % (direct - got).abs().max().item())


# The kernel -> argument-struct mapping, read from the kernel sources rather
# than restated here: each entry point takes exactly one argument block.
_KERNEL_RE = re.compile(r"__global__\s+void\s+(\w+)\s*\(\s*(\w+)\s*\*\s*arg\s*\)")


def _dl_kernel_structs(repo):
    mapping = {}
    for path in sorted(glob.glob(os.path.join(repo, "sw", "dl", "src", "*_kernels.hip"))):
        for name, struct in _KERNEL_RE.findall(open(path).read()):
            mapping[name] = struct
    return mapping


def _measure_dl_structs(repo, xlen, mapping, tmp_path):
    """sizeof every argument block, using the host compiler.

    For rv64 the host is already the right data model (LP64). For rv32,
    defining VX_CFG_XLEN=32 without __VORTEX__ selects the header's uint32_t
    pointer branch, which is the layout an rv32 build gets -- so one host
    compiler measures both, and the result is a real measurement rather than
    an arithmetic guess.
    """
    names = [k for k in sorted(mapping) if not k.startswith("mxfp8")]
    probe = tmp_path / ("dl_sizes_%d.cpp" % xlen)
    probe.write_text(
        "#include <stdio.h>\n"
        + "".join('#include "%s"\n' % os.path.basename(h) for h in sorted(
            glob.glob(os.path.join(repo, "sw", "dl", "src", "*_args.h")))
            if os.path.basename(h) != "mxfp8_args.h")
        + "int main(void){\n"
        + "".join('  printf("%s %%zu\\n", sizeof(%s));\n' % (n, mapping[n]) for n in names)
        + "  return 0;\n}\n")
    exe = tmp_path / ("dl_sizes_%d" % xlen)
    cc = os.environ.get("CXX", "c++")
    build = subprocess.run(
        [cc, "-std=c++17", "-I", os.path.join(repo, "sw", "dl", "src"),
         "-I", os.path.join(repo, "sw", "dl", "include"),
         "-DVX_CFG_XLEN=%d" % xlen, str(probe), "-o", str(exe)],
        capture_output=True, text=True, timeout=300)
    assert build.returncode == 0, build.stderr[-2000:]
    res = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr[-2000:]
    return {line.split()[0]: int(line.split()[1])
            for line in res.stdout.strip().splitlines()}


def test_dl_metadata_matches_measured_sizes(tmp_path):
    """sw/dl's args_size is hand-typed, and it had drifted -- the F07 class.

    Measured against the image metadata for every image whose argument headers
    compile outside the device context (all but mxfp8, whose header declares
    device-side helpers). Thirteen numbers were wrong when this was first run:
    conv declared 96 for 88, pool 88 for 72, bn 56 for 72, quant's w8a8 gemm 48
    for 56, and the two llm entries 24 for 32. bn was the dangerous direction --
    a declared size *smaller* than the struct -- and it appeared because adding
    a field to bn_args_t did not make anyone look at this table.

    The 32-bit column is a prediction about a tree that is not here, so it is
    checked against the Makefile rather than against a built image.
    """
    repo = _paths.find_repo()
    mapping = _dl_kernel_structs(repo)
    assert mapping, "no kernels found under sw/dl/src"

    config = open(os.path.join(_paths.find_build(), "config.mk")).read()
    tree_xlen = int(re.search(r"^XLEN\s*\??=\s*(\d+)", config, re.M).group(1))
    measured = _measure_dl_structs(repo, tree_xlen, mapping, tmp_path)

    declared = {}
    for path in sorted(glob.glob(os.path.join(_paths.find_build(), "sw", "dl",
                                              "*_meta.json"))):
        for rec in json.load(open(path)):
            declared[rec["name"]] = rec["args_size"]
    if not declared:
        pytest.skip("no sw/dl images built in this tree")

    wrong = {k: (measured[k], declared[k]) for k in measured
             if k in declared and declared[k] != measured[k]}
    assert not wrong, (
        "sw/dl's declared args_size disagrees with the structs "
        "(kernel: measured vs declared): %r" % wrong)


def test_dl_metadata_32_bit_column(tmp_path):
    """The rv32 half of the same table, checked against the Makefile."""
    repo = _paths.find_repo()
    mapping = _dl_kernel_structs(repo)
    measured = _measure_dl_structs(repo, 32, mapping, tmp_path)
    makefile = open(os.path.join(repo, "sw", "dl", "Makefile")).read()

    # resolve the variables each image's metadata references, in the rv32 branch
    tail = makefile[makefile.index("ifeq ($(XLEN),64)"):]
    rv32 = dict(re.findall(r"^(\w+)\s*:=\s*(\d+)$", tail.split("else", 1)[1].split("endif")[0], re.M))
    var_of = {}
    for m in re.finditer(r"^(\w+)_META\s*:=\s*(\[.*\])$", makefile, re.M):
        for name, var in re.findall(
                r'"name":\s*"([a-z_0-9]+)",\s*"args_size":\s*\$\((\w+)\)', m.group(2)):
            var_of.setdefault(name, var)
    for name, var in re.findall(r'"name":\s*"([a-z_0-9]+)",\s*"args_size":\s*(\w+)\}',
                                re.search(r"^LLM_META\s*:=\s*(\[.*\])$", makefile, re.M).group(1)):
        var_of.setdefault(name, var)

    wrong = {}
    for name, size in measured.items():
        var = var_of.get(name)
        if var is None or var not in rv32:
            continue
        if int(rv32[var]) != size:
            wrong[name] = (size, int(rv32[var]))
    assert not wrong, (
        "sw/dl's rv32 declared args_size disagrees with the structs "
        "(kernel: measured vs declared): %r" % wrong)
