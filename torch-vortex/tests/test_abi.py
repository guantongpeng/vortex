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

"""Argument-block ABI (W1.7).

Three things must agree about the layout of every kernel argument block: the
header the device compiler parses, the header the host compiler parses, and the
args_size in the image metadata. They were three hand-written copies before,
and the sizes had drifted (conv declared 104 bytes for an 88-byte struct, pool
80 for 72). The runtime copies exactly args_size bytes off the host stack, so a
declared size that is too large is a host over-read.

These tests need no device execution: they compare the sidecar written at build
time, the sizes the extension was compiled with, and a fresh sizeof() from the
host compiler.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

from torch_vortex import _paths

# The two structs whose hand-typed sizes were wrong. Pinned so that a silent
# ABI change fails by name with both numbers visible.
PINNED_SIZES = {"tv_conv2d_kernel": 88, "tv_pool2d_kernel": 72}


@pytest.fixture(scope="module")
def sidecar():
    vxbin = _paths.find_vxbin()
    path = vxbin + ".meta.json"
    if not os.path.exists(path):
        pytest.fail("no ABI sidecar at %s; rebuild the kernel image with "
                    "`make -C torch-vortex/kernels`" % path)
    with open(path) as f:
        return json.load(f)


def test_sidecar_matches_extension(backend, sidecar):
    host = backend._ext.arg_sizes()
    image = {k: int(v) for k, v in sidecar["args_sizes"].items()}
    assert host == image, (
        "the extension and the kernel image disagree about argument sizes; "
        "the image is stale")


def test_pinned_sizes(sidecar):
    for name, want in PINNED_SIZES.items():
        got = sidecar["args_sizes"].get(name)
        assert got == want, (
            "%s args_size is %s, expected %s -- if the struct changed on "
            "purpose, update PINNED_SIZES and rebuild the image" % (name, got, want))


def test_sidecar_matches_fresh_compiler_run(sidecar, tmp_path):
    """Catches a Makefile whose metadata generator was never re-run.

    Compiles the checked-in header with the host compiler and compares what it
    measures against the image metadata, so a stale generator cannot pass.
    """
    kernels = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"))
    if not os.path.exists(os.path.join(kernels, "torch_kernel_args.h")):
        pytest.skip("kernel sources not present in this tree")

    body = textwrap.dedent("""
        #include <stdio.h>
        #include "torch_kernel_args.h"
        #define EMIT(name, type, mbx, lmem) printf("%s %zu\\n", #name, sizeof(type));
        int main(void) { TORCH_KERNEL_TABLE(EMIT) return 0; }
    """)
    src = tmp_path / "sizeof_probe.cpp"
    src.write_text(body)
    exe = tmp_path / "sizeof_probe"
    cc = os.environ.get("CXX", "c++")
    build = subprocess.run([cc, "-std=c++17", "-I", kernels, str(src), "-o", str(exe)],
                           capture_output=True, text=True, timeout=180)
    assert build.returncode == 0, build.stderr[-2000:]
    res = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr[-2000:]

    measured = {}
    for line in res.stdout.strip().splitlines():
        name, size = line.split()
        measured[name] = int(size)
    assert measured == sidecar["args_sizes"], (
        "the checked-in header and the image metadata disagree; rebuild the "
        "kernel image with `make -C torch-vortex/kernels`")


def test_image_is_64_bit(backend, sidecar):
    assert sidecar["xlen"] == 64, (
        "the argument blocks are LP64; an rv32 image cannot be used")


def test_stale_image_is_detected(backend, tmp_path, monkeypatch):
    """A mismatched sidecar must fail at init, not produce wrong numbers."""
    import torch_vortex
    from torch_vortex.env import manifest  # noqa: F401  (import side effect)

    vxbin = _paths.find_vxbin()
    with open(vxbin + ".meta.json") as f:
        meta = json.load(f)
    meta["args_sizes"]["tv_mm_kernel"] += 8
    bad = tmp_path / "bad.vxbin"
    bad.write_bytes(b"")
    sidecar_path = str(bad) + ".meta.json"
    with open(sidecar_path, "w") as f:
        json.dump(meta, f)

    with pytest.raises(RuntimeError, match="argument-block sizes differ"):
        torch_vortex._check_arg_sizes(str(bad), backend._ext)
