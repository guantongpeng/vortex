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

"""Several kernel images in one process (W2.4).

Every image used to link at 0x80000000, and the runtime's module loader
reserves [min_vma, max_vma) per image, so the second hipModuleLoad failed with
an address-range conflict. That is why torch-vortex merged its ops and dnn
kernels into one image, and why triton-vortex runs each case in its own
subprocess.

Nothing in the runtime was ever responsible: the loader places an image
wherever its own header says, and the launch PC comes from the loaded module's
base. The fix is that each image gets its own slot at link time, from
sw/common/module_slots.mk.

These tests load images through the HIP layer directly, because the extension
resolves exactly one image at init and refuses a second one by name.
"""

import ctypes
import os
import struct

import pytest
import torch

from torch_vortex import _paths

RNG_ARGS = "<QIIIII I"      # vx_rng_uniform_args_t: out, off_lo, off_hi, k0, k1, n, pad
RNG_ARGS_SIZE = 32
HIP_BINARY_ARGS = "<QQQII"  # torch-vortex binary_args_t
HIP_BINARY_ARGS_SIZE = 32


def _hip():
    build = _paths.find_build()
    lib = ctypes.CDLL(os.path.join(build, "sw", "hip", "libhip_vortex.so"))
    ctypes.CDLL(os.path.join(build, "sw", "runtime", "libvortex.so"),
                mode=ctypes.RTLD_GLOBAL)
    sig = {
        "hipMalloc": (ctypes.c_int, [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]),
        "hipMemcpy": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_size_t, ctypes.c_int]),
        "hipModuleLoad": (ctypes.c_int, [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]),
        "hipModuleGetFunction": (ctypes.c_int, [ctypes.POINTER(ctypes.c_void_p),
                                                ctypes.c_void_p, ctypes.c_char_p]),
        "hipModuleLaunchKernel": (ctypes.c_int,
                                  [ctypes.c_void_p] + [ctypes.c_uint] * 3 +
                                  [ctypes.c_uint] * 3 +
                                  [ctypes.c_uint, ctypes.c_void_p,
                                   ctypes.c_void_p, ctypes.c_void_p]),
        "hipDeviceSynchronize": (ctypes.c_int, []),
        "hipGetErrorString": (ctypes.c_char_p, [ctypes.c_int]),
    }
    for name, (res, args) in sig.items():
        fn = getattr(lib, name)
        fn.restype = res
        fn.argtypes = args
    return lib


def _load(lib, path):
    mod = ctypes.c_void_p()
    rc = lib.hipModuleLoad(ctypes.byref(mod), str(path).encode())
    assert rc == 0, "hipModuleLoad(%s) failed: %s" % (
        path, lib.hipGetErrorString(rc).decode())
    return mod


def _function(lib, mod, name):
    fn = ctypes.c_void_p()
    rc = lib.hipModuleGetFunction(ctypes.byref(fn), mod, name.encode())
    assert rc == 0, "%s missing from the image: %s" % (
        name, lib.hipGetErrorString(rc).decode())
    return fn


def _dev(lib, payload):
    p = ctypes.c_void_p()
    assert lib.hipMalloc(ctypes.byref(p), len(payload)) == 0
    assert lib.hipMemcpy(p, payload, len(payload), 1) == 0      # H2D
    return p


def _host(lib, p, n):
    buf = ctypes.create_string_buffer(n)
    assert lib.hipMemcpy(buf, p, n, 2) == 0                      # D2H
    return buf.raw


def _launch(lib, fn, args_bytes, grid, block):
    buf = ctypes.create_string_buffer(args_bytes, len(args_bytes))
    ptr = ctypes.c_void_p(0x01)      # HIP_LAUNCH_PARAM_BUFFER_POINTER
    size = ctypes.c_void_p(0x02)     # HIP_LAUNCH_PARAM_BUFFER_SIZE
    extra = (ctypes.c_void_p * 5)(
        ptr, ctypes.cast(buf, ctypes.c_void_p), size,
        ctypes.c_void_p(len(args_bytes)), None)
    rc = lib.hipModuleLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, None,
                                   None, extra)
    assert rc == 0, lib.hipGetErrorString(rc).decode()
    assert lib.hipDeviceSynchronize() == 0


@pytest.fixture(scope="module")
def hip_lib():
    return _hip()


@pytest.fixture(scope="module")
def dl_modules(hip_lib):
    """Load the DL images once for the whole module.

    The runtime keys a loaded image by its address range, so loading the same
    file twice in one process is a conflict -- which is also why the loader
    belongs behind a fixture here rather than in each test.
    """
    return {name: _load(hip_lib, _dl_image(name)) for name in ("rng", "prim")}


def _dl_image(name):
    path = os.path.join(_paths.find_build(), "sw", "dl", name + ".vxbin")
    if not os.path.exists(path):
        pytest.skip("%s not built in this tree" % path)
    return path


def test_slots_are_distinct(backend):
    """Duplicate slots would put two images back on top of each other.

    The table is hand-written, so read it and check rather than trusting it.
    """
    table = os.path.join(_paths.find_repo(), "sw", "common", "module_slots.mk")
    slots = {}
    for line in open(table):
        line = line.strip()
        if not line.startswith("MODULE_SLOT_") or ":=" not in line:
            continue
        name, _, value = line.partition(":=")
        slots[name[len("MODULE_SLOT_"):].strip()] = value.strip()
    assert slots, "no slots found in %s" % table
    dupes = {v for v in slots.values() if list(slots.values()).count(v) > 1}
    assert not dupes, "module slots assigned twice: %r in %r" % (dupes, slots)


def test_a_second_image_loads_alongside_the_backends(backend, hip_lib, dl_modules):
    """The whole point: the extension holds an image, and a second one loads.

    Before distinct slots this failed with "address range exceeds the
    reserved range" -- the message triton-vortex's rowmax script documents.
    """
    lib = hip_lib
    a = torch.ones(4096, device="vortex")
    b = torch.ones(4096, device="vortex")
    assert bool(((a + b).cpu() == 2.0).all())

    second = dl_modules["rng"]
    _function(lib, second, "rng_philox4x32_uniform_kernel")

    # and the first image still works with the second one resident
    assert bool(((a + b).cpu() == 2.0).all())


def test_kernels_from_two_images_both_execute(backend, hip_lib, dl_modules):
    """Loading is not enough; both images' kernels have to run correctly."""
    lib = hip_lib
    n = 256
    out = _dev(lib, b"\0" * (4 * n))

    # image 2: the DL RNG kernel, launched by hand
    rng_fn = _function(lib, dl_modules["rng"], "rng_philox4x32_uniform_kernel")
    args = struct.pack(RNG_ARGS, out.value, 0, 0, 0, 0, n, 0)
    assert struct.calcsize(RNG_ARGS) == RNG_ARGS_SIZE
    _launch(lib, rng_fn, args, grid=(n + 3) // 4, block=4)

    values = struct.unpack("<%df" % n, _host(lib, out, 4 * n))
    assert all(0.0 <= v < 1.0 for v in values), "the RNG kernel did not run"
    assert len(set(values)) > n // 2, "the RNG kernel produced constants"

    # image 1: the backend's own op, via the extension
    a = torch.full((n,), 2.0, device="vortex")
    assert bool(((a * a).cpu() == 4.0).all())


def test_three_images_coexist(backend, hip_lib, dl_modules):
    assert len(dl_modules) == 2      # plus the backend's own image, loaded at import
    # and the backend's image is still usable
    a = torch.ones(64, device="vortex")
    three = torch.full((64,), 3.0, device="vortex")
    assert bool(((a * three).cpu() == 3.0).all())
