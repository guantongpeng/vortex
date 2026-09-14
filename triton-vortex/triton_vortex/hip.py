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

"""ctypes binding of libhip_vortex for the triton-vortex driver.

Packs kernel argument blocks with DEVICE-width pointers: the host is
64-bit but an rv32 device reads 32-bit pointers from the argument block
(measured in P2-02), so every launch goes through pack_args().
"""

from __future__ import annotations

import ctypes
import os
import struct


def _find_build(start):
    """Locate a configured Vortex build tree.

    Order: $VORTEX_BUILD, an ancestor with config.mk + sw/hip (running from
    a build-tree copy), then sibling build*/ directories of the repo root
    (editable install from the source tree).
    """
    env = os.environ.get("VORTEX_BUILD")
    if env and os.path.exists(os.path.join(env, "sw/hip/libhip_vortex.so")):
        return env
    d = os.path.abspath(start)
    for _ in range(6):
        if os.path.exists(os.path.join(d, "config.mk")) and os.path.exists(
                os.path.join(d, "sw/hip/libhip_vortex.so")):
            return d
        d = os.path.dirname(d)
    # source tree: walk up to the repo root, then try sibling build dirs
    d = os.path.abspath(start)
    for _ in range(6):
        if os.path.exists(os.path.join(d, "VX_config.toml")):
            # Prefer rv64 build trees (the reference configuration for the
            # backend), then anything else with the runtime built.
            cands = sorted((c for c in os.listdir(d)
                            if c.startswith("build")
                            and os.path.exists(os.path.join(
                                d, c, "sw/hip/libhip_vortex.so"))),
                           key=lambda c: (not c.endswith("64"), c))
            if cands:
                return os.path.join(d, cands[0])
            break
        d = os.path.dirname(d)
    raise RuntimeError(
        "triton_vortex: no configured build tree with sw/hip/libhip_vortex.so "
        "found (set VORTEX_BUILD)")


_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD = _find_build(_HERE)
_RUNTIME = os.path.join(_BUILD, "sw/runtime")

os.environ.setdefault("VORTEX_DRIVER", "simx")
ctypes.CDLL(os.path.join(_RUNTIME, "libvortex.so"), mode=ctypes.RTLD_GLOBAL)
hip = ctypes.CDLL(os.path.join(_BUILD, "sw/hip/libhip_vortex.so"))

# ---- prototypes -----------------------------------------------------------

hip.hipInit.argtypes = [ctypes.c_uint]
hip.hipInit.restype = ctypes.c_int
hip.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
hip.hipGetDeviceCount.restype = ctypes.c_int
hip.hipDeviceSynchronize.argtypes = []
hip.hipDeviceSynchronize.restype = ctypes.c_int
hip.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
hip.hipMalloc.restype = ctypes.c_int
hip.hipFree.argtypes = [ctypes.c_void_p]
hip.hipFree.restype = ctypes.c_int
hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                          ctypes.c_int]
hip.hipMemcpy.restype = ctypes.c_int
hip.hipModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
hip.hipModuleLoad.restype = ctypes.c_int
hip.hipModuleUnload.argtypes = [ctypes.c_void_p]
hip.hipModuleUnload.restype = ctypes.c_int
hip.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                     ctypes.c_void_p, ctypes.c_char_p]
hip.hipModuleGetFunction.restype = ctypes.c_int
hip.hipModuleLaunchKernel.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
    ctypes.c_uint32, ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
]
hip.hipModuleLaunchKernel.restype = ctypes.c_int
hip.hipGetErrorString.argtypes = [ctypes.c_int]
hip.hipGetErrorString.restype = ctypes.c_char_p

HIP_MEMCPY_H2D, HIP_MEMCPY_D2H = 1, 2

_initialized = False


def check(rc, what):
    if rc != 0:
        raise RuntimeError(
            f"triton_vortex: {what} failed: {hip.hipGetErrorString(rc).decode()}")
    return rc


def init():
    global _initialized
    if not _initialized:
        check(hip.hipInit(0), "hipInit")
        _initialized = True


def synchronize():
    check(hip.hipDeviceSynchronize(), "hipDeviceSynchronize")


def malloc(n):
    init()
    p = ctypes.c_void_p()
    check(hip.hipMalloc(ctypes.byref(p), n), "hipMalloc")
    return p.value


def free(p):
    check(hip.hipFree(ctypes.c_void_p(p)), "hipFree")


def memcpy(dst, src, n, kind):
    check(hip.hipMemcpy(ctypes.c_void_p(dst), ctypes.c_void_p(src), n, kind),
          "hipMemcpy")


def load_module(path):
    init()
    m = ctypes.c_void_p()
    check(hip.hipModuleLoad(ctypes.byref(m), path.encode()), "hipModuleLoad")
    return m.value


def unload_module(module):
    check(hip.hipModuleUnload(ctypes.c_void_p(module)), "hipModuleUnload")


def get_function(module, name):
    f = ctypes.c_void_p()
    check(hip.hipModuleGetFunction(ctypes.byref(f), ctypes.c_void_p(module),
                                   name.encode()), "hipModuleGetFunction")
    return f.value


def launch(func, grid, block, args_blob, shared=0):
    """Launch with the single-pointer arg-block ABI (args_size = len).
    `shared` maps to the launch's lmem_size field (kernels using
    __local_mem() must pass their CTA local memory request)."""
    params = (ctypes.c_void_p * 1)(ctypes.cast(args_blob, ctypes.c_void_p))
    check(hip.hipModuleLaunchKernel(
        ctypes.c_void_p(func), grid[0], grid[1] or 1, grid[2] or 1,
        block[0], block[1] or 1, block[2] or 1, shared, None, params, None),
        "hipModuleLaunchKernel")


def pack_args(fmt, values, dev_ptr_width=64):
    """Pack an arg block mirroring C struct layout: natural alignment per
    member ('p' aligns to the device pointer width), padded at the end to
    the struct alignment."""
    def align(n, a):
        return (n + a - 1) // a * a

    body = b""
    for f, v in zip(fmt, values):
        if f == "p":
            body = body.ljust(align(len(body), dev_ptr_width // 8), b"\0")
            body += int(v).to_bytes(dev_ptr_width // 8, "little")
        else:
            size = struct.calcsize("=" + f)
            body = body.ljust(align(len(body), size), b"\0")
            body += struct.pack("=" + f, v)
    return body.ljust(align(len(body), dev_ptr_width // 8), b"\0")
