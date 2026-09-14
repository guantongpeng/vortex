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

"""torch-vortex: PyTorch PrivateUse1 backend for Vortex (plan P5-01).

Import order matters and is handled here:
  1. rename_privateuse1_backend("vortex") — generates tensor.vortex() etc.
  2. load the C++ extension (allocator + guard + aten registrations)
  3. load the op kernel image compiled by ci/hipcc_vortex.py

Environment:
  VORTEX_DRIVER   backend selected by libvortex (simx / rtlsim / ...)
  TORCH_VORTEX_VXBIN   path to torch_ops.vxbin (default: kernels/ next to
                       this package)
  VORTEX_BUILD    configured build tree providing libhip_vortex/libvortex
                  (default: repo build_dl64)
"""

import os

import torch

torch.utils.rename_privateuse1_backend("vortex")

def _find_upward(marker_files, start=None):
    """Walk up from this package until all marker files exist."""
    d = start or os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if all(os.path.exists(os.path.join(d, m)) for m in marker_files):
            return d
        d = os.path.dirname(d)
    raise RuntimeError(
        "torch_vortex: could not locate a directory containing " +
        ", ".join(marker_files))


# The source tree is identified by VX_config.toml; a configured build tree
# by its generated config.mk (works both from the source checkout and from
# a build-tree copy).
_REPO = _find_upward(("VX_config.toml", "sw"))
try:
    _BUILD = _find_upward(("config.mk", "sw/runtime/libvortex.so"))
except RuntimeError:
    _BUILD = os.environ.get("VORTEX_BUILD", os.path.join(_REPO, "build_dl64"))
_BUILD = os.environ.get("VORTEX_BUILD", _BUILD)
_VXBIN = os.environ.get(
    "TORCH_VORTEX_VXBIN",
    os.path.join(os.path.dirname(__file__), "..", "kernels", "torch_ops.vxbin"),
)

_ext = None


def _load():
    """Build (once) and initialize the backend. Idempotent."""
    global _ext
    if _ext is not None:
        return _ext
    if "VORTEX_DRIVER" not in os.environ:
        os.environ["VORTEX_DRIVER"] = "simx"
    runtime_dir = os.path.join(_BUILD, "sw", "runtime")
    hip_dir = os.path.join(_BUILD, "sw", "hip")
    os.environ.setdefault("LD_LIBRARY_PATH", "")
    # dlopen resolves through rpath baked into the extension; make the
    # runtime visible for the backend .so the dispatcher dlopen()s.
    import ctypes

    ctypes.CDLL(os.path.join(runtime_dir, "libvortex.so"), mode=ctypes.RTLD_GLOBAL)
    hip_inc = os.path.join(_REPO, "sw", "hip", "include")
    src = os.path.join(os.path.dirname(__file__), "..", "src", "vortex_ext.cpp")
    from torch.utils.cpp_extension import load

    _ext = load(
        name="torch_vortex_ext",
        sources=[src],
        extra_include_paths=[hip_inc],
        extra_ldflags=[
            f"-L{hip_dir}",
            "-lhip_vortex",
            f"-L{runtime_dir}",
            "-lvortex",
            f"-Wl,-rpath,{hip_dir}",
            f"-Wl,-rpath,{runtime_dir}",
        ],
        verbose=False,
    )
    _ext.load_ops(_VXBIN)
    _register_device_module()
    return _ext


def _register_device_module():
    module = types_module()
    try:
        torch._register_device_module("vortex", module)
    except AttributeError:
        pass  # very old torch: device module registration unavailable
    # torch >= 2.5 also exposes generate_tensor_methods_for_privateuse1_backend;
    # on 2.4 rename_privateuse1_backend alone generates the tensor methods.


def types_module():
    """Minimal torch.vortex device module (device count / sync)."""
    import ctypes

    hip = ctypes.CDLL(os.path.join(_BUILD, "sw", "hip", "libhip_vortex.so"))
    hip.hipInit.argtypes = [ctypes.c_uint]
    hip.hipDeviceSynchronize.restype = ctypes.c_int
    hip.hipDeviceSynchronize.argtypes = []

    class _VortexModule:
        @property
        def is_available(self):
            return True

        def device_count(self):
            return 1

        def synchronize(self, device=None):
            rc = hip.hipDeviceSynchronize()
            if rc != 0:
                raise RuntimeError(f"vortex synchronize failed: {rc}")

    return _VortexModule()


_load()
