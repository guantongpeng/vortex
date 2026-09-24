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

"""torch-vortex: PyTorch PrivateUse1 backend for Vortex.

Import order is load-bearing and is handled here:
  1. rename_privateuse1_backend("vortex") — makes "vortex" a device string
  2. generate_methods_for_privateuse1_backend() — adds Tensor.vortex()/.is_vortex
     (torch 2.14: the rename alone generates nothing)
  3. register the torch.vortex device module — required *before* any factory
     call, or `torch.zeros(..., device="vortex")` raises ModuleNotFoundError
     from C++ before it ever reaches the dispatcher
  4. build the C++ extension (allocator + guard + aten registrations)
  5. load the kernel image compiled by ci/hipcc_vortex.py

Environment:
  VORTEX_BUILD          configured build tree (see torch_vortex._paths)
  VORTEX_DRIVER         backend selected by libvortex (simx / rtlsim / xrt)
  TORCH_VORTEX_VXBIN    kernel image (default: <build>/torch-vortex/kernels)
  TORCH_VORTEX_SRC      C++ source dir, for source-tree checkouts
  TORCH_VORTEX_SKIP_LOAD=1  register the device but do not build/load the
                        extension; lets `torch_vortex.env.check` run in an
                        environment where the backend itself is broken
  TORCH_VORTEX_VERBOSE  show the extension build output
"""

import os

import torch

from . import _paths

__all__ = ["stats", "reset_stats", "arg_sizes", "load"]

_DRIVER_DEFAULT = "simx"

# Registration is process-global and torch refuses to repeat some of it, so the
# idempotence guard cannot live in a module global: importlib.reload resets
# those, which made a second import raise "device module of <class
# 'torch.Tensor'> has already been registered with is_vortex". Stamp torch
# instead, which survives a reload.
_REGISTERED_FLAG = "_torch_vortex_registered"

_ext = None
_rng_seed = 0
_rng_offset = 0


def _register_device_module():
    """Register torch.vortex before anything can call a factory."""
    if getattr(torch, _REGISTERED_FLAG + "_module", False):
        return
    if not hasattr(torch, "_register_device_module"):
        raise RuntimeError(
            "torch_vortex: this torch build has no "
            "torch._register_device_module; the PrivateUse1 device module "
            "cannot be registered and torch.zeros(device='vortex') would "
            "fail before reaching the dispatcher")
    torch._register_device_module("vortex", _device_module())
    setattr(torch, _REGISTERED_FLAG + "_module", True)


def _register_backend_name():
    """Device rename plus the generated Tensor/Module convenience methods."""
    if getattr(torch, _REGISTERED_FLAG, False):
        return
    torch.utils.rename_privateuse1_backend("vortex")
    generate = getattr(torch.utils, "generate_methods_for_privateuse1_backend",
                       None)
    if generate is None:
        # torch >= 2.5 always has the combined form; the 2.4-era split
        # generate_tensor_methods_* pair is gone. Without this call,
        # `rename_privateuse1_backend` alone leaves .vortex()/.is_vortex
        # undefined.
        raise RuntimeError(
            "torch_vortex: torch.utils.generate_methods_for_privateuse1_backend "
            "is unavailable on torch %s" % torch.__version__)
    generate()
    setattr(torch, _REGISTERED_FLAG, True)


def _device_module():
    """The torch.vortex device module (availability query / sync)."""

    class _VortexModule:
        def is_available(self):
            """Must be a callable, not a property.

            torch 2.14's _is_privateuse1_backend_available() does
            `getattr(torch.vortex, "is_available") and is_available()`, so a
            @property returning True raised "'bool' object is not callable"
            inside FakeTensorMode. Reporting real availability is W2.3.
            """
            return True

        def device_count(self):
            return 1

        def current_device(self):
            return 0

        def set_device(self, device):
            """Only device 0 exists; anything else is refused, not ignored."""
            index = device if isinstance(device, int) else getattr(device, "index", None)
            if index not in (None, -1, 0):
                raise RuntimeError(
                    "torch_vortex: only device 0 exists, got %r" % (device,))

        def is_initialized(self):
            return True

        def properties(self, device=None):
            """What the runtime actually reports, rather than a guess.

            Same shape as torch.cuda.get_device_properties: a plain object with
            the fields the runtime can answer.
            """
            ext = _require_ext()
            props = ext.device_properties()

            class _Props:
                name = "vortex"
                major = 0
                minor = 0
                warp_size = props["warp_size"]
                max_threads_per_block = props["max_threads_per_block"]
                shared_memory_per_block = props["shared_mem_per_block"]
                total_memory = props["total_global_mem"]

                def __repr__(self):
                    return ("_Props(name='vortex', warp_size=%d, "
                            "max_threads_per_block=%d, shared_memory_per_block=%d, "
                            "total_memory=%d)"
                            % (self.warp_size, self.max_threads_per_block,
                               self.shared_memory_per_block, self.total_memory))

            return _Props()

        def _is_in_bad_fork(self):
            return False

        def manual_seed_all(self, seed):
            global _rng_seed, _rng_offset
            _rng_seed = int(seed) & ((1 << 64) - 1)
            _rng_offset = 0
            return None

        def synchronize(self, device=None):
            """Route through the extension: one barrier implementation.

            This used to reach hipDeviceSynchronize through its own ctypes
            handle. Beyond the duplication, the extension's allocator now uses
            "has anything been enqueued since the last barrier" to decide
            whether a free can be immediate, so it has to see every barrier.
            """
            _require_ext().device_synchronize()

        def get_rng_state(self, device=None):
            return get_rng_state(device)

        def set_rng_state(self, state, device=None):
            set_rng_state(state, device)

    return _VortexModule()


def _load():
    """Build (once) and initialize the backend. Idempotent."""
    global _ext
    if _ext is not None:
        return _ext

    if os.environ.get("TORCH_VORTEX_SKIP_LOAD") == "1":
        return None

    os.environ.setdefault("VORTEX_DRIVER", _DRIVER_DEFAULT)
    build = _paths.find_build()
    repo = _paths.find_repo()
    vxbin = _paths.find_vxbin(build)

    runtime_dir = os.path.join(build, "sw", "runtime")
    hip_dir = os.path.join(build, "sw", "hip")
    dl_dir = _paths.find_dl_dir(build)
    hip_inc = os.path.join(repo, "sw", "hip", "include")
    src = os.environ.get(
        "TORCH_VORTEX_EXT_SRC",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src",
                     "vortex_ext.cpp"))
    src = os.path.abspath(src)
    if not os.path.exists(src):
        raise RuntimeError(
            "torch_vortex: C++ extension source not found at %s; set "
            "TORCH_VORTEX_EXT_SRC or TORCH_VORTEX_SRC" % src)

    # The backend .so the dispatcher dlopen()s must be globally visible; the
    # extension carries rpaths for the rest.
    import ctypes

    ctypes.CDLL(os.path.join(runtime_dir, "libvortex.so"),
                mode=ctypes.RTLD_GLOBAL)

    from torch.utils.cpp_extension import load

    build_dir = os.path.join(build, "torch-vortex", "_ext")
    os.makedirs(build_dir, exist_ok=True)
    _ext = load(
        name="torch_vortex_ext",
        sources=[src],
        extra_include_paths=[
            hip_inc,
            os.path.normpath(os.path.join(os.path.dirname(src), "..", "kernels")),
            # sw/dl's public headers: the unified ops call into that library
            # rather than reimplementing its kernels. They include vortex2.h,
            # which lives in the runtime's public include dir.
            os.path.join(repo, "sw", "dl", "include"),
            os.path.join(repo, "sw", "runtime", "include"),
            os.path.join(build, "sw", "runtime", "include"),
        ],
        # No -std override: torch 2.14 requires C++20 and cpp_extension already
        # selects it. Pinning an older standard here fails deep inside ATen
        # headers with "C++20 or later compatible compiler is required".
        extra_cflags=["-Wall"],
        extra_ldflags=[
            "-L%s" % hip_dir,
            "-lhip_vortex",
            "-L%s" % runtime_dir,
            "-lvortex",
            "-L%s" % dl_dir,
            "-lvortex_dl",
            "-Wl,-rpath,%s" % hip_dir,
            "-Wl,-rpath,%s" % runtime_dir,
            "-Wl,-rpath,%s" % dl_dir,
        ],
        build_directory=build_dir,
        verbose=bool(os.environ.get("TORCH_VORTEX_VERBOSE")),
    )
    _ext.load_ops(vxbin, dl_dir)
    _check_arg_sizes(vxbin, _ext)
    return _ext


def _check_arg_sizes(vxbin, ext):
    """Fail at init if the kernel image and this extension disagree.

    gen_metadata writes <vxbin>.meta.json from the same header the host
    compiles against, so a difference means the image is stale (or was hand-
    edited), and every launch would be reading the wrong bytes. Better to
    refuse to start than to produce plausible numbers.
    """
    import json

    sidecar = vxbin + ".meta.json"
    if not os.path.exists(sidecar):
        raise RuntimeError(
            "torch_vortex: no argument-size sidecar at %s; rebuild the kernel "
            "image with `make -C torch-vortex/kernels`" % sidecar)
    with open(sidecar) as f:
        meta = json.load(f)

    if meta.get("xlen") != 64:
        raise RuntimeError(
            "torch_vortex: kernel image %s is built for xlen=%s but this "
            "backend requires 64 (its argument blocks are LP64)"
            % (vxbin, meta.get("xlen")))

    host = ext.arg_sizes()
    image = {k: int(v) for k, v in meta.get("args_sizes", {}).items()}
    if host != image:
        diff = sorted(
            "%s: host=%s image=%s" % (k, host.get(k), image.get(k))
            for k in set(host) | set(image) if host.get(k) != image.get(k))
        raise RuntimeError(
            "torch_vortex: argument-block sizes differ between this extension "
            "and %s -- the kernel image is stale, rebuild it:\n  %s"
            % (sidecar, "\n  ".join(diff)))


def load():
    """Explicitly build and initialize the backend; returns the extension."""
    return _load()


def _require_ext():
    if _ext is None:
        _load()
    if _ext is None:
        raise RuntimeError(
            "torch_vortex: the extension is not loaded "
            "(TORCH_VORTEX_SKIP_LOAD=1 is set)")
    return _ext


def stats():
    """Device-side counters: launches, transfer bytes, allocations, syncs."""
    return _require_ext().stats()


def reset_stats():
    """Zero the counters returned by stats()."""
    return _require_ext().reset_stats()


def manual_seed(seed):
    """Reset the process-local device Philox stream."""
    global _rng_seed, _rng_offset
    _rng_seed = int(seed) & ((1 << 64) - 1)
    _rng_offset = 0
    return _rng_seed


def get_rng_state(device="vortex"):
    if device not in (None, "vortex"):
        raise RuntimeError("torch_vortex RNG state only supports device='vortex'")
    value = (_rng_seed.to_bytes(8, "little") +
             _rng_offset.to_bytes(8, "little"))
    return torch.tensor(list(value), dtype=torch.uint8)


def set_rng_state(state, device="vortex"):
    global _rng_seed, _rng_offset
    if device not in (None, "vortex"):
        raise RuntimeError("torch_vortex RNG state only supports device='vortex'")
    if state.device.type != "cpu" or state.dtype != torch.uint8 or state.numel() != 16:
        raise RuntimeError("torch_vortex RNG state must be a CPU uint8 tensor of length 16")
    value = bytes(state.tolist())
    _rng_seed = int.from_bytes(value[:8], "little")
    _rng_offset = int.from_bytes(value[8:], "little")


def rand(*sizes, device="vortex"):
    """Generate float32 uniform values using the Vortex Philox kernel."""
    global _rng_offset
    if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
        sizes = tuple(sizes[0])
    if device != "vortex":
        raise RuntimeError("torch_vortex.rand only supports device='vortex'")
    shape = tuple(int(s) for s in sizes)
    n = 1
    for s in shape:
        if s < 0:
            raise ValueError("negative dimensions are not allowed")
        n *= s
    out = _require_ext().rng_uniform(list(shape), int(_rng_seed), int(_rng_offset))
    _rng_offset += (n + 3) // 4
    return out


def arg_sizes():
    """Kernel argument-block sizes this extension was compiled against."""
    return _require_ext().arg_sizes()


_register_backend_name()
_register_device_module()
_load()
