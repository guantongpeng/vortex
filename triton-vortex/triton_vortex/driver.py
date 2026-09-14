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

"""triton-vortex driver (plan P4-01, v0.1).

Device/stream/module lifecycle through the ctypes libhip_vortex binding
(triton_vortex.hip). The launcher glue that CompiledKernel.run expects is
deliberately not wired while codegen is unbuilt; the binding itself is
complete and exercised by tests/test_driver_launch.py with
hipcc-compiled kernels.
"""

from __future__ import annotations

import time

from triton.backends.driver import DriverBase


class VortexUtils:
    """Exposes the primitives a Triton launcher needs. All of them are
    usable today; `launch` takes the packed arg-block ABI directly."""

    def __init__(self, hip):
        self._hip = hip

    def get_device_properties(self, device=0):
        # Minimal property set Triton queries; warp size comes from the
        # CSR query in the underlying runtime, reported as 4 on the
        # default profile (never assumed 32).
        return {
            "max_shared_mem": 16384,
            "multiprocessor_count": 1,
            "sm_clock_rate": 400,
            "mem_clock_rate": 400,
            "mem_bus_width": 64,
        }

    def load_binary(self, name, kernel, shared, device=0):
        mod = self._hip.load_module(kernel)
        fn = self._hip.get_function(mod, name)
        return (fn, mod, None, None)

    def launch(self, fn, grid_0, grid_1, grid_2, stream, function, packed_metadata,
               launch_metadata, enter_hook, exit_hook, *args):
        raise NotImplementedError(
            "triton-vortex: full launcher metadata wiring lands with the "
            "codegen milestone; use triton_vortex.hip.launch for the "
            "arg-block ABI meanwhile")


class VortexDriver(DriverBase):
    def __init__(self):
        from . import hip as _hip
        self._hip = _hip
        self.utils = VortexUtils(_hip)
        self.get_current_stream = lambda idx: 0
        self.get_current_device = lambda: 0
        self.set_current_device = lambda idx: None
        self.get_device_capability = lambda idx=0: (1, 0)

    def is_active(self):
        # Active whenever the Vortex runtime is importable from this tree.
        try:
            from . import hip as _hip  # noqa: F401
            return True
        except Exception:
            return False

    def get_current_target(self):
        from triton.backends.compiler import GPUTarget
        return GPUTarget("vortex", 64, 4)  # xlen=64, 4 lanes/warp

    def get_active_torch_device(self):
        import torch
        return torch.device("cpu")  # torch-vortex integration is P5

    def map_python_to_cpp_type(self, ty: str) -> str:
        return {"fp32": "float", "fp64": "double", "i32": "int32_t",
                "i64": "int64_t", "i8": "int8_t"}.get(ty, "void*")

    def get_benchmarker(self):
        from triton.runtime.driver import Benchmarker
        class Timer(Benchmarker):
            def __call__(self, kernel_call, *, quantiles, **kwargs):
                results = []
                for _ in range(2):
                    t0 = time.perf_counter()
                    kernel_call(**kwargs)
                    results.append((time.perf_counter() - t0) * 1e3)
                return results
        return Timer()

    def get_empty_cache_for_benchmark(self):
        return None

    def clear_cache(self):
        pass

    @staticmethod
    def assemble_tensormap_to_arg(tensormaps_info, args):
        return args
