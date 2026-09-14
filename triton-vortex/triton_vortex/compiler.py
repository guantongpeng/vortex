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

"""triton-vortex compiler backend (plan P4-01, v0.1).

Registers the `vortex` backend and reuses Triton's generic frontend
lowering (ttir/ttgir/llir). The final `llir -> vxbin` translation is NOT
implemented yet — see the roadmap in the package README. Kernels raised to
the `bin` stage fail with an explicit error naming the missing pieces,
never with a silent CPU path.
"""

from __future__ import annotations

import hashlib
import os
from types import ModuleType
from typing import Dict

from triton.backends.compiler import BaseBackend, GPUTarget, Language


def _repo_hash():
    """Stable-enough fingerprint of the Vortex toolchain inputs for the
    compile cache key: repo HEAD + configured build dir."""
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))  # <repo>/triton-vortex/..
    head = ""
    try:
        head = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, universal_newlines=True).strip()[:12]
    except Exception:
        pass
    return f"{head}-{_build_dir()}"


def _build_dir():
    from . import hip as _hip
    return os.path.basename(os.path.dirname(_hip._BUILD)) or "unknown"


class VortexOptions:
    def __init__(self, **kwargs):
        self.num_warps = kwargs.get("num_warps", 4)
        self.num_stages = kwargs.get("num_stages", 1)
        self.num_ctas = 1
        self.cluster_dims = (1, 1, 1)
        self.warp_size = 4  # Vortex lanes/warp; queried at runtime in driver
        self.debug = kwargs.get("debug", False)
        self.bench = kwargs.get("bench", False)
        self.sanitize_overflow = False
        self.arch = "vortex64"

    def hash(self):
        return hashlib.sha256(
            repr((self.num_warps, self.num_stages, self.arch)).encode()
        ).hexdigest()[:12]


class VortexBackend(BaseBackend):
    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.capability = 1  # vortex64 fpu baseline

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "vortex"

    def parse_options(self, opts) -> VortexOptions:
        return VortexOptions(**(opts or {}))

    def hash(self):
        return f"vortex-{_repo_hash()}"

    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}

    def load_dialects(self, ctx):
        pass

    def get_codegen_implementation(self, options):
        # Minimal codegen hooks: Triton's generic code generator consults
        # these for naming/attribute decisions; the defaults (shared with
        # the in-tree backends' base behavior) suffice for IR generation.
        from triton.runtime import interpreter as _interp  # noqa: F401
        codegen_fns = type("CodegenFns", (), {})()
        module_map = {}
        return codegen_fns, module_map

    def add_stages(self, stages, options, language):
        if language == Language.TRITON:
            stages["ttir"] = lambda src, metadata: self.make_ttir(
                src, metadata, options)
            stages["ttgir"] = lambda src, metadata: self.make_ttgir(
                src, metadata, options)
        else:
            raise NotImplementedError(
                "triton-vortex: only Language.TRITON is wired up")
        stages["llir"] = lambda src, metadata: self.make_llir(
            src, metadata, options)
        stages["bin"] = lambda src, metadata: self.make_vxbin(
            src, metadata, options)

    # ---- generic Triton passes -------------------------------------------
    # Stages receive the module produced by compile()'s make_ir and must
    # return the next module (verified against the NVIDIA backend's shape).

    def make_ttir(self, mod, metadata, opt):
        from triton._C.libtriton import ir, passes
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, "make_ttir")
        return mod

    def make_ttgir(self, mod, metadata, opt):
        # Explored 2026-09-15: the ttgir->llvm-ir conversion is a
        # per-backend C++ pass (nvidia.passes.ttgpuir.add_to_llvmir /
        # AMD's equivalent) — there is no target-generic entry. A Vortex
        # version means an MLIR pass linked into the backend; until then
        # the pipeline deliberately stops before layout legalization and
        # make_vxbin raises (see below). ttgir is passed through so IR
        # dumps remain usable for writing that pass.
        return mod

    # ---- the Vortex-specific final stage ----

    def make_llir(self, mod, metadata, opt):
        raise NotImplementedError(
            "triton-vortex: the ttgir->llvm-ir pass is backend-specific "
            "(see make_ttgir note); a Vortex MLIR pass must be written and "
            "linked into the backend before an llir even exists to lower "
            "into a vxbin. Roadmap: (1) Vortex add_to_llvmir mapping "
            "program-id/lane ops onto the vx_spawn2 CSRs, (2) VOLT clang "
            "-x ir + the hipcc two-pass link to vxbin, (3) VXKMDATA arg "
            "metadata from the Triton signature. Until then: interpreter "
            "for reference numerics, triton_vortex.hip for device runs.")

    def make_vxbin(self, llir_bytes, metadata, opt):
        raise NotImplementedError(
            "triton-vortex: unreachable before make_llir is implemented "
            "(see make_llir).")
