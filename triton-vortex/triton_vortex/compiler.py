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

    # ---- generic Triton passes (frontend lowering is target-agnostic) ----

    def make_ttir(self, src, metadata, opt):
        from triton._C.libtriton import ir
        from triton.compiler.code_generator import ast_to_ttir
        mod, sig_keys, constants, attrs = ast_to_ttir(
            src, self, options=opt)
        pm = ir.pass_manager(mod.context)
        pm.enable_print(False)
        from triton._C.libtriton import transforms as _transforms
        _transforms.common.optimize_ttgir(mod, 1)
        metadata.name = src.fn.__name__
        metadata.sig_keys = sig_keys
        return mod

    def make_ttgir(self, src, metadata, opt):
        # No target-specific layout/legalization yet: the generic module
        # flows to llir; layout decisions belong to the vxbin translation.
        return src

    def make_llir(self, src, metadata, opt):
        from triton._C.libtriton import ir, passes as _passes
        mod = src
        pm = ir.pass_manager(mod.context)
        pm.enable_print(False)
        _passes.convert_scf_to_cf(pm)
        _passes.optimize_llir(pm)
        import io, contextlib
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            mod.print()
        return out.getvalue().encode()

    # ---- the Vortex-specific final stage ----

    def make_vxbin(self, llir_bytes, metadata, opt):
        raise NotImplementedError(
            "triton-vortex: llir->vxbin codegen is not implemented yet "
            "(P4 roadmap in triton-vortex/README.md): program-id lowering "
            "to Vortex CSRs, VOLT clang -x ir link with vxbin.py, and "
            "VXKMDATA arg metadata from the Triton signature. Use the "
            "Triton interpreter for reference numerics and the driver "
            "(triton_vortex.hip) with hipcc-compiled kernels meanwhile.")
