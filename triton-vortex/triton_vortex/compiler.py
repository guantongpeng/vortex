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

"""triton-vortex compiler backend (plan P4-01/P4-02).

Registers the `vortex` backend and reuses Triton's generic frontend
lowering (ttir/ttgir). The final lowering to a device image is a
deliberate, documented source-to-source path: `make_llir` serializes the
TTIR and transpiles it to a KMU C kernel (triton_vortex.ttir_to_c), and
`make_vxbin` compiles that C with ci/hipcc_vortex.py --kernel-lib=vortex2
into a real .vxbin recorded in the compile metadata. Ops outside the
transpiler whitelist fail with NotImplementedError naming the op — never
a silent CPU path.
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
        # Frontend hooks consulted during TTIR generation (tl.dot et al.
        # query these before the transpiler ever sees the op; the values
        # only shape IR generation, never the P4-02 lowering).
        self.allowed_dot_input_precisions = ["tf32", "tf32x3", "ieee"]
        self.max_num_imprecise_acc_default = 0
        self.arch = "vortex64"

    def hash(self):
        return hashlib.sha256(
            repr((self.num_warps, self.num_stages, self.arch)).encode()
        ).hexdigest()[:12]


class VortexBackend(BaseBackend):
    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.capability = 1  # vortex64 fpu baseline
        self.binary_ext = "vxbin"

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "vortex"

    def parse_options(self, opts) -> VortexOptions:
        return VortexOptions(**(opts or {}))

    def hash(self):
        return f"vortex-{_repo_hash()}"

    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}

    def pack_metadata(self, metadata):
        # Consumed by the launcher glue (driver.VortexUtils.launch): the
        # launch geometry and arg-block contract of the transpiled kernel.
        return {
            "name": metadata.name,
            "num_warps": metadata.num_warps,
            "shared": metadata.shared,
            "args_size": metadata.args_size,
            "sig": metadata.sig,
        }

    def load_dialects(self, ctx):
        pass

    def get_codegen_implementation(self, options):
        # Minimal codegen hooks: Triton's semantic layer queries these as a
        # dict during IR generation. min_dot_size is permissive so tl.dot
        # kernels reach TTIR and are then rejected by the P4-02 transpiler
        # whitelist with an error naming tt.dot (instead of an opaque
        # AttributeError inside the frontend).
        # NOTE: return the dict itself — compile() forwards this verbatim
        # to ASTSource.make_ir (the nvidia backend returns a bare dict).
        return {
            "min_dot_size": lambda lhs_type, rhs_type: (1, 1, 1),
        }

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
        # NOTE: the stage name is the artifact extension compile() uses;
        # it must equal binary_ext ("vxbin") so CompiledKernel.asm reads
        # the device image as bytes.
        stages["vxbin"] = lambda src, metadata: self.make_vxbin(
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
        # the module is passed through as TTIR and the final lowering is
        # the P4-02 source-to-source path in make_llir below. ttgir stays
        # a real (pass-through) stage so IR dumps remain usable for
        # writing that pass later.
        return mod

    # ---- the Vortex-specific final stage ----

    def make_llir(self, mod, metadata, opt):
        """TTIR -> C transpilation (plan P4-02).

        The per-backend ttgir->llvm-ir MLIR pass is not available (see
        make_ttgir), so the elementwise subset is source-to-source lowered
        instead: the TTIR module is serialized and transpiled by
        triton_vortex.ttir_to_c into a KMU C kernel; the returned "llir"
        artifact is that C source (kept in the cache next to the vxbin for
        reproducibility). Anything outside the whitelist raises
        NotImplementedError naming the op.
        """
        from . import ttir_to_c
        ptr_bits = int(str(opt.arch)[-2:])
        info = ttir_to_c.transpile(mod, ptr_bits=ptr_bits)
        metadata["name"] = info["name"]
        metadata["shared"] = info["lmem"]        # CTA local memory bytes
        metadata["args_size"] = info["args_size"]
        # Host packing contract: ordered [param, pack_args fmt] pairs;
        # constexprs are baked into the kernel and never appear here.
        metadata["sig"] = [[f[0], f[2]] for f in info["args"]]
        return info["c_src"]

    def make_vxbin(self, c_src, metadata, opt):
        """Compile the transpiled C with ci/hipcc_vortex.py into a real
        .vxbin written next to the cache metadata; return its bytes."""
        from . import ttir_to_c
        ptr_bits = int(str(opt.arch)[-2:])
        vxbin_path = ttir_to_c.compile_vxbin(
            c_src, metadata["name"], metadata["args_size"],
            metadata["shared"], metadata["hash"], ptr_bits=ptr_bits)
        metadata["vxbin_path"] = vxbin_path
        with open(vxbin_path, "rb") as f:
            return f.read()
