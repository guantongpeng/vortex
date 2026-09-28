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

"""Environment manifest: what a test result actually depended on.

Every pytest session writes one of these (see tests/conftest.py) so that a
pass or a failure can always be traced back to an interpreter, a torch build,
a build tree and a kernel image. W0.1 of docs/mydocs/05_pytorch/torch_plan.md.

Deliberately does not import torch at module level: `collect()` needs it, but
this module is also used to describe an environment in which torch is missing
or broken.
"""

import hashlib
import json
import os
import platform
import re
import subprocess
import sys

from .._paths import find_build, find_repo, find_vxbin, build_error

SCHEMA = 1

# Sourced from <build>/config.mk and the repo VERSION file, so a manifest is
# always traceable to the values the kernel Makefile consumed.
_CONFIG_KEYS = {
    "xlen": "XLEN",
    "tooldir": "TOOLDIR",
}
_VERSION_KEYS = ("VORTEX_VERSION", "TOOLCHAIN_REV", "GEM5_REV")


def _read_makefile_vars(path):
    """Pull simple `KEY ?= value` / `KEY=value` assignments out of a makefile."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            m = re.match(r"\s*([A-Z_][A-Z0-9_]*)\s*\??=\s*(.*?)\s*$", line)
            if m and not line.lstrip().startswith("#"):
                out[m.group(1)] = m.group(2)
    return out


def _sha256(path):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(args, cwd):
    try:
        out = subprocess.run(["git"] + list(args), cwd=cwd, capture_output=True,
                             text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _compiler_id():
    """The compiler that will build the extension, and its C++11 ABI flag."""
    cxx = os.environ.get("CXX", "c++")
    version = None
    abi = None
    try:
        out = subprocess.run((cxx, "--version"), capture_output=True, text=True,
                             timeout=30)
        if out.returncode == 0:
            version = out.stdout.splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    # _GLIBCXX_USE_CXX11_ABI is what actually decides whether a prebuilt
    # extension can be loaded, and it is not visible from the version string.
    try:
        src = "#include <string>\n_GLIBCXX_USE_CXX11_ABI_MARKER\n"
        out = subprocess.run((cxx, "-x", "c++", "-E", "-dM", "-"), input=src,
                             capture_output=True, text=True, timeout=30)
        m = re.search(r"#define _GLIBCXX_USE_CXX11_ABI (\d+)", out.stdout)
        if m:
            abi = int(m.group(1))
    except (OSError, subprocess.SubprocessError):
        pass
    return {"cxx_id": cxx, "cxx_version": version, "_GLIBCXX_USE_CXX11_ABI": abi}


def _optional_module(name):
    try:
        mod = __import__(name)
    except Exception:  # noqa: BLE001 - any failure means "not usable here"
        return None
    return getattr(mod, "__version__", "unknown")


def collect(build=None, vxbin=None, ext_module=None):
    """Gather the manifest as a plain dict. Never raises for a missing piece."""
    try:
        repo = find_repo()
    except RuntimeError as exc:
        repo = None
        repo_error = str(exc)
    else:
        repo_error = None

    try:
        build = build or find_build()
    except RuntimeError as exc:
        build = None
        build_error_msg = str(exc)
    else:
        build_error_msg = None

    cfg = _read_makefile_vars(os.path.join(build, "config.mk")) if build else {}
    ver = _read_makefile_vars(os.path.join(repo, "VERSION")) if repo else {}

    observed = {
        "python": platform.python_version(),
        "python_abi": "cp%d%d" % (sys.version_info[0], sys.version_info[1]),
        "interpreter_realpath": os.path.realpath(sys.executable),
        "torch": _optional_module("torch"),
        "triton": _optional_module("triton"),
        "numpy": _optional_module("numpy"),
        "torchvision": _optional_module("torchvision"),
        "platform": platform.platform(),
        "kernel_release": platform.release(),
    }
    if observed["torch"]:
        import torch
        observed["torch_cuda"] = torch.version.cuda
        observed["torch_hip"] = getattr(torch.version, "hip", None)
        observed["torch_git_version"] = getattr(torch.version, "git_version", None)
    observed.update(_compiler_id())

    vxbin = vxbin or (find_vxbin(build) if build else None)
    ext_path = None
    if ext_module is not None:
        ext_path = getattr(ext_module, "__file__", None)

    return {
        "schema": SCHEMA,
        "observed": observed,
        "build": {
            "build_dir": build,
            "discovery_error": build_error_msg,
            "xlen": int(cfg["XLEN"]) if cfg.get("XLEN", "").isdigit() else None,
            "tooldir": cfg.get("TOOLDIR"),
            "configs": os.environ.get("CONFIGS"),
            "driver": os.environ.get("VORTEX_DRIVER"),
            "vortex_version": ver.get("VORTEX_VERSION"),
            "toolchain_rev": ver.get("TOOLCHAIN_REV"),
            "gem5_rev": ver.get("GEM5_REV"),
        },
        "source": {
            "repo_root": repo,
            "discovery_error": repo_error,
            "git_commit": _git(["rev-parse", "HEAD"], repo) if repo else None,
            "git_dirty": bool(_git(["status", "--porcelain"], repo)) if repo else None,
        },
        "artifacts": {
            "vxbin_path": vxbin,
            "vxbin_sha256": _sha256(vxbin),
            "ext_so_path": ext_path,
            "ext_sha256": _sha256(ext_path),
        },
    }


def write(path=None, **kwargs):
    """Collect and write the manifest, returning (manifest, path)."""
    if path is None:
        build = kwargs.get("build") or os.environ.get("VORTEX_BUILD")
        if build is None:
            try:
                build = find_build()
            except RuntimeError as exc:
                raise build_error("cannot place the manifest: %s" % exc)
        path = os.path.join(build, "torch-vortex", "env_manifest.json")
    manifest = collect(**kwargs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest, path


def summarize(manifest):
    """One-line human summary for pytest headers and CI logs."""
    o = manifest.get("observed", {})
    b = manifest.get("build", {})
    a = manifest.get("artifacts", {})
    vx = a.get("vxbin_sha256")
    return (
        "python={python} abi={python_abi} torch={torch} triton={triton} "
        "cxx_abi={_GLIBCXX_USE_CXX11_ABI} | xlen={xlen} driver={driver} "
        "tooldir={tooldir} | vxbin={vx}".format(
            vx=(vx[:12] + "…" if vx else None), **{**o, **b}))
