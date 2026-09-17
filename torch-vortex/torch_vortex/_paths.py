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

"""Locating the source tree, the configured build tree and the kernel image.

Every lookup here is explicit and fails with the command needed to fix it.
There is deliberately no "search upward until something looks plausible"
fallback: a wrong build tree produces kernel mismatches that look like
numerical bugs, so guessing is worse than refusing.

Environment:
  VORTEX_BUILD         configured build tree (overrides discovery)
  TORCH_VORTEX_SRC     C++ extension source directory (source-tree copies only)
  TORCH_VORTEX_VXBIN   kernel image (overrides the build-tree default)
"""

import os

__all__ = ["find_repo", "find_build", "find_vxbin", "build_error"]

_BUILD_MARKERS = ("config.mk", os.path.join("sw", "runtime", "libvortex.so"))
_REPO_MARKERS = ("VX_config.toml", "sw")

_CONFIGURE_HINT = (
    "configure a build tree first, for example:\n"
    "    mkdir -p build_torch64 && cd build_torch64\n"
    "    ../configure --xlen=64 --tooldir=/data/vortex-tools\n"
    "then either run from that tree or set VORTEX_BUILD to it"
)


def build_error(msg):
    return RuntimeError("torch_vortex: " + msg)


def _package_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _has_all(directory, markers):
    return all(os.path.exists(os.path.join(directory, m)) for m in markers)


def find_repo():
    """Root of the Vortex source checkout.

    Identified by VX_config.toml + sw/, so a build-tree copy of the package
    (which has config.mk but no VX_config.toml) is not mistaken for it.
    """
    explicit = os.environ.get("TORCH_VORTEX_SRC")
    if explicit:
        if not _has_all(explicit, _REPO_MARKERS):
            raise build_error(
                "TORCH_VORTEX_SRC=%s does not look like the Vortex source tree "
                "(needs %s)" % (explicit, ", ".join(_REPO_MARKERS)))
        return os.path.abspath(explicit)

    d = _package_dir()
    for _ in range(6):
        if _has_all(d, _REPO_MARKERS):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent

    raise build_error(
        "cannot locate the Vortex source tree from %s; set TORCH_VORTEX_SRC "
        "to the checkout root" % _package_dir())


def find_build():
    """Configured build tree holding config.mk and sw/runtime/libvortex.so.

    Resolution order: VORTEX_BUILD, then a tree the package sits inside
    (build-tree copies), then the conventional sibling names.
    """
    explicit = os.environ.get("VORTEX_BUILD")
    if explicit:
        explicit = os.path.abspath(explicit)
        if not _has_all(explicit, _BUILD_MARKERS):
            raise build_error(
                "VORTEX_BUILD=%s is not a configured build tree (needs %s)"
                % (explicit, ", ".join(_BUILD_MARKERS)))
        return explicit

    # A build-tree copy of this package lives at <build>/torch-vortex/torch_vortex.
    candidate = os.path.dirname(os.path.dirname(_package_dir()))
    if _has_all(candidate, _BUILD_MARKERS):
        return candidate

    for name in ("build_torch64", "build_dl64"):
        candidate = os.path.join(find_repo(), name)
        if _has_all(candidate, _BUILD_MARKERS):
            return candidate

    raise build_error("no configured build tree found; " + _CONFIGURE_HINT)


def find_vxbin(build=None):
    """Path to the combined torch kernel image."""
    explicit = os.environ.get("TORCH_VORTEX_VXBIN")
    if explicit:
        return os.path.abspath(explicit)
    if build is None:
        build = find_build()
    return os.path.join(build, "torch-vortex", "kernels", "torch_all.vxbin")
