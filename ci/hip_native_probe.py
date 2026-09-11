#!/usr/bin/env python3
"""Check the reproducible inputs for the native HIPVortex toolchain."""

import argparse
import json
import os
import sys


def resolve_sysroot(root, xlen):
    """Locate the RISC-V sysroot for one XLEN.

    Two layouts exist in the wild: a bare sysroot directory next to the other
    tools, and the GCC cross-toolchain layout used by this repository's
    config.mk (RISCV_SYSROOT = riscv<N>-gnu-toolchain/riscv<N>-unknown-elf).
    Return the first existing path or None.
    """
    prefix = f"riscv{xlen}-unknown-elf"
    candidates = (
        os.path.join(root, prefix),
        os.path.join(root, f"riscv{xlen}-gnu-toolchain", prefix),
    )
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return None


def probe(root):
    root = os.path.abspath(root)
    llvm = os.path.join(root, "llvm-vortex")
    kernel = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sw", "kernel"))
    sysroots = {xlen: resolve_sysroot(root, xlen) for xlen in (32, 64)}
    checks = {
        "clang": os.path.isfile(os.path.join(llvm, "bin", "clang++")),
        "lld": os.path.isfile(os.path.join(llvm, "bin", "ld.lld")),
        "llvm_objcopy": os.path.isfile(os.path.join(llvm, "bin", "llvm-objcopy")),
        "riscv32_sysroot": sysroots[32] is not None,
        "riscv64_sysroot": sysroots[64] is not None,
        # The device link step needs picolibc (libc<N>) and compiler-rt
        # (libcrt<N>); without them only -c compilation can succeed.
        "libc32": os.path.isdir(os.path.join(root, "libc32")),
        "libc64": os.path.isdir(os.path.join(root, "libc64")),
        "libcrt32": os.path.isdir(os.path.join(root, "libcrt32")),
        "libcrt64": os.path.isdir(os.path.join(root, "libcrt64")),
        "vxbin_tool": os.path.isfile(os.path.join(kernel, "scripts", "vxbin.py")),
        "link32": os.path.isfile(os.path.join(kernel, "scripts", "link32.ld")),
        "link64": os.path.isfile(os.path.join(kernel, "scripts", "link64.ld")),
    }
    hip_include = os.getenv("HIP_VORTEX_INCLUDE", "")
    checks["hip_headers"] = bool(hip_include and os.path.isfile(os.path.join(hip_include, "hip", "hip_runtime.h")))
    return {
        "schema": 2,
        "tool_root": root,
        "kernel_root": kernel,
        "checks": checks,
        "paths": {
            "clang": os.path.join(llvm, "bin", "clang++"),
            "lld": os.path.join(llvm, "bin", "ld.lld"),
            "llvm_objcopy": os.path.join(llvm, "bin", "llvm-objcopy"),
            "riscv32_sysroot": sysroots[32],
            "riscv64_sysroot": sysroots[64],
            "gcc_toolchain32": os.path.join(root, "riscv32-gnu-toolchain"),
            "gcc_toolchain64": os.path.join(root, "riscv64-gnu-toolchain"),
            "hip_include": hip_include or None,
        },
        "ready": all(checks.values()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tooldir", default=os.getenv("TOOLDIR", ""))
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--strict", action="store_true", help="return failure when an input is missing")
    args = parser.parse_args(argv)
    result = probe(args.tooldir)
    if args.as_json:
        print(json.dumps(result, sort_keys=True, indent=2))
    else:
        for name, present in result["checks"].items():
            print(f"{'OK' if present else 'MISSING':7} {name}")
        print(f"native_hip_ready: {'yes' if result['ready'] else 'no'}")
    return 0 if result["ready"] or not args.strict else 1


if __name__ == "__main__":
    sys.exit(main())
