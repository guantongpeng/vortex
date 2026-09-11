#!/usr/bin/env python3
"""Check the reproducible inputs for the native HIPVortex toolchain."""

import argparse
import json
import os
import sys


def probe(root):
    root = os.path.abspath(root)
    llvm = os.path.join(root, "llvm-vortex")
    kernel = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sw", "kernel"))
    checks = {
        "clang": os.path.isfile(os.path.join(llvm, "bin", "clang++")),
        "lld": os.path.isfile(os.path.join(llvm, "bin", "ld.lld")),
        "llvm_objcopy": os.path.isfile(os.path.join(llvm, "bin", "llvm-objcopy")),
        "riscv32_sysroot": os.path.isdir(os.path.join(root, "riscv32-unknown-elf")),
        "riscv64_sysroot": os.path.isdir(os.path.join(root, "riscv64-unknown-elf")),
        "vxbin_tool": os.path.isfile(os.path.join(kernel, "scripts", "vxbin.py")),
        "link32": os.path.isfile(os.path.join(kernel, "scripts", "link32.ld")),
        "link64": os.path.isfile(os.path.join(kernel, "scripts", "link64.ld")),
    }
    hip_include = os.getenv("HIP_VORTEX_INCLUDE", "")
    checks["hip_headers"] = bool(hip_include and os.path.isfile(os.path.join(hip_include, "hip", "hip_runtime.h")))
    return {
        "schema": 1,
        "tool_root": root,
        "kernel_root": kernel,
        "checks": checks,
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
