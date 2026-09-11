#!/usr/bin/env python3
"""Minimal native HIPVortex device compiler wrapper.

The wrapper intentionally handles one reproducible operation first: compiling
one HIP translation unit to a RISC-V object. Linking and device bitcode are
separate milestones and are rejected until their inputs are installed.
"""

import argparse
import os
import subprocess
import sys

from hip_native_probe import probe


def build_command(args, extra):
    if args.arch not in ("vortex32", "vortex64"):
        raise ValueError("--offload-arch must be vortex32 or vortex64")
    xlen = args.arch[-2:]
    target = f"riscv{xlen}-unknown-elf"
    march = "rv32imaf" if xlen == "32" else "rv64imafd"
    clang = os.path.join(args.tooldir, "llvm-vortex", "bin", "clang++")
    command = [clang, "--target=" + target, "-x", "hip", "-Xclang", "-target-feature",
               "-Xclang", "+xvortex", "-march=" + march, "-O3", "-c"]
    command.extend(extra)
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offload-arch", dest="arch", required=True)
    parser.add_argument("--tooldir", default=os.getenv("TOOLDIR", ""))
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args(argv)
    if not extra:
        parser.error("an input HIP source and compiler options are required")
    status = probe(args.tooldir)
    for key in ("clang", "riscv32_sysroot", "riscv64_sysroot"):
        if not status["checks"][key]:
            print(f"hipcc-vortex: missing native input: {key}", file=sys.stderr)
            return 1
    command = build_command(args, extra)
    if args.print_command or args.dry_run:
        print(" ".join(subprocess.list2cmdline([x]) for x in command))
    if args.dry_run:
        return 0
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
