#!/usr/bin/env python3
"""Native HIPVortex device compiler wrapper.

Two operations, both reproducible:

1. compile: one HIP translation unit -> RISC-V object
   hipcc-vortex --offload-arch=vortex64 kernel.hip -c -o kernel.o

2. link: objects -> ELF -> .vxbin (the Vortex kernel image)
   hipcc-vortex --offload-arch=vortex64 kernel.o -o kernel.vxbin

The link reproduces tests/kernel/common.mk: two-pass vx_start.S compilation
(feature detection via kernel_startup.sh), static link against libvortex.a,
picolibc and compiler-rt with sw/kernel/scripts/link<N>.ld, then vxbin.py
for the image + VXSYMTAB footer.

HIP sources are compiled as C++ (-x c++) against
third_party/hip-vortex/include: VOLT clang's -x hip mode injects
AMDGPU-specific options that are invalid for the RISC-V target, and the
header-only mapping keeps the toolchain patch-free. Triple-chevron
kernel<<<>>> syntax is therefore not yet available; use
hipLaunchKernelGGL.
"""

import argparse
import os
import subprocess
import sys

from hip_native_probe import probe

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_HIP_INCLUDE = os.path.join(REPO_ROOT, "third_party", "hip-vortex", "include")
STARTUP_ADDR = "0x80000000"

REQUIRED_INPUTS = (
    "clang",
    "lld",
    "llvm_objcopy",
    "vxbin_tool",
    "hip_headers",
)


def arch_params(arch):
    if arch not in ("vortex32", "vortex64"):
        raise ValueError("--offload-arch must be vortex32 or vortex64")
    xlen = arch[-2:]
    return xlen, {
        "target": f"riscv{xlen}-unknown-elf",
        "march": "rv32imaf" if xlen == "32" else "rv64imafd",
        "mabi": "ilp32f" if xlen == "32" else "lp64d",
        "sysroot": f"riscv{xlen}_sysroot",
        "gcc_toolchain": f"gcc_toolchain{xlen}",
        "link_script": os.path.join(REPO_ROOT, "sw", "kernel", "scripts", f"link{xlen}.ld"),
        "check": f"link{xlen}",
    }


def check_inputs(status, xlen, link):
    for key in REQUIRED_INPUTS:
        if not status["checks"][key]:
            print(f"hipcc-vortex: missing native input: {key}", file=sys.stderr)
            return False
    sysroot_key = f"riscv{xlen}_sysroot"
    if not status["checks"][sysroot_key]:
        print(f"hipcc-vortex: missing native input: {sysroot_key}", file=sys.stderr)
        return False
    if link:
        if not status["checks"][f"libc{xlen}"] or not status["checks"][f"libcrt{xlen}"]:
            print(f"hipcc-vortex: missing libc{xlen}/libcrt{xlen} for device link",
                  file=sys.stderr)
            return False
        if not status["checks"][f"link{xlen}"]:
            print(f"hipcc-vortex: missing link script for xlen {xlen}", file=sys.stderr)
            return False
    return True


def gen_config_flags(build_dir, xlen, configs):
    """Mirror the kernel Makefile: expand VX_config.toml into -DVX_CFG_* flags."""
    script = os.path.join(REPO_ROOT, "ci", "gen_config.py")
    toml = os.path.join(REPO_ROOT, "VX_config.toml")
    cflags = f"{configs} -DVX_CFG_XLEN={xlen}"
    out = subprocess.check_output(
        [sys.executable, script, "--config=" + toml, "--cflags=" + cflags],
        universal_newlines=True)
    return out.split()


def build_command(args, extra, status):
    """Compile command for one HIP translation unit."""
    xlen, p = arch_params(args.arch)
    paths = status["paths"]
    command = [
        paths["clang"],
        "--target=" + p["target"],
        "--sysroot=" + paths[p["sysroot"]],
        "--gcc-toolchain=" + paths[p["gcc_toolchain"]],
        "-Xclang", "-target-feature", "-Xclang", "+xvortex",
        "-march=" + p["march"],
        "-mabi=" + p["mabi"],
        "-mcmodel=medany",
        "-fno-exceptions",
        "-fdata-sections", "-ffunction-sections",
        "-D__VORTEX__",
        "-I" + (paths["hip_include"] or DEFAULT_HIP_INCLUDE),
        "-I" + os.path.join(REPO_ROOT, "sw", "kernel", "include"),
        "-I" + os.path.join(REPO_ROOT, "sw"),
        "-I" + os.path.join(args.build_dir, "sw"),
        "-O3",
    ]
    command.extend(gen_config_flags(args.build_dir, xlen, args.configs))
    # Force C++ mode: VOLT clang's HIP mode injects AMDGPU-only driver
    # options invalid for RISC-V, and the native path maps the HIP surface
    # through third_party/hip-vortex headers instead. Without this, the
    # .hip extension alone switches clang into -x hip.
    command.extend(["-x", "c++"])
    command.extend(extra)
    return command


def link_command(args, objects, elf_path, status):
    xlen, p = arch_params(args.arch)
    paths = status["paths"]
    kernel_lib = os.path.join(args.build_dir, "sw", "kernel", "libvortex.a")
    libc = os.path.join(status["tool_root"], f"libc{xlen}")
    libcrt = os.path.join(status["tool_root"], f"libcrt{xlen}")
    command = [
        paths["clang"],
        "--target=" + p["target"],
        "--sysroot=" + paths[p["sysroot"]],
        "--gcc-toolchain=" + paths[p["gcc_toolchain"]],
        "-Xclang", "-target-feature", "-Xclang", "+xvortex",
        "-march=" + p["march"],
        "-mabi=" + p["mabi"],
        "-mcmodel=medany",
        "-nostartfiles", "-nostdlib",
        objects,
        kernel_lib,
        "-Wl,-Bstatic,--gc-sections",
        "-Wl,-T," + p["link_script"],
        "-Wl,--defsym=STARTUP_ADDR=" + STARTUP_ADDR,
        "-L" + os.path.join(libc, "lib"), "-lm", "-lc",
        os.path.join(libcrt, "lib", "baremetal",
                     f"libclang_rt.builtins-riscv{xlen}.a"),
        "-o", elf_path,
    ]
    return command


def flatten(command):
    return [x for item in command for x in (item if isinstance(item, list) else [item])]


def run(command, dry, print_command):
    command = flatten(command)
    if print_command or dry:
        print(" ".join(subprocess.list2cmdline([x]) for x in command))
    if dry:
        return 0
    return subprocess.call(command)


def link_vxbin(args, objects, output, status):
    """Two-pass startup detection, static link, then vxbin conversion."""
    xlen, _ = arch_params(args.arch)
    paths = status["paths"]
    startup_src = os.path.join(REPO_ROOT, "sw", "kernel", "src", "vx_start.S")
    startup_sh = os.path.join(REPO_ROOT, "sw", "kernel", "scripts", "kernel_startup.sh")
    build_dir = args.build_dir

    startup_flags = ["-DNEED_GP", "-DNEED_TLS", "-DNEED_INITFINI"]

    def compile_startup(out_path, flags):
        xlen_, p = arch_params(args.arch)
        cmd = [
            paths["clang"],
            "--target=" + p["target"],
            "--sysroot=" + paths[p["sysroot"]],
            "--gcc-toolchain=" + paths[p["gcc_toolchain"]],
            "-Xclang", "-target-feature", "-Xclang", "+xvortex",
            "-march=" + p["march"], "-mabi=" + p["mabi"], "-mcmodel=medany",
            "-D__VORTEX__",
            "-I" + os.path.join(REPO_ROOT, "sw", "kernel", "include"),
            "-I" + os.path.join(REPO_ROOT, "sw"),
            "-I" + os.path.join(build_dir, "sw"),
        ]
        cmd.extend(gen_config_flags(build_dir, xlen_, args.configs))
        cmd.extend(flags)
        cmd.extend(["-c", startup_src, "-o", out_path])
        return run(cmd, args.dry_run, args.print_command)

    tmp_base = output + ".hipcc_tmp"
    startup1 = tmp_base + ".startup1.o"
    probe_elf = tmp_base + ".probe.elf"
    startup2 = tmp_base + ".startup.o"
    elf_path = output if output.endswith(".elf") else tmp_base + ".elf"

    if compile_startup(startup1, startup_flags) != 0:
        return 1
    # Link once with the default startup to detect GP/TLS/INITFINI needs.
    if run(link_command(args, [startup1] + objects, probe_elf, status),
           args.dry_run, args.print_command) != 0:
        return 1
    if not args.dry_run:
        detected = subprocess.check_output(
            ["bash", startup_sh, paths["llvm_objcopy"], probe_elf],
            universal_newlines=True).split()
        if detected:
            startup_flags = detected
    if compile_startup(startup2, startup_flags) != 0:
        return 1
    if run(link_command(args, [startup2] + objects, elf_path, status),
           args.dry_run, args.print_command) != 0:
        return 1

    if elf_path != output:
        vxbin = os.path.join(REPO_ROOT, "sw", "kernel", "scripts", "vxbin.py")
        env = dict(os.environ, OBJCOPY=paths["llvm_objcopy"])
        cmd = [sys.executable, vxbin, elf_path, output]
        if args.print_command or args.dry_run:
            print(" ".join(subprocess.list2cmdline(cmd)))
        if args.dry_run:
            return 0
        rc = subprocess.call(cmd, env=env)
        if rc == 0:
            for path in (startup1, probe_elf, startup2, elf_path,
                         output + ".hipcc.o"):
                if os.path.exists(path):
                    os.remove(path)
        return rc
    return 0


def find_output(extra):
    if "-o" in extra:
        return extra[extra.index("-o") + 1]
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offload-arch", dest="arch", required=True)
    parser.add_argument("--tooldir", default=os.getenv("TOOLDIR", ""))
    parser.add_argument("--build-dir", default=os.getenv("VORTEX_BUILD_DIR", "build"),
                        help="configured Vortex build directory (VX_types.h, libvortex.a)")
    parser.add_argument("--configs", default="",
                        help="extra -DVX_CFG_* overrides passed to gen_config.py")
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args(argv)
    if not extra:
        parser.error("an input HIP source and compiler options are required")
    args.build_dir = os.path.abspath(args.build_dir)

    status = probe(args.tooldir)
    output = find_output(extra)
    inputs = [a for a in extra if a not in ("-o", output) and not a.startswith("-")]
    sources = [a for a in inputs if a.endswith((".hip", ".cpp", ".cc", ".c"))]
    objects = [a for a in inputs if a.endswith((".o", ".a"))]
    wants_link = bool(objects) or (output or "").endswith(".vxbin") or (
        bool(sources) and output and not output.endswith(".o") and "-c" not in extra)

    if not check_inputs(status, args.arch[-2:], wants_link):
        return 1
    if wants_link:
        if not (os.path.isfile(os.path.join(args.build_dir, "sw", "kernel", "libvortex.a"))):
            print(f"hipcc-vortex: {args.build_dir} has no sw/kernel/libvortex.a; "
                  "run configure + make -C sw/kernel first", file=sys.stderr)
            return 1
        # Compile any remaining sources to a temporary object first.
        if sources:
            tmp_obj = (output or "a.vxbin") + ".hipcc.o"
            cmd = build_command(args, sources + ["-c", "-o", tmp_obj], status)
            rc = run(cmd, args.dry_run, args.print_command)
            if rc != 0:
                return rc
            objects = objects + [tmp_obj]
        return link_vxbin(args, objects, output, status)
    return run(build_command(args, extra, status), args.dry_run, args.print_command)


if __name__ == "__main__":
    sys.exit(main())
