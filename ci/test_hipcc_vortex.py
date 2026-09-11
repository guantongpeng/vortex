import importlib.util
import os
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("hipcc_vortex", os.path.join(os.path.dirname(__file__), "hipcc_vortex.py"))
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def make_args(arch):
    return type("Args", (), {
        "arch": arch,
        "tooldir": "/opt/tools",
        "build_dir": "/opt/build",
        "configs": "",
        "print_command": False,
        "dry_run": False,
    })()


def fake_status():
    # Minimal probe result with every resolved path set.
    return {
        "tool_root": "/opt/tools",
        "checks": {k: True for k in (
            "clang", "lld", "llvm_objcopy", "hip_headers",
            "riscv32_sysroot", "riscv64_sysroot",
            "libc32", "libc64", "libcrt32", "libcrt64",
            "vxbin_tool", "link32", "link64")},
        "paths": {
            "clang": "/opt/tools/llvm-vortex/bin/clang++",
            "lld": "/opt/tools/llvm-vortex/bin/ld.lld",
            "llvm_objcopy": "/opt/tools/llvm-vortex/bin/llvm-objcopy",
            "riscv32_sysroot": "/opt/tools/riscv32-gnu-toolchain/riscv32-unknown-elf",
            "riscv64_sysroot": "/opt/tools/riscv64-gnu-toolchain/riscv64-unknown-elf",
            "gcc_toolchain32": "/opt/tools/riscv32-gnu-toolchain",
            "gcc_toolchain64": "/opt/tools/riscv64-gnu-toolchain",
            "hip_include": "/opt/tools/hip/include",
        },
    }


class HipccVortexTest(unittest.TestCase):
    def test_target_flags(self):
        cmd = mod.build_command(make_args("vortex64"),
                                ["kernel.hip", "-c", "-o", "kernel.o"],
                                fake_status())
        self.assertIn("--target=riscv64-unknown-elf", cmd)
        self.assertIn("-march=rv64imafd", cmd)
        self.assertIn("-mabi=lp64d", cmd)
        self.assertIn("-x", cmd)
        self.assertEqual(cmd[cmd.index("-x") + 1], "c++")
        self.assertIn("-c", cmd)

    def test_rv32_flags(self):
        cmd = mod.build_command(make_args("vortex32"), ["kernel.hip"],
                                fake_status())
        self.assertIn("-march=rv32imaf", cmd)
        self.assertIn("-mabi=ilp32f", cmd)

    def test_bad_arch(self):
        with self.assertRaises(ValueError):
            mod.build_command(make_args("sm_80"), ["kernel.hip"], fake_status())

    def test_sysroot_and_hip_include_resolved_from_probe(self):
        cmd = mod.build_command(make_args("vortex64"), ["kernel.hip"],
                                fake_status())
        self.assertIn("--sysroot=/opt/tools/riscv64-gnu-toolchain/riscv64-unknown-elf", cmd)
        self.assertIn("-I/opt/tools/hip/include", cmd)

    def test_link_command_shape(self):
        cmd = mod.flatten(mod.link_command(
            make_args("vortex64"), ["startup.o", "kernel.o"], "app.elf",
            fake_status()))
        self.assertIn("startup.o", cmd)
        self.assertIn("kernel.o", cmd)
        self.assertIn("/opt/build/sw/kernel/libvortex.a", cmd)
        self.assertIn("-Wl,-T," + os.path.join(mod.REPO_ROOT, "sw", "kernel", "scripts", "link64.ld"), cmd)
        self.assertTrue(any("libclang_rt.builtins-riscv64.a" in a for a in cmd))

    def test_check_inputs_rejects_missing_sysroot(self):
        status = fake_status()
        status["checks"]["riscv64_sysroot"] = False
        self.assertFalse(mod.check_inputs(status, "64", False))


if __name__ == "__main__":
    unittest.main()
