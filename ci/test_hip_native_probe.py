import importlib.util
import os
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("hip_native_probe", os.path.join(os.path.dirname(__file__), "hip_native_probe.py"))
probe_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe_mod)


class HipNativeProbeTest(unittest.TestCase):
    def test_missing_root_is_reported_without_strict_failure(self):
        result = probe_mod.probe("/path/that/does/not/exist")
        self.assertFalse(result["ready"])
        self.assertFalse(result["checks"]["clang"])
        self.assertEqual(result["schema"], 2)

    def test_fake_toolchain_is_detected(self):
        self._check_ready_with_sysroots(lambda root, xlen: os.path.join(root, f"riscv{xlen}-unknown-elf"))

    def test_gcc_toolchain_sysroot_layout_is_detected(self):
        # config.mk layout: riscv<N>-gnu-toolchain/riscv<N>-unknown-elf.
        self._check_ready_with_sysroots(
            lambda root, xlen: os.path.join(root, f"riscv{xlen}-gnu-toolchain", f"riscv{xlen}-unknown-elf"))

    def test_paths_report_resolved_sysroots(self):
        with tempfile.TemporaryDirectory() as root:
            nested = os.path.join(root, "riscv64-gnu-toolchain", "riscv64-unknown-elf")
            os.makedirs(nested)
            result = probe_mod.probe(root)
            self.assertEqual(result["paths"]["riscv64_sysroot"], nested)
            self.assertIsNone(result["paths"]["riscv32_sysroot"])

    def _check_ready_with_sysroots(self, sysroot_path):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as hip:
            for rel in ("bin/clang++", "bin/ld.lld", "bin/llvm-objcopy"):
                path = os.path.join(root, "llvm-vortex", rel)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                open(path, "w").close()
            for xlen in (32, 64):
                os.makedirs(sysroot_path(root, xlen))
                os.makedirs(os.path.join(root, f"libc{xlen}"))
                os.makedirs(os.path.join(root, f"libcrt{xlen}"))
            os.makedirs(os.path.join(hip, "hip"))
            open(os.path.join(hip, "hip", "hip_runtime.h"), "w").close()
            old = os.environ.get("HIP_VORTEX_INCLUDE")
            os.environ["HIP_VORTEX_INCLUDE"] = hip
            try:
                result = probe_mod.probe(root)
            finally:
                if old is None:
                    os.environ.pop("HIP_VORTEX_INCLUDE", None)
                else:
                    os.environ["HIP_VORTEX_INCLUDE"] = old
            self.assertTrue(result["ready"])


if __name__ == "__main__":
    unittest.main()
