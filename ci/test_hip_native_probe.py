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

    def test_fake_toolchain_is_detected(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as hip:
            for rel in ("bin/clang++", "bin/ld.lld", "bin/llvm-objcopy", "riscv32-unknown-elf", "riscv64-unknown-elf"):
                path = os.path.join(root, "llvm-vortex", rel) if rel.startswith("bin/") else os.path.join(root, rel)
                if rel.startswith("bin/"):
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    open(path, "w").close()
                else:
                    os.makedirs(path)
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
