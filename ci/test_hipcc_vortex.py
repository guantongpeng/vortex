import importlib.util
import os
import unittest

SPEC = importlib.util.spec_from_file_location("hipcc_vortex", os.path.join(os.path.dirname(__file__), "hipcc_vortex.py"))
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


class HipccVortexTest(unittest.TestCase):
    def test_target_flags(self):
        args = type("Args", (), {"arch": "vortex64", "tooldir": "/opt/tools"})()
        cmd = mod.build_command(args, ["kernel.hip", "-o", "kernel.o"])
        self.assertIn("--target=riscv64-unknown-elf", cmd)
        self.assertIn("-march=rv64imafd", cmd)
        self.assertIn("-c", cmd)

    def test_bad_arch(self):
        args = type("Args", (), {"arch": "sm_80", "tooldir": "/opt/tools"})()
        with self.assertRaises(ValueError):
            mod.build_command(args, ["kernel.hip"])


if __name__ == "__main__":
    unittest.main()
