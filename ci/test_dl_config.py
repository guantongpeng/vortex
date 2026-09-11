#!/usr/bin/env python3
"""Unit tests for the DL configuration matrix renderer."""

import json
import pathlib
import tempfile
import unittest

import dl_config


class DlConfigTest(unittest.TestCase):
    def test_repository_matrix_has_expected_profiles(self):
        matrix = dl_config.load_matrix(dl_config.DEFAULT_MATRIX)
        self.assertEqual(
            [item["name"] for item in matrix["profiles"]],
            ["dl_functional", "dl_rtl", "dl_fpga"],
        )

    def test_render_does_not_override_xlen(self):
        matrix = dl_config.load_matrix(dl_config.DEFAULT_MATRIX)
        flags, configs = dl_config.render(dl_config.profile(matrix, "dl_rtl"), 64)
        self.assertIn("--threads=16", flags)
        self.assertNotIn("-DVX_CFG_XLEN=64", configs)

    def test_invalid_profile_is_rejected(self):
        matrix = dl_config.load_matrix(dl_config.DEFAULT_MATRIX)
        with self.assertRaises(ValueError):
            dl_config.profile(matrix, "missing")

    def test_duplicate_profile_is_rejected(self):
        matrix = {
            "schema_version": 1,
            "profiles": [
                {
                    "name": "same",
                    "xlen": [32],
                    "shape": {
                        "clusters": 1,
                        "cores": 1,
                        "warps": 1,
                        "threads": 1,
                        "l2cache": False,
                        "l3cache": False,
                    },
                    "configs": [],
                },
                {
                    "name": "same",
                    "xlen": [32],
                    "shape": {
                        "clusters": 1,
                        "cores": 1,
                        "warps": 1,
                        "threads": 1,
                        "l2cache": False,
                        "l3cache": False,
                    },
                    "configs": [],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "matrix.json"
            path.write_text(json.dumps(matrix), encoding="utf-8")
            with self.assertRaises(ValueError):
                dl_config.load_matrix(path)

    def test_unknown_config_name_is_rejected(self):
        matrix = {
            "schema_version": 1,
            "profiles": [
                {
                    "name": "bad-config",
                    "xlen": [32],
                    "shape": {
                        "clusters": 1,
                        "cores": 1,
                        "warps": 1,
                        "threads": 1,
                        "l2cache": False,
                        "l3cache": False,
                    },
                    "configs": ["-DVX_CFG_DOES_NOT_EXIST"],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "matrix.json"
            path.write_text(json.dumps(matrix), encoding="utf-8")
            with self.assertRaises(ValueError):
                dl_config.load_matrix(path)


if __name__ == "__main__":
    unittest.main()
