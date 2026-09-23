# Copyright © 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import pytest
import torch


def test_integer_arange_defaults_to_int64(backend):
    out = torch.arange(2, 9, 2, device="vortex")
    assert out.dtype == torch.int64
    assert out.cpu().tolist() == [2, 4, 6, 8]


def test_float_and_int32_arange(backend):
    f = torch.arange(0.5, 3.0, 0.5, dtype=torch.float32, device="vortex")
    i = torch.arange(5, dtype=torch.int32, device="vortex")
    assert torch.allclose(f.cpu(), torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5]))
    assert i.cpu().tolist() == [0, 1, 2, 3, 4]


def test_negative_step_and_empty_skip(backend):
    backend.reset_stats()
    out = torch.arange(5, -1, -2, device="vortex")
    empty = torch.arange(2, 2, device="vortex")
    assert out.cpu().tolist() == [5, 3, 1]
    assert empty.numel() == 0
    assert backend.stats()["launches"] == 1
    assert backend.stats()["skipped_launches"] == 1


def test_arange_rejects_unsupported_dtype(backend):
    with pytest.raises(RuntimeError, match="only float32, int32 and int64"):
        torch.arange(4, dtype=torch.float64, device="vortex")


def test_integer_dtype_rejects_fractional_step(backend):
    with pytest.raises(RuntimeError, match="integer arange dtype"):
        torch.arange(0.0, 4.0, 0.5, dtype=torch.int32, device="vortex")
