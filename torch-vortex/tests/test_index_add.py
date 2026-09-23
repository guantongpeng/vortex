# Copyright © 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""index_add uses a deterministic scan so duplicate indices remain correct."""

import pytest
import torch

from helpers import assert_matches_cpu, assert_rejected


def v(t):
    return t.to("vortex")


def test_index_add_matches_cpu_with_duplicate_indices(backend):
    self_cpu = torch.randn(2, 4, 3)
    index_cpu = torch.tensor([1, 1, 3], dtype=torch.int64)
    source_cpu = torch.randn(2, 3, 3)
    alpha = 0.5
    backend.reset_stats()
    got = torch.index_add(v(self_cpu), 1, v(index_cpu), v(source_cpu), alpha=alpha)
    assert backend.stats()["launches"] == 1
    want = torch.index_add(self_cpu, 1, index_cpu, source_cpu, alpha=alpha)
    assert_matches_cpu(got, want, rtol=1e-6, atol=1e-6)


def test_index_add_out_and_inplace_match_cpu(backend):
    self_cpu = torch.randn(2, 3)
    index_cpu = torch.tensor([1, 1, 2], dtype=torch.int64)
    source_cpu = torch.randn(2, 3)
    out = torch.empty_like(self_cpu).to("vortex")
    got = torch.index_add(v(self_cpu), 1, v(index_cpu), v(source_cpu),
                          alpha=0.5, out=out)
    want = torch.index_add(self_cpu, 1, index_cpu, source_cpu, alpha=0.5)
    assert got.data_ptr() == out.data_ptr()
    assert_matches_cpu(got, want)

    inplace = v(self_cpu.clone())
    before = inplace._version
    inplace.index_add_(1, v(index_cpu), v(source_cpu), alpha=0.5)
    assert inplace._version > before
    assert_matches_cpu(inplace, want)


def test_index_add_accepts_int32_and_negative_dim(backend):
    self_cpu = torch.zeros(3, 5)
    index_cpu = torch.tensor([0, 4, 0], dtype=torch.int32)
    source_cpu = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    got = torch.index_add(v(self_cpu), -1, v(index_cpu), v(source_cpu))
    want = torch.index_add(self_cpu, -1, index_cpu, source_cpu)
    assert_matches_cpu(got, want, rtol=0, atol=0)


def test_index_add_empty_index_copies_self(backend):
    self_cpu = torch.randn(2, 3)
    index_cpu = torch.empty(0, dtype=torch.int64)
    source_cpu = torch.empty(2, 0)
    backend.reset_stats()
    got = torch.index_add(v(self_cpu), 1, v(index_cpu), v(source_cpu))
    assert backend.stats()["launches"] == 0
    assert_matches_cpu(got, self_cpu, rtol=0, atol=0)


def test_index_add_rejects_bad_shapes_and_values(backend):
    self_v = v(torch.zeros(2, 4))
    index_v = v(torch.tensor([0, 1], dtype=torch.int64))
    source_v = v(torch.ones(2, 2))
    bad_index_v = v(torch.tensor([0, 4], dtype=torch.int64))
    bad_rank_v = v(torch.tensor([[0, 1]], dtype=torch.int64))
    bad_rank_source_v = v(torch.ones(2, 1))
    bad_source_v = v(torch.ones(2, 3))
    f64_source_v = v(torch.ones(2, 2).double())

    before = backend.stats()
    with pytest.raises(RuntimeError, match="out-of-range"):
        torch.index_add(self_v, 1, bad_index_v, source_v)
    after = backend.stats()
    assert after["launches"] == before["launches"] + 1
    with assert_rejected("one-dimensional", backend):
        torch.index_add(self_v, 1, bad_rank_v, bad_rank_source_v)
    with assert_rejected("dimension must equal index length", backend):
        torch.index_add(self_v, 1, index_v, bad_source_v)
    with assert_rejected("float32", backend):
        torch.index_add(self_v, 1, index_v, f64_source_v)
