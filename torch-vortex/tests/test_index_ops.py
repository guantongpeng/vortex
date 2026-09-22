# Copyright © 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Gather and scatter use device-side index kernels with explicit validation."""

import pytest
import torch

from helpers import assert_matches_cpu, assert_rejected


def v(t):
    return t.to("vortex")


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_gather_matches_cpu(backend, dim):
    shape = (2, 3, 4)
    d = dim if dim >= 0 else len(shape) + dim
    index_shape = list(shape)
    index_shape[d] = 2
    index = torch.randint(shape[d], index_shape, dtype=torch.int64)
    x = torch.randn(shape)
    got = torch.gather(v(x), dim, v(index))
    assert_matches_cpu(got, torch.gather(x, dim, index), rtol=0, atol=0)


def test_gather_accepts_int32_and_subshapes(backend):
    x = torch.randn(2, 3, 4)
    index = torch.tensor([[[0, 1], [2, 0], [1, 2]],
                         [[1, 0], [0, 2], [2, 1]]], dtype=torch.int32)
    backend.reset_stats()
    got = torch.gather(v(x), 1, v(index))
    assert backend.stats()["launches"] == 1
    assert_matches_cpu(got, torch.gather(x, 1, index), rtol=0, atol=0)


def test_scatter_matches_cpu(backend):
    base = torch.randn(2, 3, 4)
    index = torch.tensor([[[0, 1, 0, 1], [1, 0, 1, 0]],
                         [[1, 0, 1, 0], [0, 1, 0, 1]]], dtype=torch.int64)
    src = torch.randn(2, 2, 4)
    got = torch.scatter(v(base), 1, v(index), v(src))
    assert_matches_cpu(got, torch.scatter(base, 1, index, src), rtol=0, atol=0)


def test_scatter_accepts_int32_and_submits_one_kernel(backend):
    base = torch.zeros(2, 3)
    index = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32)
    src = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    backend.reset_stats()
    got = torch.scatter(v(base), 1, v(index), v(src))
    assert backend.stats()["launches"] == 1
    assert_matches_cpu(got, torch.scatter(base, 1, index, src), rtol=0, atol=0)


def test_empty_indices_skip_kernel(backend):
    x = torch.randn(2, 3)
    index = torch.empty((2, 0), dtype=torch.int64)
    backend.reset_stats()
    got = torch.gather(v(x), 1, v(index))
    assert backend.stats()["launches"] == 0
    assert_matches_cpu(got, torch.gather(x, 1, index), rtol=0, atol=0)

    backend.reset_stats()
    got = torch.scatter(v(x), 1, v(index), v(torch.empty((2, 0))))
    assert backend.stats()["launches"] == 0
    assert_matches_cpu(got, torch.scatter(x, 1, index, torch.empty((2, 0))),
                       rtol=0, atol=0)


def test_index_ops_reject_bad_inputs(backend):
    x = v(torch.randn(2, 3))
    index = v(torch.tensor([[0, 1], [1, 2]], dtype=torch.int64))
    src = v(torch.randn(2, 2))
    bad_index = v(torch.tensor([[0, -1], [1, 2]], dtype=torch.int64))
    noncontig = v(torch.tensor([[0, 1], [1, 2]], dtype=torch.int64).t())
    f32_index = v(torch.tensor([[0.0, 1.0], [1.0, 2.0]]))
    bad_src = v(torch.randn(1, 2))

    before = backend.stats()
    with pytest.raises(RuntimeError, match="out-of-range"):
        torch.gather(x, 1, bad_index)
    after = backend.stats()
    assert after["launches"] == before["launches"] + 1
    with assert_rejected("contiguous", backend):
        torch.gather(x, 1, noncontig)
    with assert_rejected("int32 or int64", backend):
        torch.gather(x, 1, f32_index)
    with assert_rejected("same shape", backend):
        torch.scatter(x, 1, index, bad_src)
    with assert_rejected("sparse_grad", backend):
        torch.gather(x, 1, index, sparse_grad=True)
