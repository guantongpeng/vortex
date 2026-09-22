# Copyright © 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""cat and stack use the device copy kernel rather than a host assembly."""

import pytest
import torch

from helpers import assert_matches_cpu, assert_rejected


def v(t):
    return t.to("vortex")


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_cat_matches_cpu(backend, dim):
    xs = ([torch.randn(2, 3), torch.randn(4, 3)] if dim == 0 else
          [torch.randn(2, 3), torch.randn(2, 4)])
    got = torch.cat([v(x) for x in xs], dim=dim)
    assert_matches_cpu(got, torch.cat(xs, dim=dim), rtol=0, atol=0)


def test_cat_three_dim_and_empty_piece(backend):
    xs = [torch.randn(2, 0, 5), torch.randn(2, 3, 5), torch.randn(2, 1, 5)]
    backend.reset_stats()
    got = torch.cat([v(x) for x in xs], dim=1)
    assert backend.stats()["launches"] == 2
    assert_matches_cpu(got, torch.cat(xs, dim=1), rtol=0, atol=0)


@pytest.mark.parametrize("dim", [0, 1, 2, -1])
def test_stack_matches_cpu(backend, dim):
    xs = [torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3)]
    got = torch.stack([v(x) for x in xs], dim=dim)
    assert_matches_cpu(got, torch.stack(xs, dim=dim), rtol=0, atol=0)


def test_stack_launches_once_per_input(backend):
    xs = [torch.randn(3, 2), torch.randn(3, 2), torch.randn(3, 2)]
    backend.reset_stats()
    got = torch.stack([v(x) for x in xs], dim=-1)
    assert backend.stats()["launches"] == len(xs)
    assert_matches_cpu(got, torch.stack(xs, dim=-1), rtol=0, atol=0)


def test_empty_stack_and_one_dim_cat(backend):
    xs = [torch.empty(0, 3), torch.empty(0, 3)]
    got = torch.stack([v(x) for x in xs], dim=0)
    assert got.shape == (2, 0, 3)
    assert_matches_cpu(got, torch.stack(xs, dim=0), rtol=0, atol=0)

    xs = [torch.tensor([1.0, 2.0]), torch.tensor([3.0])]
    assert_matches_cpu(torch.cat([v(x) for x in xs]), torch.cat(xs), rtol=0,
                       atol=0)


def test_concat_rejects_bad_shapes_and_layout(backend):
    cat_a = v(torch.randn(2, 3))
    cat_b = v(torch.randn(4, 4))
    stack_a = v(torch.randn(2, 3))
    stack_b = v(torch.randn(3, 3))
    noncontig = v(torch.randn(3, 2).t())
    rhs = v(torch.randn(2, 2))
    f64 = v(torch.randn(2, 3).double())

    with assert_rejected("shape mismatch", backend):
        torch.cat([cat_a, cat_b], dim=0)
    with assert_rejected("same shape", backend):
        torch.stack([stack_a, stack_b], dim=0)
    with assert_rejected("contiguous", backend):
        torch.cat([noncontig, rhs], dim=0)
    with assert_rejected("float32", backend):
        torch.cat([f64, cat_a], dim=0)
    with pytest.raises(ValueError, match="non-empty list"):
        torch.cat([], dim=0)
