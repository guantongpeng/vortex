# Copyright © 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reductions (W3.2).

The kernel reduces the trailing dimension of a contiguous (rows, cols) view;
a full reduction and a middle dimension are normalised into that shape on the
host with movedim+contiguous. These tests cover both normalisations, because
only exercising the trailing case would leave the interesting one untested.
"""

import pytest
import torch

from helpers import assert_matches_cpu, assert_rejected


def v(t):
    return t.to("vortex")


X2 = torch.randn(4, 6)
X3 = torch.randn(2, 3, 4)
X4 = torch.randn(2, 3, 4, 5)


@pytest.mark.parametrize("tensor", [X2, X3, X4], ids=["2d", "3d", "4d"])
@pytest.mark.parametrize("fn", ["sum", "mean"])
def test_full_reduction(backend, tensor, fn):
    assert_matches_cpu(getattr(v(tensor), fn)(), getattr(tensor, fn)(),
                       rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("tensor", [X2, X3, X4], ids=["2d", "3d", "4d"])
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_dim_reduction(backend, tensor, dim):
    if dim >= tensor.dim():
        pytest.skip("dim out of range for this tensor")
    for fn in ("sum", "mean"):
        assert_matches_cpu(getattr(v(tensor), fn)(dim=dim),
                           getattr(tensor, fn)(dim=dim), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("keepdim", [False, True])
def test_keepdim(backend, keepdim):
    got = v(X3).sum(dim=1, keepdim=keepdim)
    want = X3.sum(dim=1, keepdim=keepdim)
    assert got.shape == want.shape
    assert_matches_cpu(got, want, rtol=1e-5, atol=1e-6)


def test_amax(backend):
    assert_matches_cpu(torch.amax(v(X2)), torch.amax(X2))
    assert_matches_cpu(torch.amax(v(X3), dim=1), torch.amax(X3, dim=1))
    assert_matches_cpu(v(X2).max(), X2.max())


@pytest.mark.parametrize("values,expected", [
    ([float("-inf"), float("-inf")], float("-inf")),
    ([float("-inf"), -1.0], -1.0),
])
def test_amax_all_neg_inf(backend, values, expected):
    """The accumulator starts at -INFINITY, not a finite sentinel.

    The pool kernel used to seed with -3.4e38f, which silently dropped -inf
    inputs; a reduction seeded the same way would do it again.
    """
    t = torch.tensor([values])
    got = torch.amax(v(t), dim=1).cpu()
    assert got[0] == expected
    assert_matches_cpu(torch.amax(v(t)), torch.amax(t))


def test_amax_of_a_signed_zero_tie(backend):
    """A tie between +0.0 and -0.0 resolves to the earlier element.

    torch.amax of [0.0, -0.0] is +0.0 and of [-0.0, 0.0] is -0.0, so the
    sign follows the first element and assert_close cannot see any of it. The
    combine used `a > b`, which keeps the *later* operand on an equality, and
    those are the only inputs where `>` and `>=` differ.
    """
    from helpers import signbit_of
    for row, want_sign in (([0.0, -0.0], False), ([-0.0, 0.0], True),
                           ([-0.0, -0.0], True)):
        t = torch.tensor([row])
        got = torch.amax(backend_amax(t), dim=1)
        assert signbit_of(got)[0] is want_sign, (
            "amax%r kept the wrong zero: %r" % (row, signbit_of(got)))
        assert signbit_of(got) == signbit_of(torch.amax(t, dim=1))


def backend_amax(t):
    return t.to("vortex")


def test_amax_propagates_nan(backend):
    t = torch.tensor([[1.0, float("nan")], [3.0, 2.0]])
    assert_matches_cpu(torch.amax(v(t), dim=1), torch.amax(t, dim=1))


def test_sum_of_zeros(backend):
    assert_matches_cpu(v(torch.zeros(3, 5)).sum(), torch.zeros(3, 5).sum(),
                       rtol=0, atol=0)


def test_multiple_dims_is_refused(backend):
    """Refused by name rather than reduced one dimension at a time silently."""
    tensor = v(X3)
    with assert_rejected("over several dimensions", backend):
        tensor.sum(dim=(0, 1))


def test_dtype_override_is_refused(backend):
    tensor = v(X2)
    with assert_rejected("dtype", backend):
        tensor.sum(dtype=torch.float64)


def test_dim_out_of_range(backend):
    tensor = v(X2)
    with assert_rejected("out of range", backend):
        tensor.sum(dim=5)


def test_max_with_dim(backend):
    """max(dim=) returns the value/index pair from the argmax path."""
    got = v(X2).max(dim=1)
    want = X2.max(dim=1)
    assert torch.equal(got.indices.cpu(), want.indices)
    assert_matches_cpu(got.values, want.values, rtol=0, atol=0)
