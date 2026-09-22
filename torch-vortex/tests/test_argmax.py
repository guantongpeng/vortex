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

"""argmax and max(dim=) (W3.2).

One DL pass produces the index and the extreme it points at, so max(dim=)'s
pair costs no more than argmax's index alone. The properties worth pinning are
the two that a reduction can get wrong without looking wrong: ties go to the
earliest element, and the value max(dim=) returns must be the one *at the index
it reports* -- not merely a correct maximum.
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
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_argmax_over_every_dim(backend, tensor, dim):
    if dim >= tensor.dim():
        pytest.skip("dim out of range for this tensor")
    got = torch.argmax(v(tensor), dim=dim)
    want = torch.argmax(tensor, dim=dim)
    assert got.dtype == torch.int64, "indices must be int64, got %s" % got.dtype
    assert got.shape == want.shape
    assert torch.equal(got.cpu(), want), "indices differ from CPU"


@pytest.mark.parametrize("tensor", [X2, X3, X4], ids=["2d", "3d", "4d"])
def test_max_dim_matches_cpu(backend, tensor):
    got = torch.max(v(tensor), dim=-1)
    want = torch.max(tensor, dim=-1)
    assert got.indices.dtype == torch.int64
    assert torch.equal(got.indices.cpu(), want.indices)
    assert_matches_cpu(got.values, want.values, rtol=0, atol=0)


@pytest.mark.parametrize("keepdim", [False, True])
def test_keepdim(backend, keepdim):
    got = torch.argmax(v(X3), dim=1, keepdim=keepdim)
    want = torch.argmax(X3, dim=1, keepdim=keepdim)
    assert got.shape == want.shape
    gotm = torch.max(v(X3), dim=1, keepdim=keepdim)
    assert gotm.values.shape == gotm.indices.shape == want.shape


def test_full_reduction_gives_a_zero_dim_index(backend):
    """No dim means the whole tensor, and the answer is 0-dimensional."""
    got = torch.argmax(v(X3))
    want = torch.argmax(X3)
    assert got.dim() == 0, "argmax() came back as shape %s" % (tuple(got.shape),)
    assert got.shape == want.shape
    assert torch.equal(got.cpu(), want)
    assert torch.equal(torch.argmax(v(torch.tensor([1.0, 0.0, 2.0]))).cpu(),
                       torch.tensor(2))
    # a 1-element tensor has one index, and it is 0
    assert torch.argmax(v(torch.tensor([7.0]))).cpu().item() == 0


@pytest.mark.parametrize("dim", [0, -1])
def test_zero_dim_accepts_zero_and_minus_one(backend, dim):
    x = torch.tensor(7.0)
    got = torch.argmax(v(x), dim=dim)
    want = torch.argmax(x, dim=dim)
    assert got.dim() == 0 and torch.equal(got.cpu(), want)
    gotm = torch.max(v(x), dim=dim)
    wantm = torch.max(x, dim=dim)
    assert gotm.values.dim() == gotm.indices.dim() == 0
    assert gotm.values.cpu().item() == wantm.values.item()
    assert gotm.indices.cpu().item() == wantm.indices.item()


def test_the_value_is_the_one_at_the_index(backend):
    """max(dim=)'s two halves must agree, which ties are what breaks.

    A reduction that resolves a tie by taking the *later* element still reports
    a correct maximum -- it is the pairing that goes wrong. torch's answer is
    the earliest index and the value at it, and with a row holding both +0.0
    and -0.0 that difference is visible in the sign.
    """
    torch.manual_seed(3)
    for t in (torch.randn(5, 9), torch.randn(4, 17)):
        got = torch.max(v(t), dim=-1)
        idx = got.indices.cpu().unsqueeze(-1)
        gathered = t.gather(-1, idx).squeeze(-1)
        assert torch.equal(got.values.cpu(), gathered), (
            "max(dim=) returned a value that is not at the index it returned")
        assert torch.equal(got.indices.cpu(), torch.argmax(t, dim=-1))


@pytest.mark.parametrize("row,idx", [
    ([0.0, -0.0], 0),
    ([-0.0, 0.0], 0),
    ([-0.0, -0.0], 0),
    ([1.0, 1.0, 1.0, 1.0], 0),
    ([float("-inf"), float("-inf")], 0),
])
def test_ties_take_the_earliest_element(backend, row, idx):
    t = torch.tensor([row])
    got = torch.max(v(t), dim=-1)
    assert got.indices.cpu().item() == idx
    assert torch.argmax(v(t), dim=-1).cpu().item() == idx
    # and the value is that element's, sign of zero included
    assert got.values.cpu().item() == t[0][idx].item()
    assert torch.signbit(got.values.cpu()).item() == torch.signbit(t[0][idx]).item()


def test_nan_wins_and_its_index_is_the_first(backend):
    torch.manual_seed(5)
    for row in ([1.0, float("nan"), 3.0], [float("nan"), 1.0],
                [1.0, float("nan"), float("nan")]):
        t = torch.tensor([row])
        got = torch.max(v(t), dim=-1)
        assert torch.isnan(got.values.cpu()).all()
        assert torch.equal(got.indices.cpu(), torch.argmax(t, dim=-1))
        assert torch.equal(torch.argmax(v(t), dim=-1).cpu(),
                           torch.argmax(t, dim=-1))


def test_empty_inputs(backend):
    """Zero rows is an empty answer; a zero reduction dim is refused.

    torch draws the same line and refuses the second with a different message,
    because there is no index in an empty row at all.
    """
    backend.reset_stats()
    assert torch.argmax(torch.empty(0, 4, device="vortex"), -1).shape == (0,)
    assert torch.max(torch.empty(0, 4, device="vortex"), -1).indices.shape == (0,)
    assert backend.stats()["launches"] == 0, "an empty input launched a kernel"

    empty = torch.empty(0, 4, device="vortex")
    with assert_rejected("no elements", backend, may_allocate=True):
        torch.argmax(empty)          # a full reduction of nothing
    zero_dim = torch.empty(2, 0, device="vortex")
    with assert_rejected("empty dimension", backend, may_allocate=True):
        torch.argmax(zero_dim, -1)


def test_dtype_and_layout_are_refused(backend):
    double = torch.empty(2, 3, dtype=torch.float64, device="vortex")
    with assert_rejected("float32", backend):
        torch.argmax(double, 1)
    strided = v(torch.randn(4, 6).t())
    assert not strided.is_contiguous()
    with assert_rejected("contiguous", backend):
        torch.argmax(strided, 0)
    x2 = v(X2)
    with assert_rejected("out of range", backend):
        torch.argmax(x2, 2)


def test_steady_state_is_device_only(backend, stats):
    x = v(X3)
    torch.argmax(x, 1)              # warm up
    backend.reset_stats()
    for _ in range(3):
        torch.argmax(x, 1)
        torch.max(x, 1)
    st = stats()
    assert st["h2d_bytes"] == 0 and st["d2h_bytes"] == 0, (
        "argmax moved bytes to or from the host: %r" % st)
