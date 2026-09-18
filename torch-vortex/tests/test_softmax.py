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

"""softmax, log_softmax and logsumexp (W3.2).

All three are one kernel in sw/dl: the same row max and the same shifted
exponentials, differing only in what is written out. softmax and log_softmax
keep every dimension, so a dim that is not the last one needs the tensor moved
and the answer moved back -- that round trip is what these tests exist to
exercise, since the trailing case would pass without it.
"""

import pytest
import torch
import torch.nn.functional as F

from helpers import assert_matches_cpu, assert_rejected


def v(t):
    return t.to("vortex")


X3 = torch.randn(2, 3, 4)
X4 = torch.randn(2, 3, 4, 5)


@pytest.mark.parametrize("fn", [F.softmax, F.log_softmax], ids=["softmax", "log_softmax"])
@pytest.mark.parametrize("shape", [(2, 3, 4), (2, 3, 4, 5), (7,), (1, 1)], ids=str)
def test_every_dim_matches_cpu(backend, fn, shape):
    """Every dim, not just the trailing one: they are different code paths."""
    torch.manual_seed(sum(shape))
    x = torch.randn(*shape)
    for dim in range(-x.dim(), x.dim()):
        assert_matches_cpu(fn(v(x), dim), fn(x, dim), rtol=1e-5, atol=1e-6)


def test_the_modules_reach_the_same_kernel(backend):
    """nn.Softmax and nn.LogSoftmax dispatch to these schemas."""
    torch.manual_seed(21)
    assert_matches_cpu(torch.nn.Softmax(dim=1)(v(X3)),
                       torch.nn.Softmax(dim=1)(X3), rtol=1e-5, atol=1e-6)
    assert_matches_cpu(torch.nn.LogSoftmax(dim=-1)(v(X3)),
                       torch.nn.LogSoftmax(dim=-1)(X3), rtol=1e-5, atol=1e-6)


def test_logsumexp_matches_cpu(backend):
    torch.manual_seed(22)
    assert_matches_cpu(torch.logsumexp(v(X3), 1), torch.logsumexp(X3, 1),
                       rtol=1e-5, atol=1e-6)
    assert_matches_cpu(torch.logsumexp(v(X3), -1, keepdim=True),
                       torch.logsumexp(X3, -1, keepdim=True), rtol=1e-5, atol=1e-6)
    assert_matches_cpu(torch.logsumexp(v(X3), [0]), torch.logsumexp(X3, [0]),
                       rtol=1e-5, atol=1e-6)


def test_log_softmax_is_not_log_of_softmax(backend):
    """The stable form, checked where the unstable one loses everything.

    log(softmax(x)) underflows to -inf as soon as an element is more than ~88
    below the row max; the kernel computes (x - max) - log(sum) instead. A
    test at ordinary magnitudes cannot tell the two apart.
    """
    x = torch.tensor([[0.0, -100.0, -200.0]])
    got = F.log_softmax(v(x), -1)
    want = F.log_softmax(x, -1)
    assert torch.isfinite(got.cpu()).all(), (
        "log_softmax underflowed: %r" % got.cpu().tolist())
    assert_matches_cpu(got, want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("row,what", [
    ([[1.0, float("nan"), 3.0]], "a NaN"),
    ([[float("inf"), 0.0]], "a +inf"),
    ([[float("-inf"), float("-inf")]], "all -inf"),
    ([[1e38, 1e38]], "a value that overflows exp"),
    ([[0.0, -1000.0]], "a very negative value"),
])
def test_special_values_match_cpu(backend, row, what):
    """The max shift is what makes these work; without it exp overflows.

    A NaN row comes out NaN through the exponential and the sum, so the row
    max being NaN-blind does not matter -- which is why it is left that way.
    """
    t = torch.tensor(row, dtype=torch.float32)
    for fn in (F.softmax, F.log_softmax):
        assert_matches_cpu(fn(v(t), -1), fn(t, -1), rtol=1e-6, atol=1e-7,
                           equal_nan=True)
    assert_matches_cpu(torch.logsumexp(v(t), -1), torch.logsumexp(t, -1),
                       rtol=1e-6, atol=1e-7, equal_nan=True)


def test_degenerate_shapes(backend):
    """A 0-d tensor stays 0-d, a 1-element row is 1, and an empty stays empty."""
    scalar = torch.tensor(3.0)
    got = F.softmax(v(scalar), 0)
    assert got.dim() == 0, "a 0-d input came back as shape %s" % (tuple(got.shape),)
    assert_matches_cpu(got, F.softmax(scalar, 0), rtol=0, atol=0)

    one = torch.tensor([[5.0]])
    assert_matches_cpu(F.softmax(v(one), 1), torch.tensor([[1.0]]), rtol=0, atol=0)

    backend.reset_stats()
    empty = torch.empty(0, 4, device="vortex")
    assert F.softmax(empty, 1).shape == (0, 4)
    assert backend.stats()["launches"] == 0, "an empty input launched a kernel"


def test_dim_out_of_range_is_refused(backend):
    x = v(X3)
    scalar = v(torch.tensor(3.0))
    with assert_rejected("out of range", backend):
        F.softmax(x, 3)
    with assert_rejected("out of range", backend):
        F.log_softmax(x, -4)
    # a 0-d tensor takes only 0 and -1, which is torch's rule too
    with assert_rejected("out of range", backend):
        torch._softmax(scalar, 1, False)


def test_several_dims_at_once_is_refused(backend):
    """logsumexp over two dims is refused rather than reduced one at a time.

    torch supports it. This is the same v1 boundary sum/mean/amax already
    draw, and it is named rather than silently serialised.
    """
    x = v(X3)
    with assert_rejected("over several dimensions", backend):
        torch.logsumexp(x, [0, 1])


def test_half_to_float_is_refused(backend):
    """The half-in/float-out form of the schema needs half support."""
    half = torch.empty(2, 3, dtype=torch.float16, device="vortex")
    with assert_rejected("half_to_float", backend):
        torch._softmax(half, 1, True)


def test_dtype_and_layout_are_refused(backend):
    double = torch.empty(2, 3, dtype=torch.float64, device="vortex")
    with assert_rejected("float32", backend):
        F.softmax(double, 1)
    strided = v(torch.randn(4, 6).t())
    assert not strided.is_contiguous()
    with assert_rejected("contiguous", backend):
        F.softmax(strided, 0)


def test_steady_state_is_device_only(backend, stats):
    """No host round-trip, and the moved path is copies on the device.

    The middle-dim path moves the tensor, normalises it and moves it back; all
    three steps are device work, which the transfer counters are what say.
    """
    x = v(X3)
    F.softmax(x, 1)                 # warm up
    backend.reset_stats()
    for _ in range(3):
        F.softmax(x, 1)
        F.log_softmax(x, 1)
        torch.logsumexp(x, 1)
    st = stats()
    assert st["h2d_bytes"] == 0 and st["d2h_bytes"] == 0, (
        "the softmax family moved bytes to or from the host: %r" % st)
