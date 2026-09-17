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

"""as_strided and strided copies (W3.2).

as_strided is the metadata operation behind transpose, permute, t and any
indexing, so without it none of those were reachable at all. The copy kernel is
what makes a strided tensor usable: hipMemcpy moves a linear range, and a
transposed view is not one.

Both directions matter and they are not symmetric. A strided *source* is a
read pattern; a strided *destination* is a write pattern, and it is what
`t.t().to("vortex")` asks for -- _to_copy preserves the source layout -- so
refusing it (as v1 did) made that expression fail.
"""

import pytest
import torch

from helpers import assert_matches_cpu


def v(t):
    return t.to("vortex")


X = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)


def test_transpose_is_a_view(backend):
    """as_strided shares storage; it must not copy."""
    vx = v(X)
    t = vx.transpose(1, 2)
    assert t.shape == (2, 4, 3)
    assert t.stride() == (12, 1, 4)
    assert t.data_ptr() == vx.data_ptr(), "transpose copied instead of viewing"
    assert not t.is_contiguous()


def test_transpose_round_trip_reads_back(backend):
    vx = v(X)
    assert torch.equal(vx.transpose(1, 2).cpu(), X.transpose(1, 2))
    # a stride pattern that is neither the original nor its transpose
    assert torch.equal(vx.permute(2, 0, 1).cpu(), X.permute(2, 0, 1))
    # t() is 2-D only, per PyTorch
    m = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    assert torch.equal(v(m).t().cpu(), m.t())


def test_contiguous_materialises_a_strided_view(backend):
    vx = v(X)
    c = vx.transpose(1, 2).contiguous()
    assert c.is_contiguous()
    assert torch.equal(c.cpu(), X.transpose(1, 2).contiguous())


def test_strided_destination(backend):
    """`t.t().to("vortex")`: _to_copy asks for a strided destination."""
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    got = v(x.t())
    assert not got.is_contiguous(), "the test is not exercising a strided destination"
    assert torch.equal(got.cpu(), x.t().contiguous())


def test_indexing(backend):
    vx = v(X)
    assert torch.equal(vx[:, 1:3].cpu(), X[:, 1:3])
    assert torch.equal(vx[1:].cpu(), X[1:])
    assert torch.equal(vx[..., 0].cpu(), X[..., 0])
    assert torch.equal(vx[0, 1, 2].cpu(), X[0, 1, 2])


def test_commands_on_strided_views(backend):
    vx = v(X)
    assert torch.equal(vx.transpose(1, 2).contiguous().cpu(),
                       X.transpose(1, 2).contiguous())
    assert torch.equal(vx.transpose(1, 2).flatten().cpu(),
                       X.transpose(1, 2).flatten())


def test_strided_copy_round_trip_is_exact(backend):
    """A strided copy has no arithmetic in it, so it must be bit-exact."""
    vx = v(X)
    back = vx.transpose(1, 2).cpu()
    assert torch.equal(back, X.transpose(1, 2))


def test_ops_still_require_contiguity(backend):
    """The elementwise kernels index linearly, so a strided input is refused.

    Supporting strided elementwise means stride-aware indexing in every
    kernel; until then the boundary has to stay loud rather than being
    papered over with an implicit copy.
    """
    vx = v(torch.randn(4, 6))
    with pytest.raises(RuntimeError):
        vx.t() + vx.t()
