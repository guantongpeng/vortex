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

"""view / reshape (W1.5).

The previous view rebuilt a TensorImpl straight from the requested sizes: no
-1 resolution, no element-count check, no storage offset. view(-1) produced a
tensor with a negative size and view(1000) on a 4-element tensor was accepted.
view now delegates to ATen, so the checks are ATen's own and the messages are
too. Where behaviour is not obvious the test asserts parity with CPU torch on
the same input rather than guessing.
"""

import pytest
import torch

from helpers import assert_rejected


def test_view_basic(backend):
    x = torch.arange(8, dtype=torch.float32)
    vx = x.to("vortex")
    assert vx.view(2, 4).shape == (2, 4)
    assert vx.view(2, 4).cpu().tolist() == x.view(2, 4).tolist()


def test_view_minus_one(backend):
    x = torch.arange(8, dtype=torch.float32)
    assert x.to("vortex").view(-1).shape == (8,)
    assert x.to("vortex").view(2, -1).shape == (2, 4)
    assert x.to("vortex").view(-1, 4).shape == (2, 4)


def test_view_shares_storage_and_version(backend):
    x = torch.randn(8)
    vx = x.to("vortex")
    flat = vx.view(-1)
    assert flat.data_ptr() == vx.data_ptr(), "view did not share storage"

    # mutate through the view; the base must see the version bump. copy_ is the
    # v1 in-place op that is registered (add_/mul_ are W3.2).
    before = vx._version
    flat.copy_(torch.ones(8, device="vortex"))
    assert vx._version > before, "the view did not share the version counter"

    flat2 = vx.view(2, 4)
    assert flat2._version == vx._version


def test_view_zero_element(backend):
    assert torch.empty(0).to("vortex").view(0).shape == (0,)
    assert torch.empty(0).to("vortex").view(0, 0).shape == (0, 0)


def test_view_rejects_wrong_numel(backend):
    vx = torch.arange(8, dtype=torch.float32).to("vortex")
    with assert_rejected("invalid for input of size", backend):
        vx.view(2, 3)
    with assert_rejected("invalid for input of size", backend):
        vx.view(1000)


def test_view_rejects_two_minus_ones(backend):
    vx = torch.arange(6, dtype=torch.float32).to("vortex")
    with assert_rejected("only one dimension", backend):
        vx.view(-1, -1)


def test_reshape_parity_with_cpu(backend):
    """reshape may copy; assert the same input gives the same answer as CPU."""
    x = torch.randn(4, 6)
    non_contig = x.t()
    # `.to("vortex")` on a transposed tensor asks for a strided destination,
    # which v1 refuses; the copy+reshape path is exercised from a contiguous
    # device tensor instead, and the refusal is pinned here.
    got = non_contig.contiguous().to("vortex").reshape(24)
    assert got.shape == (24,)
    assert torch.equal(got.cpu(), non_contig.reshape(24))

    # Refused, not silently copied: _to_copy asks for a strided destination.
    with pytest.raises(RuntimeError, match="non-contiguous vortex tensor"):
        non_contig.to("vortex")
