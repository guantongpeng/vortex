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

"""Elementwise ops (W1.1): relu is not inplace, and its special values match.

relu used to run the in-place kernel on its input and return that same tensor,
which mutates a caller's data under a non-mutating schema. These tests pin the
non-mutating contract, the shared-input case that exposed it, and the two
special values a naive `x > 0 ? x : 0` gets wrong: NaN and -0.0.
"""

import pytest
import torch

from helpers import assert_matches_cpu, assert_rejected, signbit_of

SPECIAL = [float("-inf"), -1.0, -0.0, 0.0, 1.0, float("inf"), float("nan")]


def test_relu_does_not_touch_its_input(backend):
    x = torch.randn(65, dtype=torch.float32)
    x0 = x.clone()
    vx = x.to("vortex")
    out = torch.relu(vx)

    assert out.device.type == "vortex"
    # the input is untouched, and the result is a different allocation
    assert torch.equal(vx.cpu(), x0)
    assert out.data_ptr() != vx.data_ptr()
    assert vx._version == 0


def test_relu_two_consumers_of_one_input(backend):
    """The shape of the original bug: a residual branch reusing its input."""
    x = torch.randn(48, dtype=torch.float32)
    vx = x.to("vortex")
    ones = torch.ones_like(x).to("vortex")
    a = torch.relu(vx)
    b = torch.relu(vx) + ones
    assert_matches_cpu(a, torch.relu(x))
    assert_matches_cpu(b, torch.relu(x) + 1.0)
    assert torch.equal(vx.cpu(), x), "relu modified the shared input"


def test_relu_inplace_is_the_mutating_one(backend):
    x = torch.randn(33, dtype=torch.float32)
    vx = x.to("vortex")
    before = vx._version
    vx.relu_()
    assert vx._version > before, "relu_ did not bump the version counter"
    assert_matches_cpu(vx, torch.relu(x))


def test_relu_special_values(backend):
    x = torch.tensor(SPECIAL, dtype=torch.float32)
    got = torch.relu(x.to("vortex")).cpu()
    want = torch.relu(x)
    torch.testing.assert_close(got, want, rtol=0, atol=0, equal_nan=True)

    # assert_close cannot see the sign of zero, so check it explicitly:
    # relu(-0.0) is -0.0, not +0.0.
    assert signbit_of(got)[2] is True, "relu(-0.0) lost its sign"
    assert signbit_of(got)[3] is False


def test_relu_rejects_bad_input(backend):
    f64 = torch.ones(4, dtype=torch.float64).to("vortex")
    with assert_rejected("float32", backend):
        torch.relu(f64)


def test_add_and_mul(backend):
    a = torch.randn(77, dtype=torch.float32)
    b = torch.randn(77, dtype=torch.float32)
    assert_matches_cpu(a.to("vortex") + b.to("vortex"), a + b)
    assert_matches_cpu(a.to("vortex") * b.to("vortex"), a * b)


def test_scalar_operand_is_refused_by_name(backend):
    """`x + 1.0` reaches add.Tensor with a 0-dim CPU tensor.

    Supported in W3.2, not here, but the message has to name the real boundary
    rather than claim the operand should have been a vortex tensor.
    """
    x = torch.ones(4, device="vortex")
    with assert_rejected("scalar operands", backend):
        x + 1.0


def test_empty_tensor_launches_nothing(backend):
    backend.reset_stats()
    x = torch.empty(0, device="vortex")
    out = torch.relu(x)
    assert out.numel() == 0
    assert backend.stats()["launches"] == 0
