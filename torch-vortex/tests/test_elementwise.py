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

"""Elementwise operators (W3.2).

Arithmetic, scalar operands, in-place forms and the unary maths. Every case
compares against the CPU run of the same expression, so the reference is
PyTorch's own semantics rather than a restatement of what the kernel does.
"""

import pytest
import torch

from helpers import assert_matches_cpu, assert_rejected

X = torch.tensor([0.5, 1.0, 2.0, 4.0, 0.25, 3.0, 8.0, 1.5])


def v(t):
    return t.to("vortex")


@pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
def test_tensor_tensor(backend, op):
    assert_matches_cpu(getattr(torch, op)(v(X), v(X)), getattr(torch, op)(X, X))


@pytest.mark.parametrize("expr", [
    "x + 1.0", "x - 0.5", "x * 3.0", "x / 2.0",
    "1.0 + x", "2.0 - x", "3.0 * x", "4.0 / x",
])
def test_scalar_operands(backend, expr):
    """`x + 1.0` arrives as add.Tensor with a 0-dim CPU tensor."""
    x = X
    assert_matches_cpu(eval(expr.replace("x", "v(X)")), eval(expr.replace("x", "X")))


@pytest.mark.parametrize("alpha", [0.5, 2.0])
def test_add_and_sub_alpha(backend, alpha):
    """sub is not add: the alpha scaling has to compose with the operation."""
    assert_matches_cpu(torch.add(v(X), v(X), alpha=alpha),
                       torch.add(X, X, alpha=alpha))
    assert_matches_cpu(torch.sub(v(X), v(X), alpha=alpha),
                       torch.sub(X, X, alpha=alpha))
    assert_matches_cpu(torch.sub(2.0, v(X), alpha=alpha),
                       torch.sub(2.0, X, alpha=alpha))
    assert_matches_cpu(torch.add(v(X), 1.0, alpha=alpha),
                       torch.add(X, 1.0, alpha=alpha))


def test_alpha_zero(backend):
    assert_matches_cpu(torch.add(v(X), v(X), alpha=0.0),
                       torch.add(X, X, alpha=0.0))


@pytest.mark.parametrize("name", ["neg", "abs", "exp", "log", "sqrt", "rsqrt",
                                  "sigmoid", "tanh", "reciprocal"])
def test_unary_math(backend, name):
    fn = getattr(torch, name)
    assert_matches_cpu(fn(v(X)), fn(X), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", ["maximum", "minimum"])
def test_minimum_maximum(backend, name):
    fn = getattr(torch, name)
    assert_matches_cpu(fn(v(X), v(X) * 0.5), fn(X, X * 0.5))


def test_minimum_maximum_propagate_nan(backend):
    """fmaxf/fminf ignore NaN; torch.maximum/minimum do not."""
    a = torch.tensor([1.0, float("nan"), 3.0])
    b = torch.tensor([2.0, 2.0, 2.0])
    assert_matches_cpu(torch.maximum(v(a), v(b)), torch.maximum(a, b))
    assert_matches_cpu(torch.minimum(v(a), v(b)), torch.minimum(a, b))


@pytest.mark.parametrize("stmt", [
    "y.add_(v(X))", "y.sub_(v(X))", "y.mul_(v(X))", "y.div_(v(X))",
    "y.add_(1.0)", "y.mul_(3.0)", "y.sub_(0.5)", "y.div_(2.0)",
    "y.add_(v(X), alpha=2.0)", "y.sub_(v(X), alpha=2.0)",
    "y.neg_()", "y.exp_()", "y.abs_()", "y.relu_()",
])
def test_inplace(backend, stmt):
    want = X.clone()
    eval(stmt.replace("y", "want").replace("v(X)", "X"))
    y = X.clone().to("vortex")
    eval(stmt)
    assert_matches_cpu(y, want, rtol=1e-5, atol=1e-6)


def test_inplace_bumps_the_version_counter(backend):
    y = X.clone().to("vortex")
    before = y._version
    y.add_(v(X))
    assert y._version > before


def test_division_by_zero_matches_cpu(backend):
    """IEEE semantics, not a guard: 1/0 is inf and 0/0 is NaN."""
    a = torch.tensor([1.0, 0.0, -1.0])
    b = torch.tensor([0.0, 0.0, 0.0])
    assert_matches_cpu(v(a) / v(b), a / b)


@pytest.mark.parametrize("shapes", [
    ((4, 6), (6,)),          # bias vector
    ((4, 6), (1, 6)),        # row
    ((4, 6), (4, 1)),        # column
    ((4, 6), (1, 1)),        # scalar-shaped tensor
    ((2, 3, 4), (4,)),       # trailing only
    ((2, 3, 4), (3, 1)),     # middle
    ((2, 3, 4), (2, 1, 1)),
])
def test_broadcasting(backend, shapes):
    """`x + bias` is the shape models actually write."""
    torch.manual_seed(7)
    a = torch.randn(*shapes[0])
    b = torch.randn(*shapes[1])
    for op in ("add", "sub", "mul", "div"):
        assert_matches_cpu(getattr(torch, op)(v(a), v(b)),
                           getattr(torch, op)(a, b), rtol=1e-5, atol=1e-6)


def test_broadcast_matches_expanded_reference(backend):
    """Checked against an explicit expand, not just against CPU torch."""
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    b = torch.tensor([1.0, 2.0, 3.0])
    want = a + b.unsqueeze(0).expand(2, 3)
    assert_matches_cpu(v(a) + v(b), want, rtol=0, atol=0)


def test_inplace_broadcast_cannot_grow(backend):
    """In place, a broadcast that would grow self is refused."""
    col = v(torch.randn(4, 1))
    wide = v(torch.randn(4, 6))
    with assert_rejected("which in place cannot do", backend):
        col.add_(wide)


@pytest.mark.parametrize("name,kwargs", [
    ("silu", {}), ("gelu", {}), ("gelu", {"approximate": "tanh"}),
])
def test_activations(backend, name, kwargs):
    fn = getattr(torch.nn.functional, name)
    assert_matches_cpu(fn(v(X), **kwargs), fn(X, **kwargs), rtol=1e-5, atol=1e-6)


def test_gelu_forms_differ(backend):
    """The default is erf-based; approximate='tanh' is a different function."""
    a = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    va = v(a)   # hoisted: the device transfer is not part of what is rejected
    erf_form = torch.nn.functional.gelu(va).cpu()
    tanh_form = torch.nn.functional.gelu(va, approximate="tanh").cpu()
    assert not torch.allclose(erf_form, tanh_form, rtol=1e-6, atol=1e-6)
    assert_matches_cpu(torch.nn.functional.gelu(va),
                       torch.nn.functional.gelu(a), rtol=1e-6, atol=1e-7)
    with assert_rejected("unsupported", backend):
        torch.nn.functional.gelu(va, approximate="nonsense")


def test_shape_mismatch_is_refused(backend):
    # operands are prepared outside the assertion: moving them to the device
    # is not part of the call being rejected
    four, five = v(torch.ones(4)), v(torch.ones(5))
    with assert_rejected("must match the size", backend):
        four + five


def test_non_f32_is_refused(backend):
    a = v(torch.ones(4, dtype=torch.float64))
    b = v(torch.ones(4, dtype=torch.float64))
    with assert_rejected("float32", backend):
        a + b
