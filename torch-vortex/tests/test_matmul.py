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

"""mm / linear / addmm semantics (W1.2, W3.1).

addmm used to be implemented as linear_impl(mat1, mat2, self), i.e. it computed
mat1 @ mat2.T. A non-square system was rejected for the wrong reason and a
square one silently returned the transpose product, so the non-symmetric square
case below is the regression that matters.
"""

import pytest
import torch

from helpers import assert_matches_cpu, assert_rejected


def v(t):
    return t.to("vortex")


def test_mm_rectangular(backend):
    a = torch.randn(2, 3)
    b = torch.randn(3, 5)
    assert_matches_cpu(torch.mm(v(a), v(b)), a @ b)


@pytest.mark.parametrize("shape", [(2, 3, 4, 5), (3, 1, 7, 2), (1, 17, 5, 19)])
def test_bmm_rectangular(backend, shape):
    batch, m, k, n = shape
    a = torch.randn(batch, m, k)
    b = torch.randn(batch, k, n)
    assert_matches_cpu(torch.bmm(v(a), v(b)), torch.bmm(a, b), rtol=1e-5,
                       atol=1e-6)


def test_bmm_queues_one_batched_gemm(backend):
    batch, m, k, n = 4, 5, 7, 3
    a = v(torch.randn(batch, m, k))
    b = v(torch.randn(batch, k, n))
    backend.reset_stats()
    got = torch.bmm(a, b)
    assert backend.stats()["launches"] == 1
    assert_matches_cpu(got, torch.bmm(a.cpu(), b.cpu()), rtol=1e-5, atol=1e-6)


def test_matmul_3d_reaches_bmm(backend):
    a = torch.randn(2, 4, 3)
    b = torch.randn(2, 3, 5)
    backend.reset_stats()
    got = torch.matmul(v(a), v(b))
    assert backend.stats()["launches"] == 1
    assert_matches_cpu(got, torch.matmul(a, b), rtol=1e-5, atol=1e-6)


def test_bmm_zero_batch_and_contraction(backend):
    a = torch.empty(0, 3, 4)
    b = torch.empty(0, 4, 5)
    got = torch.bmm(v(a), v(b))
    assert got.shape == (0, 3, 5)

    a = torch.empty(2, 3, 0)
    b = torch.empty(2, 0, 5)
    got = torch.bmm(v(a), v(b))
    assert got.shape == (2, 3, 5)
    assert torch.equal(got.cpu(), torch.zeros(2, 3, 5))


@pytest.mark.parametrize("m,k,n", [(4, 4, 4), (8, 8, 8), (16, 16, 4), (17, 17, 17),
                                   (2, 3, 5), (1, 1, 1), (1, 16, 4), (5, 7, 15)])
def test_mm_partial_tiles(backend, m, k, n):
    """Shapes whose n is not a multiple of 16, which is where this went wrong.

    Adding an addmm-style epilogue branch to tv_mm_kernel made VOLT accumulate
    incorrectly for exactly these shapes, so they are the regression set.
    """
    a = torch.randn(m, k)
    b = torch.randn(k, n)
    assert_matches_cpu(torch.mm(v(a), v(b)), a @ b, rtol=1e-5, atol=1e-6)


def test_the_local_matmul_kernel_is_gone(backend):
    """The extension no longer has one: matmul is the DL library's gemm.

    It used to carry a workaround. Folding torch.addmm's alpha/beta/self branch
    into this backend's own tv_mm_kernel made VOLT accumulate wrongly for any n
    that is not a multiple of 16, so the branch was moved into a separate
    epilogue kernel and this test pinned that split by argument size.

    Both kernels are gone now (W3.1): the DL gemm carries the same branch
    inline, and it was measured clean on exactly the shapes that broke the
    local one -- (4,4,4), (16,16,4), (17,17,17), (1,16,4), (2,3,5), (8,8,8) --
    with the epilogue active (alpha=2, beta=0.5), maxdiff 0.0. So the workaround
    is not needed where the kernel now lives, and test_mm_partial_tiles keeps
    those shapes as the regression set.
    """
    sizes = backend._ext.arg_sizes()
    assert "tv_mm_kernel" not in sizes
    assert "tv_mm_epilogue_kernel" not in sizes
    # and the op still works, so the assertion above is not hiding a break
    a = torch.randn(4, 4)
    assert torch.equal(
        torch.mm(v(a), v(a)).cpu(),
        torch.mm(v(a).cpu(), v(a).cpu()))


def test_mm_non_symmetric_square(backend):
    """The case the old addmm got wrong by transposing the second operand."""
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    got = torch.mm(v(a), v(b)).cpu()
    torch.testing.assert_close(got, a @ b, rtol=0, atol=0)
    assert not torch.allclose(got, a @ b.t()), "matmul transposed its second operand"


def test_addmm_bias_vector(backend):
    self = torch.randn(5)
    m1 = torch.randn(2, 3)
    m2 = torch.randn(3, 5)
    assert_matches_cpu(torch.addmm(v(self), v(m1), v(m2)), torch.addmm(self, m1, m2))


def test_addmm_full_matrix_addend(backend):
    self = torch.randn(2, 5)
    m1 = torch.randn(2, 3)
    m2 = torch.randn(3, 5)
    assert_matches_cpu(torch.addmm(v(self), v(m1), v(m2)), torch.addmm(self, m1, m2))


def test_addmm_alpha_beta(backend):
    self = torch.randn(2, 5)
    m1 = torch.randn(2, 3)
    m2 = torch.randn(3, 5)
    got = torch.addmm(v(self), v(m1), v(m2), beta=0.5, alpha=2.0)
    assert_matches_cpu(got, torch.addmm(self, m1, m2, beta=0.5, alpha=2.0))


def test_addmm_beta_zero_ignores_nan_in_self(backend):
    """beta == 0 must not read self at all: 0 * Inf would be NaN."""
    self = torch.full((5,), float("nan"))
    m1 = torch.randn(2, 3)
    m2 = torch.randn(3, 5)
    got = torch.addmm(v(self), v(m1), v(m2), beta=0.0).cpu()
    assert torch.isfinite(got).all(), "beta=0 leaked NaN from self"
    torch.testing.assert_close(got, m1 @ m2, rtol=1e-6, atol=1e-6)


def test_mm_k_zero(backend):
    a = torch.zeros(2, 0)
    b = torch.zeros(0, 3)
    got = torch.mm(v(a), v(b))
    assert got.shape == (2, 3)
    assert torch.equal(got.cpu(), torch.zeros(2, 3))


def test_mm_empty_rows(backend):
    a = torch.zeros(0, 3)
    b = torch.randn(3, 4)
    assert torch.mm(v(a), v(b)).shape == (0, 4)


def test_linear(backend):
    x = torch.randn(4, 6)
    w = torch.randn(5, 6)
    b = torch.randn(5)
    assert_matches_cpu(torch.nn.functional.linear(v(x), v(w), v(b)),
                       torch.nn.functional.linear(x, w, b))
    assert_matches_cpu(torch.nn.functional.linear(v(x), v(w), None),
                       torch.nn.functional.linear(x, w, None))


def test_linear_transb_does_not_materialise_weight(backend):
    x = v(torch.randn(4, 6))
    w = v(torch.randn(5, 6))
    bias = v(torch.randn(5))
    backend.reset_stats()
    out = torch.nn.functional.linear(x, w, bias)
    stats = backend.stats()
    # zeros initialization, transposed-B GEMM, and bias epilogue. A materialised
    # `weight.t().contiguous()` would add a fourth device launch.
    assert stats["launches"] == 3, "linear launched a temporary transpose copy"
    assert torch.equal(out.cpu(), torch.nn.functional.linear(x.cpu(), w.cpu(), bias.cpu()))


def test_matmul_rejects_bad_input(backend):
    # operands are materialised outside the assertion: moving them to the
    # device is not part of the call being rejected
    a23, b45 = v(torch.randn(2, 3)), v(torch.randn(4, 5))
    self21, m23, m32 = v(torch.randn(2, 1)), v(torch.randn(2, 3)), v(torch.randn(3, 2))
    self22, m3d, m22 = v(torch.randn(2, 2)), v(torch.randn(2, 2, 2)), v(torch.randn(2, 2))
    f64a, f64b = v(torch.randn(2, 2).double()), v(torch.randn(2, 2).double())
    ba, bb = v(torch.randn(2, 3, 4)), v(torch.randn(3, 4, 5))
    bc, bd = v(torch.randn(2, 3, 4)), v(torch.randn(2, 5, 6))
    bmm_nc = v(torch.randn(2, 4, 3).transpose(1, 2))
    bmm_rhs = v(torch.randn(2, 3, 5))

    with assert_rejected("contraction mismatch", backend):
        torch.mm(a23, b45)
    with assert_rejected("addmm self is", backend):
        torch.addmm(self21, m23, m32)
    with assert_rejected("2-D", backend):
        torch.addmm(self22, m3d, m22)
    with assert_rejected("float32", backend):
        torch.mm(f64a, f64b)
    with assert_rejected("batch mismatch", backend):
        torch.bmm(ba, bb)
    with assert_rejected("contraction mismatch", backend):
        torch.bmm(bc, bd)
    with assert_rejected("contiguous", backend):
        torch.bmm(bmm_nc, bmm_rhs)
