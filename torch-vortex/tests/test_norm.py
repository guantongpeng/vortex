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

"""Layer norm and RMS norm (W3.2).

Both normalise over the trailing dimensions, which is the shape sw/dl's row
kernels take, so the ATen side validates the shape and the DL library computes.
Registering aten::native_layer_norm is enough for torch.nn.functional.layer_norm
and torch.nn.LayerNorm: ATen's aten::layer_norm is a composite that calls it,
which is measurable rather than assumed -- these tests call the public spelling.
"""

import pytest
import torch
import torch.nn.functional as F

from helpers import assert_matches_cpu, assert_rejected


def v(t):
    return t.to("vortex")


# ---- layer norm -----------------------------------------------------------

@pytest.mark.parametrize("shape,ns", [
    ((2, 3, 4), [4]),
    ((2, 3, 4, 5), [5]),
    ((2, 3, 4, 5), [4, 5]),
    ((2, 3, 4, 5), [3, 4, 5]),
    ((8,), [8]),
    ((1, 1, 1), [1]),
], ids=lambda a: str(a))
def test_layer_norm_matches_cpu(backend, shape, ns):
    torch.manual_seed(sum(shape) * 100 + len(ns))
    x = torch.randn(*shape)
    assert_matches_cpu(F.layer_norm(v(x), ns), F.layer_norm(x, ns),
                       rtol=1e-5, atol=1e-6)


def test_layer_norm_affine_and_no_affine(backend):
    """The absent affine is a mode, not an error.

    nn.LayerNorm(elementwise_affine=False) and F.layer_norm(x, shape) pass no
    weight or bias; torch computes gamma=1, beta=0. The DL kernel required both
    until now, so this path could not be called at all.
    """
    torch.manual_seed(7)
    x = torch.randn(2, 3, 4)
    w, b = torch.randn(4), torch.randn(4)
    assert_matches_cpu(F.layer_norm(v(x), [4]), F.layer_norm(x, [4]),
                       rtol=1e-5, atol=1e-6)
    assert_matches_cpu(F.layer_norm(v(x), [4], v(w), v(b)),
                       F.layer_norm(x, [4], w, b), rtol=1e-5, atol=1e-6)
    # torch's own two spellings agree, so ours must too
    torch.testing.assert_close(F.layer_norm(v(x), [4]).cpu(),
                               F.layer_norm(v(x), [4], v(torch.ones(4)),
                                            v(torch.zeros(4))).cpu(),
                               rtol=1e-6, atol=1e-6)


def test_native_layer_norm_returns_its_statistics(backend):
    """Out, mean and rstd all come back, and rstd is the reciprocal.

    rstd is 1/sqrt(var + eps) -- not var, and not 1/var -- and the variance is
    the biased estimator over the normalized dims.
    """
    torch.manual_seed(11)
    x = torch.randn(2, 3, 4)
    got = torch.native_layer_norm(v(x), [4], None, None, 1e-5)
    want = torch.native_layer_norm(x, [4], None, None, 1e-5)
    assert len(got) == 3
    assert got[1].shape == want[1].shape == (2, 3, 1)
    assert got[2].shape == want[2].shape == (2, 3, 1)
    for i, name in enumerate(("out", "mean", "rstd")):
        assert_matches_cpu(got[i], want[i], rtol=1e-5, atol=1e-6)
        assert got[i].device.type == "vortex", "%s escaped the device" % name


def test_layer_norm_variance_estimator(backend):
    """E[x^2] - mean^2 is not the variance in float32.

    For values around 1e6 both terms are ~1e12 and their difference is ~1, so
    the one-pass form cancels to nothing: it produced rstd = 1/sqrt(eps) =
    316.2 where the answer is 0.894. Checked against CPU, which uses Welford,
    so this fails whether the kernel regresses or merely returns to one pass.
    """
    x = torch.tensor([[1e6, 1e6 + 1.0, 1e6 + 2.0, 1e6 + 3.0]])
    got = F.layer_norm(v(x), [4]).cpu()
    want = F.layer_norm(x, [4])
    # this one is meaningful to float32: the inputs have no headroom, so the
    # deviation is 1 of 1e6 and everything after the seventh digit is gone
    assert (got - want).abs().max().item() < 1e-3, (
        "the normalised output for 1e6-scale input is wrong: %r vs %r"
        % (got.tolist(), want.tolist()))
    _, _, rstd = torch.native_layer_norm(v(x), [4], None, None, 1e-5)
    assert abs(rstd.cpu().item() - (1.0 / (1.25 + 1e-5) ** 0.5)) < 1e-4


# ---- rms norm -------------------------------------------------------------

@pytest.mark.parametrize("shape,ns", [((2, 3, 4), [4]), ((2, 4, 5), [4, 5]),
                                      ((8,), [8])], ids=lambda a: str(a))
def test_rms_norm_matches_cpu(backend, shape, ns):
    torch.manual_seed(sum(shape))
    x = torch.randn(*shape)
    assert_matches_cpu(torch.rms_norm(v(x), ns), torch.rms_norm(x, ns),
                       rtol=1e-5, atol=1e-6)


def test_rms_norm_weight_and_default_eps(backend):
    """eps=None is the dtype's epsilon, not zero.

    Measured on CPU: rms_norm with eps=None is bit-identical to
    eps=torch.finfo(torch.float32).eps (1.1920928955078125e-07) and different
    from eps=0. Substituting 0 would put the two a fraction apart, so the
    comparison is made at the default rather than at the tolerance edge.
    """
    torch.manual_seed(13)
    x = torch.randn(2, 3, 4)
    w = torch.randn(4)
    assert_matches_cpu(torch.rms_norm(v(x), [4], v(w)), torch.rms_norm(x, [4], w),
                       rtol=1e-5, atol=1e-6)
    assert_matches_cpu(torch.rms_norm(v(x), [4]), torch.rms_norm(x, [4]),
                       rtol=1e-5, atol=1e-6)
    # "the default is the machine epsilon" is only worth saying if the default
    # is distinguishable from something else -- and the comparison has to be
    # device-against-device to be bitwise. Comparing the default against CPU
    # let an eps=None -> 0 substitution through: the difference is 4.8e-07
    # there, well inside any tolerance worth using.
    eps0 = float(torch.finfo(torch.float32).eps)
    # The copy down is exact for float32, so this stays bitwise.
    default = torch.rms_norm(v(x), [4]).cpu()
    assert torch.equal(default, torch.rms_norm(v(x), [4], None, eps=eps0).cpu()), (
        "eps=None is not the machine epsilon")
    assert not torch.equal(default, torch.rms_norm(v(x), [4], None, eps=0.0).cpu()), (
        "eps=None behaved like eps=0")
    assert_matches_cpu(torch.rms_norm(v(x), [4], None, eps=0.0),
                       torch.rms_norm(x, [4], None, eps=0.0), rtol=1e-5, atol=1e-6)


# ---- refusals and boundaries ----------------------------------------------

def test_normalized_shape_must_be_the_trailing_dims(backend):
    """Refused by name, before anything is allocated.

    Every operand is built first: assert_rejected counts the bytes and launches
    a rejected call produces, and a `.to("vortex")` inside the block would be
    the call under test's own setup being counted against it.
    """
    x = v(torch.randn(2, 3, 4))
    flat = v(torch.randn(3))
    with assert_rejected("trailing", backend):
        F.layer_norm(x, [2, 3])
    with assert_rejected("trailing", backend):
        torch.rms_norm(x, [3, 4, 5])
    with assert_rejected("trailing", backend):
        torch.rms_norm(flat, [4])


def test_either_half_of_the_affine_alone(backend):
    """torch takes weight without bias, and bias without weight.

    The absent half means its identity -- gamma 1 or beta 0 -- rather than an
    error, so refusing either alone would be a refusal where torch computes.
    """
    torch.manual_seed(17)
    x = torch.randn(2, 3, 4)
    w, b = torch.randn(4), torch.randn(4)
    assert_matches_cpu(F.layer_norm(v(x), [4], v(w)), F.layer_norm(x, [4], w),
                       rtol=1e-5, atol=1e-6)
    assert_matches_cpu(F.layer_norm(v(x), [4], None, v(b)),
                       F.layer_norm(x, [4], None, b), rtol=1e-5, atol=1e-6)
    # and each alone is the same call with the other half's identity
    torch.testing.assert_close(
        F.layer_norm(v(x), [4], v(w)).cpu(),
        F.layer_norm(v(x), [4], v(w), v(torch.zeros(4))).cpu(),
        rtol=1e-6, atol=1e-7)
    assert_matches_cpu(torch.native_layer_norm(v(x), [4], v(w), None, 1e-5)[0],
                       torch.native_layer_norm(x, [4], w, None, 1e-5)[0],
                       rtol=1e-5, atol=1e-6)


def test_normalized_shape_must_not_be_empty(backend):
    """torch refuses an empty normalized_shape; accepting it computes nonsense.

    Every element becomes its own row, so the output is identically zero and
    the statistics are the input -- a result rather than the mistake.
    """
    x = v(torch.randn(2, 3))
    with assert_rejected("at least one dimension", backend):
        torch.native_layer_norm(x, [], None, None, 1e-5)
    with assert_rejected("at least one dimension", backend):
        torch.rms_norm(x, [], None)


def test_affine_shape_is_checked_not_just_its_size(backend):
    """A (2,2) weight against normalized_shape [4] has 4 elements and is wrong.

    torch names both shapes; counting elements accepts it and multiplies the
    flattened storage instead.
    """
    x = v(torch.randn(2, 3, 4))
    bad = v(torch.randn(2, 2))
    with assert_rejected("normalized shape", backend):
        F.layer_norm(x, [4], bad, bad)
    with assert_rejected("normalized shape", backend):
        torch.rms_norm(x, [4], bad)


def test_dtype_is_float32_only(backend):
    """Allocation is dtype-agnostic; compute is not.

    The tensors are allocated on the device rather than converted to it,
    because the conversion is refused for its own reason and would be the
    error under test.
    """
    half = torch.empty(2, 4, dtype=torch.float16, device="vortex")
    double = torch.empty(2, 4, dtype=torch.float64, device="vortex")
    with assert_rejected("float32", backend):
        F.layer_norm(half, [4])
    with assert_rejected("float32", backend):
        torch.rms_norm(double, [4])


def test_strided_input_is_refused(backend):
    """A transposed input is refused rather than silently copied.

    ATen's aten::layer_norm composite passes the strided tensor straight
    through to native_layer_norm, so this is the same boundary the reductions
    draw. The transfer that builds it happens outside the block: a rejected
    call must not be charged for its own setup.
    """
    strided = v(torch.randn(4, 6).t())
    assert not strided.is_contiguous()
    with assert_rejected("contiguous", backend):
        F.layer_norm(strided, [4])


def test_zero_batch_and_zero_normalized_dim(backend):
    """Two different zero shapes, and torch answers each differently."""
    # (0, 4) with shape [4]: no rows at all, so every output is empty
    got = torch.native_layer_norm(torch.empty(0, 4, device="vortex"), [4],
                                  None, None, 1e-5)
    want = torch.native_layer_norm(torch.empty(0, 4), [4], None, None, 1e-5)
    for g, w in zip(got, want):
        assert g.shape == w.shape
        assert g.numel() == 0
    assert F.layer_norm(torch.empty(0, 4, device="vortex"), [4]).shape == (0, 4)

    # (2, 0) with shape [0]: nothing to normalise, but the statistics are not
    # empty -- torch gives mean 0 and rstd NaN, which is a value to match and
    # not an empty to skip
    got = torch.native_layer_norm(torch.empty(2, 0, device="vortex"), [0],
                                  None, None, 1e-5)
    want = torch.native_layer_norm(torch.empty(2, 0), [0], None, None, 1e-5)
    assert got[0].shape == (2, 0)
    assert got[1].shape == want[1].shape == (2, 1)
    # torch.equal is false for NaN, and rstd here is NaN by construction
    torch.testing.assert_close(got[1].cpu(), want[1].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(got[2].cpu(), want[2].cpu(), rtol=0, atol=0,
                               equal_nan=True)
    assert torch.isnan(got[2].cpu()).all()
    assert (got[1].cpu() == 0.0).all()
    assert torch.rms_norm(torch.empty(2, 0, device="vortex"), [0]).shape == (2, 0)


def test_steady_state_is_device_only(backend, stats):
    """A steady-state forward moves no bytes either way.

    Computing the statistics on the host would have been one way to dodge the
    variance problem, and it is what batch norm used to do; this is what says
    it was not done here.
    """
    x = v(torch.randn(4, 8, 16))
    F.layer_norm(x, [16])           # warm up before the counters are read
    torch.rms_norm(x, [16])
    backend.reset_stats()
    for _ in range(3):
        F.layer_norm(x, [16])
        torch.rms_norm(x, [16])
    st = stats()
    assert st["h2d_bytes"] == 0 and st["d2h_bytes"] == 0, (
        "the norms moved bytes to or from the host: %r" % st)
    assert st["launches"] >= 6, (
        "six norm calls produced %d launches" % st["launches"])
