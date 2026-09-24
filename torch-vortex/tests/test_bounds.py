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

"""Shape, range and empty-tensor boundaries (W1.8).

Conv/pool output sizes used to be computed in uint32_t, so an input smaller
than the kernel wrapped around into a huge output size and grid instead of
erroring. Max pool seeded its accumulator with the finite -3.4e38f, which
silently dropped -inf and turned NaN into that sentinel.
"""

import pytest
import torch

from helpers import assert_matches_cpu, assert_rejected


def test_pool_max_special_values_match_cpu(backend):
    """max pool over -inf and NaN windows.

    The accumulator used to be seeded with the finite -3.4e38f, so an all -inf
    window came back as that sentinel and a NaN window silently lost the NaN.
    """
    cases = [
        torch.full((1, 1, 2, 2), float("-inf")),          # all -inf window
        torch.tensor([[[[float("nan"), 0.0], [3.0, 4.0]]]]),
        torch.tensor([[[[-float("inf"), -1.0], [1.0, 2.0]]]]),
        torch.tensor([[[[-float("inf"), float("nan")], [1.0, 2.0]]]]),
    ]
    for x in cases:
        got = torch.nn.functional.max_pool2d(x.to("vortex"), 2).cpu()
        want = torch.nn.functional.max_pool2d(x, 2)
        torch.testing.assert_close(got, want, rtol=0, atol=0, equal_nan=True)

    assert (torch.nn.functional.max_pool2d(
        cases[0].to("vortex"), 2).cpu() == float("-inf")).all()
    assert torch.isnan(torch.nn.functional.max_pool2d(
        cases[1].to("vortex"), 2).cpu()).all()


def test_pool_rejects_zero_stride(backend):
    x = torch.randn(1, 2, 4, 4).to("vortex")
    with assert_rejected("stride must be > 0", backend):
        torch.nn.functional.max_pool2d(x, 2, stride=0)


def test_pool_rejects_kernel_larger_than_input(backend):
    x = torch.randn(1, 2, 2, 2).to("vortex")
    with assert_rejected("is smaller than kernel", backend):
        torch.nn.functional.max_pool2d(x, 3)


def test_conv_rejects_negative_padding(backend):
    x = torch.randn(1, 2, 4, 4).to("vortex")
    w = torch.randn(3, 2, 3, 3).to("vortex")
    with assert_rejected("padding must be >= 0", backend):
        torch.nn.functional.conv2d(x, w, padding=-1)


def test_conv_rejects_kernel_larger_than_input(backend):
    x = torch.randn(1, 2, 2, 2).to("vortex")
    w = torch.randn(3, 2, 4, 4).to("vortex")
    with assert_rejected("is smaller than kernel", backend):
        torch.nn.functional.conv2d(x, w)


@pytest.mark.parametrize("ci,kernel,groups,width", [(1, 5, 1, 7), (3, 7, 1, 23),
                                                      (513, 3, 1, 3), (514, 3, 2, 3)])
def test_conv_weight_tiles(backend, ci, kernel, groups, width):
    torch.manual_seed(71)
    x = torch.randn(1, ci, kernel, max(kernel, width))
    w = torch.randn(2, ci // groups, kernel, kernel) / ci**0.5
    b = torch.randn(2)
    dx, dw, db = (t.to("vortex") for t in (x, w, b))
    before = backend.stats()
    got = torch.nn.functional.conv2d(dx, dw, db, groups=groups)
    after = backend.stats()
    assert after["launches"] - before["launches"] == 1
    assert after["d2h_bytes"] == before["d2h_bytes"]
    want = torch.nn.functional.conv2d(x, w, b, groups=groups)
    assert_matches_cpu(got, want, rtol=3e-5, atol=3e-5)


def test_empty_inputs_launch_nothing(backend):
    backend.reset_stats()
    w = torch.randn(4, 3, 3, 3).to("vortex")

    out = torch.nn.functional.conv2d(torch.empty(0, 3, 8, 8, device="vortex"), w,
                                     padding=1)
    assert out.shape == (0, 4, 8, 8)

    assert backend.stats()["launches"] == 0, "an empty input launched a kernel"


def test_empty_elementwise_and_reductions_launch_nothing(backend):
    """Zero elements is not a launch, on the DL path either.

    The elementwise and reduction kernels now live in sw/dl, whose entry points
    refuse n == 0 (and rows == 0) rather than accepting an empty job. The old
    local kernels never saw the case: the launch helper dropped a zero grid
    first. So the rule has to be stated on this side, and if it stops being
    stated the DL library answers with an error rather than an empty result.
    """
    backend.reset_stats()
    empty = torch.empty(0, 4, device="vortex")

    assert torch.relu(empty).shape == (0, 4)
    # zero rows: an empty result, so nothing to compute and nothing to write
    assert torch.amax(empty, dim=1).shape == (0,)

    st = backend.stats()
    assert st["launches"] == 0, "an empty tensor launched a kernel: %r" % st


def test_full_reduction_of_an_empty_tensor_has_no_max(backend):
    """amax of a tensor with no elements is undefined, and torch says so.

    sum and mean do have answers (0 and NaN) and this path writes them with
    one fill rather than a reduction; max has none, so inventing -inf -- which
    is what the old kernel's -INFINITY seed produced -- would be a value torch
    never returns. Refusing is the parity-preserving answer.
    """
    empty = torch.empty(0, device="vortex")
    # the output tensor is sized before the reduction is handed the layout, so
    # the refusal comes after an allocation -- the launch check still applies
    with assert_rejected("no elements", backend, may_allocate=True):
        torch.amax(empty)

    assert empty.sum().cpu().item() == 0.0
    assert torch.isnan(empty.mean().cpu())

    # Reducing a dimension that is itself empty is still refused, and that is
    # the pre-existing gap rather than something this path closed: the layout
    # normalisation divides by the row length, so it has no shape to build.
    # torch returns zeros(3) here. Pinned so the gap stays visible.
    with pytest.raises(RuntimeError):
        torch.empty(3, 0, device="vortex").sum(dim=1)


def test_degenerate_pool_window_matches_cpu(backend):
    """A 0-size spatial input is rejected by PyTorch too, so refusal is parity."""
    with pytest.raises(RuntimeError):
        torch.nn.functional.max_pool2d(torch.empty(1, 2, 0, 0), 2)
    with assert_rejected("smaller than kernel", backend):
        torch.nn.functional.max_pool2d(torch.empty(1, 2, 0, 0, device="vortex"), 2)


def test_conv_and_pool_parity(backend):
    x = torch.randn(2, 3, 9, 7)
    w = torch.randn(4, 3, 3, 3)
    b = torch.randn(4)
    assert_matches_cpu(
        torch.nn.functional.conv2d(x.to("vortex"), w.to("vortex"), b.to("vortex"),
                                   stride=2, padding=1),
        torch.nn.functional.conv2d(x, w, b, stride=2, padding=1),
        rtol=1e-5, atol=1e-5)
    assert_matches_cpu(torch.nn.functional.max_pool2d(x.to("vortex"), 3, 2, 1),
                       torch.nn.functional.max_pool2d(x, 3, 2, 1))


def test_grouped_and_depthwise_conv_matches_cpu(backend):
    x = torch.randn(2, 4, 7, 6)
    w = torch.randn(6, 2, 3, 3)
    b = torch.randn(6)
    got = torch.nn.functional.conv2d(x.to("vortex"), w.to("vortex"),
                                     b.to("vortex"), padding=1, groups=2)
    want = torch.nn.functional.conv2d(x, w, b, padding=1, groups=2)
    assert_matches_cpu(got, want, rtol=1e-5, atol=1e-5)

    xd = torch.randn(1, 4, 5, 5)
    wd = torch.randn(4, 1, 3, 3)
    assert_matches_cpu(
        torch.nn.functional.conv2d(xd.to("vortex"), wd.to("vortex"),
                                   padding=1, groups=4),
        torch.nn.functional.conv2d(xd, wd, padding=1, groups=4),
        rtol=1e-5, atol=1e-5)


def test_dilated_conv_matches_cpu(backend):
    x = torch.randn(1, 3, 9, 10)
    w = torch.randn(4, 3, 3, 2)
    b = torch.randn(4)
    got = torch.nn.functional.conv2d(x.to("vortex"), w.to("vortex"),
                                     b.to("vortex"), padding=(2, 1),
                                     dilation=(2, 1))
    want = torch.nn.functional.conv2d(x, w, b, padding=(2, 1),
                                     dilation=(2, 1))
    assert_matches_cpu(got, want, rtol=1e-5, atol=1e-5)


def test_dilated_pool_matches_cpu(backend):
    x = torch.randn(1, 2, 9, 10)
    got = torch.nn.functional.max_pool2d(
        x.to("vortex"), 3, stride=2, padding=1, dilation=(2, 1),
        ceil_mode=True).cpu()
    want = torch.nn.functional.max_pool2d(
        x, 3, stride=2, padding=1, dilation=(2, 1), ceil_mode=True)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def test_adaptive_avg_pool(backend):
    x = torch.randn(2, 3, 5, 7)
    assert_matches_cpu(torch.nn.functional.adaptive_avg_pool2d(x.to("vortex"), 1),
                       torch.nn.functional.adaptive_avg_pool2d(x, 1),
                       rtol=1e-5, atol=1e-6)


def test_avg_pool2d_matches_cpu(backend):
    x = torch.randn(2, 3, 5, 7)
    for count_include_pad in (True, False):
        got = torch.nn.functional.avg_pool2d(
            x.to("vortex"), 3, stride=2, padding=1,
            count_include_pad=count_include_pad).cpu()
        want = torch.nn.functional.avg_pool2d(
            x, 3, stride=2, padding=1,
            count_include_pad=count_include_pad)
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def test_pool_ceil_mode_matches_cpu(backend):
    x = torch.randn(1, 2, 5, 6)
    for fn in (torch.nn.functional.max_pool2d, torch.nn.functional.avg_pool2d):
        got = fn(x.to("vortex"), 3, stride=2, padding=1, ceil_mode=True).cpu()
        want = fn(x, 3, stride=2, padding=1, ceil_mode=True)
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def test_avg_pool2d_divisor_override(backend):
    x = torch.randn(1, 2, 4, 4).to("vortex")
    got = torch.nn.functional.avg_pool2d(x, 2, stride=2, divisor_override=3)
    want = torch.nn.functional.avg_pool2d(x.cpu(), 2, stride=2, divisor_override=3)
    assert_matches_cpu(got, want, rtol=1e-5, atol=1e-6)


def test_device_rng_is_refused_by_name(backend):
    """torch.randn(device='vortex') needs a Generator (W3.5), not an op."""
    with pytest.raises(RuntimeError):
        torch.randn(4, device="vortex")
    with pytest.raises(RuntimeError, match="no vortex implementation"):
        torch.ones(4).to("vortex").normal_()


def test_max_pool_return_indices_matches_cpu(backend):
    x = torch.tensor([[[[1.0, 5.0, 5.0],
                        [9.0, 2.0, 3.0],
                        [4.0, 8.0, 7.0]]]])
    got, got_idx = torch.nn.functional.max_pool2d(
        x.to("vortex"), 2, stride=1, return_indices=True)
    want, want_idx = torch.nn.functional.max_pool2d(
        x, 2, stride=1, return_indices=True)
    assert_matches_cpu(got, want, rtol=0, atol=0)
    assert_matches_cpu(got_idx, want_idx, rtol=0, atol=0)


def test_interpolate_nearest_modes_match_cpu(backend):
    x = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    for mode in ("nearest", "nearest-exact"):
        for size in ((5, 7), (2, 3)):
            got = torch.nn.functional.interpolate(x.to("vortex"), size=size,
                                                   mode=mode).cpu()
            want = torch.nn.functional.interpolate(x, size=size, mode=mode)
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        got = torch.nn.functional.interpolate(x.to("vortex"),
                                              scale_factor=(1.7, 1.5),
                                              mode=mode).cpu()
        want = torch.nn.functional.interpolate(x, scale_factor=(1.7, 1.5), mode=mode)
        torch.testing.assert_close(got, want, rtol=0, atol=0)
