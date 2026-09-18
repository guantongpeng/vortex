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


def test_conv_reports_lmem_exhaustion(backend):
    """A full-resolution ResNet stem needs 18 KiB against a 16 KiB limit."""
    prop = backend._ext.device_properties()
    limit = prop["shared_mem_per_block"]
    # pick a ci whose filter exceeds the limit, if one exists in range
    ci = limit // (3 * 3 * 4) + 1
    x = torch.randn(1, ci, 8, 8).to("vortex")
    w = torch.randn(4, ci, 3, 3).to("vortex")
    with assert_rejected("local memory", backend):
        torch.nn.functional.conv2d(x, w, padding=1)


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


def test_adaptive_avg_pool(backend):
    x = torch.randn(2, 3, 5, 7)
    assert_matches_cpu(torch.nn.functional.adaptive_avg_pool2d(x.to("vortex"), 1),
                       torch.nn.functional.adaptive_avg_pool2d(x, 1),
                       rtol=1e-5, atol=1e-6)


def test_avg_pool2d_is_not_registered(backend):
    """A documented boundary: only adaptive_avg_pool2d(1) is in the v1 op set."""
    x = torch.randn(2, 3, 4, 4).to("vortex")
    with pytest.raises(RuntimeError, match="no vortex implementation"):
        torch.nn.functional.avg_pool2d(x, 2)


def test_device_rng_is_refused_by_name(backend):
    """torch.randn(device='vortex') needs a Generator (W3.5), not an op."""
    with pytest.raises(RuntimeError):
        torch.randn(4, device="vortex")
    with pytest.raises(RuntimeError, match="no vortex implementation"):
        torch.ones(4).to("vortex").normal_()
