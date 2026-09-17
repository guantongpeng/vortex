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

"""Inference batch norm (W1.3).

The channel used to be derived as (idx / (total/c)) % C, which is N*H*W rather
than the spatial span H*W and therefore picks the right channel only when N=1.
Every test here uses N > 1 and per-channel-distinct parameters, because the
default BatchNorm buffers (mean=0, var=1, weight=1, bias=0) hide the error even
at N>1: a wrong channel with identical parameters produces identical output.

The second thing pinned here is that no byte crosses the bus: the old path
copied running_var to the host, took a sqrt, and copied it back on every call.
"""

import pytest
import torch
import torch.nn as nn

from helpers import assert_matches_cpu, assert_no_transfers, assert_rejected


def randomize_bn(bn, seed):
    """Give every channel its own parameters so a wrong channel is visible."""
    g = torch.Generator().manual_seed(seed)
    c = bn.running_mean.numel()
    bn.running_mean.copy_(torch.randn(c, generator=g))
    bn.running_var.copy_(torch.rand(c, generator=g) + 0.5)
    if bn.weight is not None:  # affine=False has neither weight nor bias
        bn.weight.data.copy_(torch.randn(c, generator=g))
        bn.bias.data.copy_(torch.randn(c, generator=g))
    return bn


@pytest.mark.parametrize("n", [1, 2, 3])
def test_batch_norm_matches_cpu(backend, n):
    # non-square spatial dims on purpose: H*W and W*H must both be exercised
    x = torch.randn(n, 3, 3, 5)
    bn = randomize_bn(nn.BatchNorm2d(3), seed=100 + n).eval()
    ref = bn(x)
    assert_matches_cpu(bn.to("vortex")(x.to("vortex")), ref, rtol=1e-5, atol=1e-6)


def test_batch_norm_distinct_channels_within_one_sample(backend):
    """C=4, H=W=1: total/c == H*W == 1, so this isolates the channel *order*."""
    x = torch.randn(2, 4, 1, 2)
    bn = randomize_bn(nn.BatchNorm2d(4), seed=7).eval()
    ref = bn(x)
    assert_matches_cpu(bn.to("vortex")(x.to("vortex")), ref, rtol=1e-5, atol=1e-6)


def test_batch_norm_optional_affine(backend):
    x = torch.randn(2, 3, 4, 4)
    bn = randomize_bn(nn.BatchNorm2d(3, affine=False), seed=11).eval()
    ref = bn(x)
    assert_matches_cpu(bn.to("vortex")(x.to("vortex")), ref, rtol=1e-5, atol=1e-6)


def test_batch_norm_eps(backend):
    x = torch.randn(2, 3, 4, 4)
    for eps in (1e-5, 1e-3):
        bn = randomize_bn(nn.BatchNorm2d(3, eps=eps), seed=13).eval()
        ref = bn(x)
        assert_matches_cpu(bn.to("vortex")(x.to("vortex")), ref,
                           rtol=1e-5, atol=1e-6)


def test_batch_norm_no_host_roundtrip(backend):
    """Steady-state BN must not move bytes: no D2H, no H2D, no new allocation."""
    x = torch.randn(2, 3, 4, 4).to("vortex")
    bn = randomize_bn(nn.BatchNorm2d(3), seed=17).eval().to("vortex")

    backend.reset_stats()
    for _ in range(10):
        bn(x)
    st = backend.stats()
    assert_no_transfers({"h2d_bytes": 0, "d2h_bytes": 0}, st, "batch_norm")
    # weights and running stats were moved to the device before the window
    assert st["allocations"] <= 10, (
        "batch norm is allocating per call: %r" % st)


def test_batch_norm_rejects_training(backend):
    # Called directly: nn.BatchNorm2d in train mode decomposes into other aten
    # ops before it ever reaches native_batch_norm.
    x = torch.randn(2, 3, 4, 4).to("vortex")
    mean = torch.zeros(3, device="vortex")
    var = torch.ones(3, device="vortex")
    w = torch.ones(3, device="vortex")
    b = torch.zeros(3, device="vortex")
    with assert_rejected("training is unsupported", backend):
        torch.native_batch_norm(x, w, b, mean, var, True, 0.1, 1e-5)


def test_batch_norm_rejects_bad_parameters(backend):
    x = torch.randn(2, 3, 4, 4).to("vortex")
    mean = torch.zeros(3, device="vortex")
    var = torch.ones(3, device="vortex")
    w = torch.ones(3, device="vortex")
    b = torch.zeros(3, device="vortex")

    bad_len = torch.zeros(5, device="vortex")
    on_cpu = torch.zeros(3)

    with assert_rejected("has 5 elements", backend):
        torch.native_batch_norm(x, w, b, bad_len, var, False, 0.1, 1e-5)
    with assert_rejected("rather than the vortex device", backend):
        torch.native_batch_norm(x, w, b, on_cpu, var, False, 0.1, 1e-5)
