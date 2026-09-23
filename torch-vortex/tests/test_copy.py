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

"""copy_ / to() (W1.4).

The old implementation memcpy'd self.nbytes() bytes with no dtype, shape or
stride check, so a transposed source copied storage order instead of the
logical tensor. That is the regression test_transposed_source_is_materialised
below covers.
"""

import pytest
import torch

from helpers import assert_rejected


def test_roundtrip_is_bit_exact(backend):
    x = torch.arange(64, dtype=torch.float32) * 0.25 - 3.0
    back = x.to("vortex").cpu()
    assert torch.equal(back, x)


def test_transposed_source_is_materialised(backend):
    """A non-contiguous source must copy its logical contents, not storage order.

    `t.t().to("vortex")` asks for a *strided* destination (ATen's _to_copy
    preserves the source layout), which v1 refuses -- loudly, and that
    refusal is asserted below. The copy itself is exercised through copy_ into
    a contiguous destination, which is the case that used to silently copy
    storage order.
    """
    t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    non_contig = t.t()
    assert not non_contig.is_contiguous()

    dst = torch.empty(4, 3, device="vortex")
    dst.copy_(non_contig)
    assert torch.equal(dst.cpu(), non_contig.contiguous())

    # The strided destination _to_copy asks for is served by the copy kernel
    # now (it used to be refused).
    assert torch.equal(non_contig.to("vortex").cpu(), non_contig.contiguous())


def test_sliced_source(backend):
    x = torch.randn(16)
    assert torch.equal(x[2:6].to("vortex").cpu(), x[2:6])
    assert torch.equal(x[::2].to("vortex").cpu(), x[::2])


def test_device_to_device(backend):
    a = torch.randn(32).to("vortex")
    b = torch.empty(32, device="vortex")
    b.copy_(a)
    assert torch.equal(b.cpu(), a.cpu())


def test_empty_tensor(backend):
    backend.reset_stats()
    x = torch.empty(0).to("vortex")
    assert x.numel() == 0
    assert x.cpu().shape == (0,)


def test_self_copy_is_a_noop(backend):
    x = torch.randn(8).to("vortex")
    before = x.cpu()
    backend.reset_stats()
    x.copy_(x)
    st = backend.stats()
    assert torch.equal(x.cpu(), before)
    assert st["d2d_bytes"] == 0, "self copy moved bytes"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int32,
                                    torch.int64, torch.bool])
def test_dtype_conversion_roundtrip(backend, dtype):
    src = torch.tensor([-3.5, -1.0, 0.0, 1.25, 7.0], dtype=torch.float32)
    want = src.to(dtype)
    got = torch.empty(src.shape, dtype=dtype, device="vortex")
    got.copy_(src)
    assert torch.equal(got.cpu(), want)

    back = torch.empty(src.shape, dtype=torch.float32, device="vortex")
    back.copy_(got)
    assert torch.equal(back.cpu(), want.to(torch.float32))


def test_broadcast_copy(backend):
    src = torch.tensor([[1.0, 2.0, 3.0]])
    dst = torch.empty((4, 3), device="vortex")
    dst.copy_(src)
    assert torch.equal(dst.cpu(), src.expand(4, 3))


def test_broadcast_copy_with_dtype_conversion(backend):
    src = torch.tensor([[1.5, -2.0]], dtype=torch.float32)
    dst = torch.empty((3, 2), dtype=torch.float16, device="vortex")
    dst.copy_(src)
    assert torch.equal(dst.cpu(), src.to(torch.float16).expand(3, 2))


def test_rejects_shape_mismatch(backend):
    dst = torch.empty(4, device="vortex")
    src = torch.ones(5, device="vortex")
    with assert_rejected("shape mismatch", backend):
        dst.copy_(src)


def test_strided_destination(backend):
    """A strided destination is served by the copy kernel, not refused.

    It used to be a TORCH_CHECK; the plan deferred strided destinations to
    W3.2, which is where they landed.
    """
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    dst = torch.empty_strided((4, 3), (1, 4), device="vortex")
    assert not dst.is_contiguous()
    dst.copy_(x.t())
    assert torch.equal(dst.cpu(), x.t().contiguous())


def test_rejects_strided_factory_output(backend):
    """A strided vortex tensor is constructible but unusable: say so early."""
    dst = torch.empty_strided((4, 3), (1, 4), device="vortex")
    with assert_rejected("contiguous", backend):
        torch.relu(dst)
