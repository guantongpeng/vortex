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

"""torch-vortex basic acceptance (plan P5-01): device registration,
device-memory allocation, CPU<->vortex copies and eager add/mul/fill
against CPU references. Every op used here must run on the device —
a CPU fallback in this test is a failure, not a shortcut."""

import os
import sys

import pytest
import torch


@pytest.fixture(scope="module")
def backend():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import torch_vortex  # noqa: F401  (loads ext + ops image)

    # torch 2.4: rename_privateuse1_backend alone; 2.5+ adds .vortex()
    assert "vortex" in str(torch.tensor(1, device="vortex").device)
    return torch_vortex


def test_device_registration(backend):
    t = torch.zeros(4, device="vortex")
    assert t.device.type == "vortex"


def test_alloc_free(backend):
    t = torch.empty(1024, device="vortex")
    assert t.numel() == 1024
    del t


def test_h2d_d2h_roundtrip(backend):
    ref = torch.arange(8, dtype=torch.float32) * 0.5
    t = ref.to("vortex")
    back = t.cpu()
    assert torch.equal(ref, back)


def test_fill(backend):
    t = torch.full((16,), 3.25, device="vortex")
    back = t.cpu()
    assert torch.all(back == 3.25)


def test_add(backend):
    a = torch.randn(65, dtype=torch.float32)  # non-multiple-of-block
    b = torch.randn(65, dtype=torch.float32)
    va, vb = a.to("vortex"), b.to("vortex")
    out = va + vb
    assert out.device.type == "vortex"
    torch.testing.assert_close(out.cpu(), a + b, rtol=1e-6, atol=1e-7)


def test_mul(backend):
    a = torch.randn(77, dtype=torch.float32)
    b = torch.randn(77, dtype=torch.float32)
    out = a.to("vortex") * b.to("vortex")
    torch.testing.assert_close(out.cpu(), a * b, rtol=1e-6, atol=1e-7)


def test_promotes_float64_loudly(backend):
    a = torch.ones(4, dtype=torch.float64).to("vortex")
    b = torch.ones(4, dtype=torch.float64).to("vortex")
    out = a + b
    assert out.dtype == torch.float64
    torch.testing.assert_close(out.cpu(), a.cpu() + b.cpu())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
