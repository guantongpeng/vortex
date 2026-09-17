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

"""Factory dispatch isolation (W1.6).

The extension used to register empty/empty_strided on the BackendSelect key,
which replaces ATen's own backend-select kernel process-wide. Every non-vortex
factory call was then answered by at::detail::empty_cpu: torch.empty(3),
torch.zeros, torch.arange and device='meta' included.

The baseline is taken in a subprocess *before* torch_vortex is imported and
compared against the same probe afterwards, so the assertion is "importing
this backend changes nothing about other devices" rather than a guess about
what those calls used to return.
"""

import json
import subprocess
import sys
import textwrap

import pytest
import torch

PROBE = textwrap.dedent("""
    import json, torch
    out = {}
    def rec(k, t):
        out[k] = [str(t.dtype), t.device.type, list(t.shape), list(t.stride())]
    rec("empty3", torch.empty(3))
    rec("zeros3", torch.zeros(3))
    rec("arange3", torch.arange(3))
    rec("meta", torch.empty(3, device="meta"))
    rec("empty0", torch.empty(0))
    rec("f64", torch.empty(3, dtype=torch.float64))
    rec("strided", torch.empty_strided((2, 3), (3, 1)))
    out["zeros_vals"] = torch.zeros(3).tolist()
    out["arange_vals"] = torch.arange(3).tolist()
    print(json.dumps(out))
""")


def _run_probe(with_backend):
    src = PROBE if not with_backend else "import torch_vortex\n" + PROBE
    res = subprocess.run([sys.executable, "-c", src], capture_output=True,
                         text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-3000:]
    return json.loads(res.stdout.strip().splitlines()[-1])


def test_cpu_and_meta_factories_are_untouched(backend):
    before = _run_probe(with_backend=False)
    after = _run_probe(with_backend=True)
    diffs = {k: (before[k], after.get(k)) for k in before if before[k] != after.get(k)}
    assert not diffs, "importing torch_vortex changed factory behaviour: %r" % diffs


def test_meta_stays_on_meta(backend):
    t = torch.empty(3, device="meta")
    assert t.device.type == "meta"


def test_vortex_factories(backend):
    assert torch.empty(3, device="vortex").device.type == "vortex"
    assert torch.zeros(3, device="vortex").device.type == "vortex"


def test_faketensor_does_no_device_work(backend):
    from torch._subclasses.fake_tensor import FakeTensorMode

    backend.reset_stats()
    with FakeTensorMode():
        t = torch.empty(4, 6, device="vortex")
        _ = t.view(24)
    st = backend.stats()
    assert st["launches"] == 0 and st["allocations"] == 0, (
        "FakeTensor performed real device work: %r" % st)


def test_bad_device_index_is_rejected(backend):
    with pytest.raises(RuntimeError, match="device 0"):
        torch.empty(3, device="vortex:1")


def test_unimplemented_op_raises_by_name(backend):
    """The fallback must throw with the operator name, not run on the CPU."""
    x = torch.randn(4).to("vortex")
    with pytest.raises(RuntimeError, match="no vortex implementation"):
        torch.fft.fft(x)


def test_backward_does_not_silently_discard(backend):
    """Inference-only: a gradient through a vortex tensor must not be silent."""
    x = torch.randn(4, requires_grad=True).to("vortex").requires_grad_()
    with pytest.raises(RuntimeError):
        torch.relu(x).sum().backward()
