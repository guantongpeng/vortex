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

"""Device module, synchronisation and teardown (W2.2/W2.3).

The device module is where "the backend is usable" is observable from Python:
is_available, device_count, current_device, synchronize. Each one has to mean
something -- torch 2.14 calls is_available() during FakeTensor setup, so a
@property returning True is a TypeError rather than a cosmetic wart.
"""

import time

import pytest
import torch


def test_device_module_surface(backend):
    assert torch.vortex.is_available() is True
    assert torch.vortex.device_count() == 1
    assert torch.vortex.current_device() == 0
    assert torch.vortex.is_initialized() is True
    # Callable, not a property: torch's _is_privateuse1_backend_available does
    # `getattr(module, "is_available") and is_available()`.
    assert callable(torch.vortex.is_available)


def test_synchronize_actually_waits(backend):
    """synchronize() used to map onto vx_queue_flush, which waits for nothing.

    It is hard to observe "did it wait" directly, so this measures it: an idle
    synchronize returns in microseconds, while one issued after an
    asynchronous launch has to cover the kernel's runtime. The idle measurement
    calibrates the threshold, so the assertion is a ratio rather than a
    machine-specific constant.
    """
    torch.vortex.synchronize()
    t0 = time.perf_counter()
    torch.vortex.synchronize()
    idle = time.perf_counter() - t0

    n = 1 << 13
    a = torch.ones(n, device="vortex")
    b = torch.ones(n, device="vortex")
    torch.vortex.synchronize()

    t0 = time.perf_counter()
    out = a + b                     # enqueued; the launch itself does not wait
    launch_only = time.perf_counter() - t0
    torch.vortex.synchronize()      # this must cover the kernel
    busy = time.perf_counter() - t0

    assert bool((out.cpu() == 2.0).all()), "the kernel did not run correctly"
    assert launch_only < busy, (
        "the launch blocked, so this test cannot tell sync apart from it")
    assert busy > idle * 50, (
        "synchronize() returned in %.6fs having enqueued work that takes "
        "longer; it is not waiting (idle sync: %.6fs, %.1fx)"
        % (busy, idle, busy / max(idle, 1e-9)))


def test_device_index_is_single(backend):
    with pytest.raises(RuntimeError, match="device 0"):
        torch.empty(3, device="vortex:1")


def test_repeated_import_is_idempotent(backend):
    """Re-importing must not re-register the backend or reload the image.

    reload() resets the module's globals, so the idempotence guards cannot live
    there: the second import used to call generate_methods_for_privateuse1_backend
    again (RuntimeError) and then hipModuleLoad the same image again ("address
    range overlaps with existing allocation").
    """
    import importlib

    import torch_vortex

    before = torch_vortex._ext
    importlib.reload(torch_vortex)
    assert torch_vortex._ext is before
    assert torch.empty(2, device="vortex").device.type == "vortex"
    three = torch.full((4,), 3.0, device="vortex")
    assert bool(((torch.ones(4, device="vortex") * three).cpu() == 3.0).all())


def test_second_different_image_is_refused(backend):
    """One process, one image -- until W2.4.

    Silently reusing the first image's kernels would run the wrong code, so a
    different path must be an error rather than a no-op.
    """
    with pytest.raises(RuntimeError, match="already loaded the kernel image"):
        backend._ext.load_ops("/nonexistent/other.vxbin", "")
