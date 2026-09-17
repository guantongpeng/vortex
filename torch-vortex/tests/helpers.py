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

"""Shared assertion helpers.

The point of most of these is to make a claim *checkable* rather than merely
plausible: that a rejected call never touched the device, that a sign of zero
survived, that no byte crossed the bus.
"""

import contextlib

import pytest
import torch


def to_vortex(t):
    return t.to("vortex")


def assert_matches_cpu(got, want, rtol=1e-6, atol=1e-7, equal_nan=True):
    """Compare a device result against a CPU reference after a D2H copy."""
    assert got.device.type == "vortex", "result escaped to %s" % got.device
    torch.testing.assert_close(got.cpu(), want.cpu(), rtol=rtol, atol=atol,
                               equal_nan=equal_nan)


def signbit_of(t):
    """Sign bits as a Python list; assert_close cannot see the sign of zero."""
    return torch.signbit(t.cpu().to(torch.float32)).tolist()


@contextlib.contextmanager
def assert_rejected(match, backend, may_allocate=False):
    """The op must refuse loudly, and must not have run anything.

    `may_allocate` relaxes the allocation check for paths that legitimately
    size an output before validating (empty_like precedes a shape check, for
    instance); the launch and transfer checks always apply.
    """
    before = backend.stats()
    with pytest.raises(RuntimeError, match=match):
        yield
    after = backend.stats()
    assert after["launches"] == before["launches"], (
        "rejected call still launched a kernel: %r -> %r" % (before, after))
    for key in ("h2d_bytes", "d2h_bytes", "d2d_bytes"):
        assert after[key] == before[key], (
            "rejected call still moved bytes (%s): %r -> %r"
            % (key, before, after))
    if not may_allocate:
        assert after["allocations"] == before["allocations"], (
            "rejected call still allocated: %r -> %r" % (before, after))


def assert_no_transfers(before, after, what=""):
    for key in ("h2d_bytes", "d2h_bytes"):
        assert after[key] == before[key], (
            "%s performed a host round-trip (%s: %d -> %d)"
            % (what, key, before[key], after[key]))
