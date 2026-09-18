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

"""Memory lifetime (W2.1).

A kernel launch returns before the kernel runs, and the c10 allocator's
deleter used to call hipFree, which returns the address to the device
allocator's free list immediately. Measured on the previous behaviour: a
32768-element relu recycled its input's address 44 us after the launch
returned, with 11.6 s of kernel still to run.

The corruption is not directly observable -- the runtime posts every command
into one in-order ring, so the recycler happens to get the address after the
reader is done -- so these tests assert the mechanism instead: a free issued
while work is outstanding must be deferred, and the address must not come back
until it has been.
"""

import gc

import pytest
import torch


def test_free_is_deferred_while_work_is_outstanding(backend):
    n = 1 << 13
    a = torch.ones(n, device="vortex")
    b = torch.ones(n, device="vortex")
    torch.vortex.synchronize()          # start from a drained queue
    held = a.data_ptr()
    backend.reset_stats()

    out = a + b                         # enqueued; the launch does not wait
    del a
    gc.collect()
    fresh = torch.empty(n, device="vortex")

    st = backend.stats()
    assert st["frees"] > 0, "the tensor was never freed; test is not measuring anything"
    assert st["frees"] > st["immediate_frees"], (
        "the free ran immediately despite an outstanding launch: %r" % st)
    assert fresh.data_ptr() != held, (
        "the address was handed back while a kernel could still read it")

    # and the work is still correct once it retires
    assert bool((out.cpu() == 2.0).all())
    del fresh, b, out
    gc.collect()


# Ops that submit their kernel through the sw/dl library rather than through
# this extension's own launch helper. Each takes the input and a buffer of the
# same element count and returns without waiting.
DL_OPS = {
    "conv2d": lambda x, w: torch.nn.functional.conv2d(x, w, padding=1),
}


@pytest.mark.parametrize("opus", DL_OPS.values(), ids=list(DL_OPS))
def test_free_is_deferred_while_a_dl_kernel_is_outstanding(backend, opus):
    """The same lifetime rule, for the launch path that bypasses launch().

    W3.1 moved conv, pooling, batch norm and matmul into sw/dl. Those entry
    points enqueue on the queue directly, so they never pass through the helper
    where the deferred free's epochs are kept -- a free after one of them took
    the immediate hipFree path, with the kernel still queued to read the buffer.

    Everything the device does before the op under test is built and retired
    first, and the counters are reset after: if any other launch were still
    outstanding, the epochs would be dirty for a reason that has nothing to do
    with the DL path, and this would pass without testing anything.
    """
    n, c, h, w = 1, 4, 32, 32
    x = torch.ones(n, c, h, w, device="vortex")
    weight = torch.ones(4, c, 3, 3, device="vortex")
    torch.vortex.synchronize()
    held = x.data_ptr()
    backend.reset_stats()

    out = opus(x, weight)
    del x
    gc.collect()
    fresh = torch.empty(n * c * h * w, device="vortex")

    st = backend.stats()
    assert st["frees"] > 0, "the tensor was never freed; test is not measuring anything"
    assert st["frees"] > st["immediate_frees"], (
        "the free ran immediately despite a DL launch still outstanding: %r" % st)
    assert fresh.data_ptr() != held, (
        "the address was handed back while a DL kernel could still read it")

    del fresh, weight, out
    gc.collect()


def test_free_is_deferred_after_a_strided_copy(backend):
    """A strided D2D copy enqueues a kernel; it does not drain the queue.

    This branch used to end in a hipMemcpy, which enqueues and waits, so it
    announced that the queue had drained. When the copy became a kernel launch
    the announcement stayed, and it clears the allocator's epoch -- so the
    source, which in a moved softmax is owned only by a C++ local, was freed
    immediately while the copy was still queued to read it.
    """
    base = torch.randn(6, 4).to("vortex")
    dst = torch.empty(4, 6, device="vortex")
    torch.vortex.synchronize()
    backend.reset_stats()
    dst.copy_(base.t())            # strided on both sides: a kernel, not a memcpy
    del base
    gc.collect()
    st = backend.stats()
    assert st["launches"] > 0, "the strided copy did not launch: %r" % st
    assert st["frees"] > 0, "nothing was freed; test is not measuring anything"
    assert st["frees"] > st["immediate_frees"], (
        "the free ran immediately despite the copy still being queued: %r" % st)
    del dst
    gc.collect()


def test_free_is_immediate_once_the_queue_is_drained(backend):
    """The fast path: no outstanding work means no deferral is needed.

    Deferring every free would be correct but costly -- no address would ever
    come back to the pool. This asserts the mechanism rather than the address:
    which address the runtime's first-fit allocator hands out next depends on
    its free-list state, so pinning the pointer would be testing that instead.
    """
    n = 1 << 12
    a = torch.ones(n, device="vortex")
    torch.vortex.synchronize()          # drain, so the queue is provably idle
    backend.reset_stats()
    del a
    gc.collect()

    st = backend.stats()
    assert st["frees"] > 0, "nothing was freed; test is not measuring anything"
    assert st["immediate_frees"] == st["frees"], (
        "a free on a drained queue was deferred anyway: %r" % st)


def test_repeated_churn_does_not_grow_the_address_space(backend):
    """Allocate/free in a loop and check addresses come back around."""
    n = 1 << 12
    seen = set()
    for _ in range(12):
        t = torch.empty(n, device="vortex")
        seen.add(t.data_ptr())
        del t
    assert len(seen) < 12, (
        "12 allocate/free rounds produced %d distinct addresses; the allocator "
        "is not reusing anything" % len(seen))
