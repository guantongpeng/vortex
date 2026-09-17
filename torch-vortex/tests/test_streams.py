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

"""Streams, events and the device guard (W2.2).

What the guard used to do: report one process-global stream that exchangeStream
wrote and nothing read, hand out the default stream from getNewStream, and let
every launch pass nullptr. So two "streams" were equal, a stream context did
nothing, and all device work went to one queue regardless.

These tests assert the properties the plan asks for -- nested context restore,
two threads isolated, correct cross-stream event dependency -- plus the one
that makes streams mean anything: an op issued inside a stream context must
actually land on that stream.

Most of them observe through timing, because there is no non-blocking query on
this runtime (W2.2 records that gap). Each timing assertion calibrates against
its own idle measurement rather than a wall-clock constant.
"""

import threading
import time

import pytest
import torch


def _idle_sync_cost(stream):
    stream.synchronize()
    t0 = time.perf_counter()
    stream.synchronize()
    return time.perf_counter() - t0


def test_streams_are_distinct(backend):
    s1 = torch.Stream(device="vortex")
    s2 = torch.Stream(device="vortex")
    assert s1 != s2, "getNewStream handed out the same stream twice"
    assert s1.stream_id != s2.stream_id
    assert s1.device.type == "vortex"


def test_default_stream_is_id_zero(backend):
    assert torch.accelerator.current_stream().stream_id == 0


def test_context_manager_restores(backend):
    before = torch.accelerator.current_stream()
    s = torch.Stream(device="vortex")
    with s:
        assert torch.accelerator.current_stream() == s
    assert torch.accelerator.current_stream() == before


def test_nested_contexts_restore_in_order(backend):
    outer = torch.Stream(device="vortex")
    inner = torch.Stream(device="vortex")
    base = torch.accelerator.current_stream()
    with outer:
        assert torch.accelerator.current_stream() == outer
        with inner:
            assert torch.accelerator.current_stream() == inner
        assert torch.accelerator.current_stream() == outer
    assert torch.accelerator.current_stream() == base


def test_ops_run_on_the_current_stream(backend):
    """The point of the whole feature: the launch must follow the context.

    Compared against the same synchronize on an idle stream, so the threshold
    is a ratio. If launches ignored the context (they used to), both cases
    would return at once and the ratio would be ~1.
    """
    n = 1 << 13
    a = torch.ones(n, device="vortex")
    b = torch.ones(n, device="vortex")
    s = torch.Stream(device="vortex")
    torch.accelerator.synchronize()

    idle = _idle_sync_cost(s)

    # work issued elsewhere: this stream stays idle
    a + b
    t0 = time.perf_counter()
    s.synchronize()
    unrelated = time.perf_counter() - t0

    torch.accelerator.synchronize()
    with s:
        out = a + b
    t0 = time.perf_counter()
    s.synchronize()
    owing = time.perf_counter() - t0

    assert bool((out.cpu() == 2.0).all())
    assert unrelated < idle * 20 + 1e-3, (
        "synchronising an idle stream took %.6fs; it is not per-stream"
        % unrelated)
    assert owing > unrelated * 20, (
        "an op issued inside a stream context did not land on that stream: "
        "its synchronize took %.6fs against %.6fs for an unrelated stream"
        % (owing, unrelated))


def test_cross_stream_event_dependency(backend):
    """A later stream must be able to wait on an earlier stream's event."""
    n = 1 << 12
    a = torch.ones(n, device="vortex")
    b = torch.full((n,), 2.0, device="vortex")
    s1 = torch.Stream(device="vortex")
    s2 = torch.Stream(device="vortex")
    torch.accelerator.synchronize()

    with s1:
        first = a + b                 # 3.0 everywhere
        ev = torch.Event()
        ev.record(s1)

    s2.wait_event(ev)
    with s2:
        second = first * b            # 6.0 everywhere
    torch.accelerator.synchronize()

    assert bool((first.cpu() == 3.0).all())
    assert bool((second.cpu() == 6.0).all()), "cross-stream dependency lost"


def test_event_record_and_synchronize(backend):
    n = 1 << 12
    a = torch.ones(n, device="vortex")
    b = torch.ones(n, device="vortex")
    ev = torch.Event()
    torch.accelerator.synchronize()
    idle = _idle_sync_cost(torch.accelerator.current_stream())

    t0 = time.perf_counter()
    out = a + b
    ev.record()
    ev.synchronize()                  # must cover the add
    busy = time.perf_counter() - t0

    assert bool((out.cpu() == 2.0).all())
    assert busy > idle * 20 + 1e-3, (
        "Event.record/synchronize did not wait for the stream (%.6fs vs %.6fs "
        "idle)" % (busy, idle))


def test_two_threads_have_separate_current_streams(backend):
    """The guard's state is thread-local; it used to be one process-global."""
    seen = {}
    ready = threading.Barrier(2, timeout=60)

    def worker(name, stream):
        with stream:
            ready.wait()
            time.sleep(0.05)          # let the other thread set its own
            seen[name] = torch.accelerator.current_stream()
            ready.wait()

    t1 = threading.Thread(target=worker,
                          args=("a", torch.Stream(device="vortex")))
    t2 = threading.Thread(target=worker,
                          args=("b", torch.Stream(device="vortex")))
    t1.start()
    t2.start()
    t1.join(timeout=120)
    t2.join(timeout=120)

    assert seen["a"] != seen["b"], (
        "both threads saw the same current stream: %r" % (seen,))
    assert torch.accelerator.current_stream().stream_id == 0, (
        "a worker thread leaked its stream into the main thread")


def test_query_is_refused_loudly(backend):
    """No non-blocking query exists on this runtime; say so rather than guess."""
    s = torch.Stream(device="vortex")
    with pytest.raises(RuntimeError, match="doesn't support querying streams"):
        s.query()
