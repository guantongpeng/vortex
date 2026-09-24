# Copyright © 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import torch


def v(t):
    return t.to("vortex")


def test_clone_copies_without_aliasing(backend):
    x = v(torch.arange(12, dtype=torch.float32).reshape(3, 4))
    y = x.clone()
    assert y.device.type == "vortex"
    assert y.data_ptr() != x.data_ptr()
    assert torch.equal(y.cpu(), x.cpu())
    y.fill_(9)
    assert torch.equal(x.cpu(), torch.arange(12, dtype=torch.float32).reshape(3, 4))


def test_clone_preserves_strided_layout_by_default(backend):
    x = v(torch.arange(12, dtype=torch.float32).reshape(3, 4)).t()
    y = x.clone()
    assert y.stride() == x.stride()
    assert not y.is_contiguous()
    assert torch.equal(y.cpu(), x.cpu())


def test_clone_can_request_contiguous_memory_format(backend):
    x = v(torch.arange(12, dtype=torch.float32).reshape(3, 4)).t()
    y = x.clone(memory_format=torch.contiguous_format)
    assert y.is_contiguous()
    assert torch.equal(y.cpu(), x.cpu())


def test_contiguous_returns_alias_only_when_already_contiguous(backend):
    x = v(torch.arange(12, dtype=torch.float32).reshape(3, 4))
    assert x.contiguous().data_ptr() == x.data_ptr()
    t = x.t()
    y = t.contiguous()
    assert y.data_ptr() != t.data_ptr()
    assert y.is_contiguous()
    assert torch.equal(y.cpu(), t.cpu())
