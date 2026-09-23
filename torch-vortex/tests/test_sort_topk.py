# Copyright © 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import torch


def test_sort_matches_cpu_on_last_and_middle_dimensions(backend):
    x = torch.tensor([[3.0, -1.0, 2.0], [4.0, 0.5, 4.0]])
    got_v, got_i = torch.sort(x.to("vortex"), dim=1)
    want_v, want_i = torch.sort(x, dim=1)
    torch.testing.assert_close(got_v.cpu(), want_v)
    torch.testing.assert_close(got_i.cpu(), want_i)

    y = torch.tensor([[[2.0, -1.0], [3.0, 0.0], [1.0, 4.0]],
                      [[-2.0, 5.0], [7.0, 6.0], [8.0, -3.0]]])
    got_v, got_i = torch.sort(y.to("vortex"), dim=0, descending=True)
    want_v, want_i = torch.sort(y, dim=0, descending=True)
    torch.testing.assert_close(got_v.cpu(), want_v)
    torch.testing.assert_close(got_i.cpu(), want_i)


def test_topk_largest_smallest_and_zero_k(backend):
    x = torch.tensor([[1.0, 9.0, 3.0, 7.0], [4.0, -2.0, 8.0, 0.0]])
    for largest in (True, False):
        got_v, got_i = torch.topk(x.to("vortex"), 2, dim=1,
                                  largest=largest, sorted=True)
        want_v, want_i = torch.topk(x, 2, dim=1, largest=largest, sorted=True)
        torch.testing.assert_close(got_v.cpu(), want_v)
        torch.testing.assert_close(got_i.cpu(), want_i)

    values, indices = torch.topk(x.to("vortex"), 0, dim=1)
    assert values.shape == (2, 0) and indices.shape == (2, 0)


def test_sort_and_topk_out_overloads(backend):
    x = torch.tensor([[4.0, 1.0, 3.0, 2.0]])
    values = torch.empty_like(x).to("vortex")
    indices = torch.empty(x.shape, dtype=torch.long, device="vortex")
    rv, ri = torch.sort(x.to("vortex"), out=(values, indices))
    assert rv.data_ptr() == values.data_ptr() and ri.data_ptr() == indices.data_ptr()
    torch.testing.assert_close(values.cpu(), torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    assert indices.cpu().tolist() == [[1, 3, 2, 0]]

    tv = torch.empty((1, 2), device="vortex")
    ti = torch.empty((1, 2), dtype=torch.long, device="vortex")
    rv, ri = torch.topk(x.to("vortex"), 2, out=(tv, ti))
    assert rv.data_ptr() == tv.data_ptr() and ri.data_ptr() == ti.data_ptr()
    assert tv.cpu().tolist() == [[4.0, 3.0]]
    assert ti.cpu().tolist() == [[0, 2]]
