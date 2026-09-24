# Copyright © 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import pytest
import torch


def test_group_norm_with_and_without_affine(backend):
    x = torch.randn(2, 4, 3, 5)
    for groups in (1, 2, 4):
        weight = torch.randn(4)
        bias = torch.randn(4)
        got = torch.nn.functional.group_norm(x.to("vortex"), groups,
                                             weight.to("vortex"), bias.to("vortex"),
                                             eps=1e-5)
        want = torch.nn.functional.group_norm(x, groups, weight, bias, eps=1e-5)
        torch.testing.assert_close(got.cpu(), want, rtol=1e-5, atol=1e-5)
        got = torch.nn.functional.group_norm(x.to("vortex"), groups, None, None,
                                             eps=1e-5)
        want = torch.nn.functional.group_norm(x, groups, None, None, eps=1e-5)
        torch.testing.assert_close(got.cpu(), want, rtol=1e-5, atol=1e-5)


def test_group_norm_rejects_bad_groups(backend):
    x = torch.randn(1, 4, 2, 2).to("vortex")
    with pytest.raises(RuntimeError, match="groups must divide"):
        torch.nn.functional.group_norm(x, 3)


@pytest.mark.parametrize("shape", [(3, 4), (2, 4, 7), (1, 4, 2, 3, 2)])
@pytest.mark.parametrize("affine", ["weight", "bias"])
def test_group_norm_spatial_ranks_and_optional_affine(backend, shape, affine):
    x = torch.randn(shape)
    weight = torch.randn(4) if affine == "weight" else None
    bias = torch.randn(4) if affine == "bias" else None
    got = torch.nn.functional.group_norm(
        x.to("vortex"), 2,
        weight.to("vortex") if weight is not None else None,
        bias.to("vortex") if bias is not None else None)
    want = torch.nn.functional.group_norm(x, 2, weight, bias)
    torch.testing.assert_close(got.cpu(), want, rtol=1e-5, atol=1e-5)


def test_group_norm_empty_batch_no_launch(backend):
    x = torch.empty((0, 4, 3, 2), device="vortex")
    backend.reset_stats()
    assert torch.nn.functional.group_norm(x, 2).shape == x.shape
    assert backend.stats()["launches"] == 0
