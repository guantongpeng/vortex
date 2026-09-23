# Copyright © 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""FP32 nll_loss_forward semantics on the Vortex device."""

import pytest
import torch

from helpers import assert_matches_cpu


def v(t):
    return t.to("vortex")


def test_nll_loss_reductions_and_weight(backend):
    logp = torch.log_softmax(torch.randn(3, 4), dim=1)
    target = torch.tensor([1, 3, 0], dtype=torch.int64)
    weight = torch.tensor([1.0, 0.5, 2.0, 1.5])
    for reduction in ("none", "sum", "mean"):
        got = torch.nn.functional.nll_loss(v(logp), v(target), v(weight),
                                           reduction=reduction)
        want = torch.nn.functional.nll_loss(logp, target, weight,
                                            reduction=reduction)
        assert_matches_cpu(got, want, rtol=1e-6, atol=1e-6)


def test_nll_loss_1d_and_ignore_index(backend):
    logp = torch.log_softmax(torch.randn(5), dim=0)
    target = torch.tensor(2, dtype=torch.int64)
    assert_matches_cpu(torch.nn.functional.nll_loss(v(logp), v(target)),
                       torch.nn.functional.nll_loss(logp, target),
                       rtol=1e-6, atol=1e-6)

    logp2 = torch.log_softmax(torch.randn(3, 5), dim=1)
    target2 = torch.tensor([1, -100, 3], dtype=torch.int64)
    got = torch.nn.functional.nll_loss(v(logp2), v(target2), ignore_index=-100,
                                       reduction="mean")
    want = torch.nn.functional.nll_loss(logp2, target2, ignore_index=-100,
                                        reduction="mean")
    assert_matches_cpu(got, want, rtol=1e-6, atol=1e-6)


def test_nll_loss_rejects_bad_target(backend):
    logp = torch.log_softmax(torch.randn(2, 4), dim=1)
    target = torch.tensor([0, 4], dtype=torch.int64)
    before = backend.stats()
    with pytest.raises(RuntimeError, match="out-of-range"):
        torch.nn.functional.nll_loss(v(logp), v(target))
    assert backend.stats()["launches"] == before["launches"] + 1
