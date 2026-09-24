# Copyright © 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import pytest
import torch


def test_dropout_eval_is_device_identity(backend):
    x = torch.randn(2, 3).to("vortex")
    backend.reset_stats()
    y = torch.dropout(x, 0.5, False)
    assert y.data_ptr() == x.data_ptr()
    assert backend.stats()["launches"] == 0
    torch.testing.assert_close(y.cpu(), x.cpu())


def test_dropout_training_is_rejected(backend):
    x = torch.randn(2, 3).to("vortex")
    with pytest.raises(RuntimeError, match="training is unsupported"):
        torch.dropout(x, 0.5, True)
