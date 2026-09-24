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

"""Mini-ResNet end-to-end on the vortex device (plan P5-02).

A ResNet-family network (stem + basic blocks with residual adds +
downsampling shortcut + global average pool + linear head) executed entirely
through the registered PrivateUse1 ops on simx, against the CPU run of the
same module with the same weights.

Two things this test has to get right to be worth anything:

  * batch size > 1. The batch-norm channel index was derived as total/c, which
    is N*H*W rather than H*W, so it picked the right channel only at N == 1 and
    the original single-sample test could not see it.
  * non-default batch-norm buffers. With mean=0, var=1, weight=1, bias=0 every
    channel produces the same output, so even at N > 1 a wrong channel
    selection is invisible. Every BN here is randomised.

The model-level test is marked slow (it dominates the suite's runtime); the
milestone gate runs it with --tier=full.
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

from helpers import assert_matches_cpu


def randomize_batch_norms(module, seed):
    """Give every BatchNorm its own per-channel parameters.

    Randomising the buffers is what turns a wrong channel index into a wrong
    number rather than a masked one.
    """
    g = torch.Generator().manual_seed(seed)
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            c = m.running_mean.numel()
            m.running_mean.copy_(torch.randn(c, generator=g))
            m.running_var.copy_(torch.rand(c, generator=g) + 0.5)
            if m.weight is not None:
                m.weight.data.copy_(torch.randn(c, generator=g))
                m.bias.data.copy_(torch.randn(c, generator=g))
    return module


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False),
                nn.BatchNorm2d(cout),
            )

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(y + idt)


class MiniResNet(nn.Module):
    def __init__(self, ch=4, classes=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer1 = nn.Sequential(BasicBlock(ch, ch, 1))
        self.layer2 = BasicBlock(ch, 2 * ch, 2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2 * ch, classes),
        )

    def forward(self, x):
        return self.head(self.layer2(self.layer1(self.stem(x))))


class SmallResNet18(nn.Module):
    def __init__(self, classes=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 1, 3, 1, 1, bias=False), nn.BatchNorm2d(1),
            nn.ReLU(), nn.MaxPool2d(2))

        def layer(cin, cout, stride):
            return nn.Sequential(BasicBlock(cin, cout, stride),
                                  BasicBlock(cout, cout, 1))

        self.layer1 = layer(1, 1, 1)
        self.layer2 = layer(1, 2, 2)
        self.layer3 = layer(2, 4, 2)
        self.layer4 = layer(4, 8, 2)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(8, classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)


def build(seed, batch, ch=4, classes=2, size=8):
    """A CPU reference and a weight-identical vortex copy, plus an input."""
    torch.manual_seed(seed)
    ref_model = randomize_batch_norms(MiniResNet(ch, classes), seed).eval()
    dev_model = MiniResNet(ch, classes)
    dev_model.load_state_dict(ref_model.state_dict())
    dev_model = dev_model.eval().to("vortex")
    x = torch.randn(batch, 3, size, size)
    return ref_model, dev_model, x


@pytest.mark.slow
@pytest.mark.parametrize("batch", [1, 2, 3])
def test_logits_match_cpu(backend, batch):
    ref_model, dev_model, x = build(seed=20260915 + batch, batch=batch)
    with torch.inference_mode():
        ref = ref_model(x)
        got = dev_model(x.to("vortex"))
    assert got.device.type == "vortex"
    assert_matches_cpu(got, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.slow
def test_intermediate_layers_match_cpu(backend):
    """Parity layer by layer, so a wrong middle does not hide behind logits."""
    ref_model, dev_model, x = build(seed=7, batch=2)
    names = ["stem", "layer1", "layer2", "head"]
    ref_out, dev_out = {}, {}

    def capture(store):
        def hook(name):
            def fn(_module, _inputs, output):
                store[name] = output.detach()
            return fn
        return hook

    for name in names:
        getattr(ref_model, name).register_forward_hook(capture(ref_out)(name))
        getattr(dev_model, name).register_forward_hook(capture(dev_out)(name))

    with torch.inference_mode():
        ref_model(x)
        dev_model(x.to("vortex"))

    for name in names:
        assert name in dev_out, "%s never ran on the device" % name
        assert_matches_cpu(dev_out[name], ref_out[name], rtol=1e-3, atol=1e-3)


@pytest.mark.slow
def test_forward_uses_the_device(backend):
    """Every op in the forward path is device-registered, and it shows up.

    A silent CPU fallback would leave the launch count far below the number of
    operators in the graph, so counting is the assertion.
    """
    ref_model, dev_model, x = build(seed=99, batch=2)
    with torch.inference_mode():
        backend.reset_stats()
        out = dev_model(x.to("vortex"))
        st = backend.stats()

    assert out.shape == (2, 2)
    # stem 2 + layer1 6 + layer2 12 + head 2 operators, each at least one launch
    assert st["launches"] >= 20, (
        "only %d launches for a ResNet forward; something ran on the CPU: %r"
        % (st["launches"], st))


@pytest.mark.slow
def test_training_path_is_refused(backend):
    """Inference only. A backward through the device must not be silent."""
    _, dev_model, x = build(seed=5, batch=2)
    vx = x.to("vortex").requires_grad_()
    with pytest.raises(RuntimeError):
        dev_model(vx).sum().backward()


@pytest.mark.slow
def test_small_resnet18_matches_cpu(backend):
    torch.manual_seed(20260924)
    ref = randomize_batch_norms(SmallResNet18(), 20260924).eval()
    dev = SmallResNet18().eval()
    dev.load_state_dict(ref.state_dict())
    dev.to("vortex")
    x = torch.randn(1, 3, 8, 8)
    with torch.inference_mode():
        want = ref(x)
        got = dev(x.to("vortex"))
    assert_matches_cpu(got, want, rtol=2e-3, atol=2e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
