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
downsampling shortcut + global average pool + linear head) executed
entirely through the registered PrivateUse1 ops on simx. The CPU run of
the same module with the same weights is the reference. Small channels
on purpose: simx is a functional simulator, and the milestone is
numerical parity, not throughput.
"""

import os
import sys

import pytest
import torch
import torch.nn as nn


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
    def __init__(self, ch=8, classes=4):
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


@pytest.fixture(scope="module")
def model_and_input():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    torch.manual_seed(20260915)
    m = MiniResNet(ch=8, classes=4)
    x = torch.randn(1, 3, 16, 16)
    return m, x


def test_mini_resnet_device_matches_cpu(model_and_input):
    import torch_vortex  # noqa: F401

    m, x = model_and_input
    ref = m.eval()(x)                       # CPU reference
    md = MiniResNet(ch=8, classes=4)
    md.load_state_dict(m.state_dict())      # identical weights
    out = md.eval().to("vortex")(x.to("vortex"))
    assert out.device.type == "vortex"
    got = out.cpu()
    torch.testing.assert_close(got, ref, rtol=1e-3, atol=1e-3)
    print("mini-resnet logits device:", got.tolist())
    print("mini-resnet logits cpu   :", ref.tolist())


def test_no_silent_cpu_ops(model_and_input):
    # The forward must not raise NotImplementedError anywhere — i.e. every
    # op in the ResNet path is device-registered (the loud-failure rule).
    import torch_vortex  # noqa: F401

    m, x = model_and_input
    md = MiniResNet(ch=8, classes=4)
    md.load_state_dict(m.state_dict())
    out = md.eval().to("vortex")(x.to("vortex"))
    assert out.shape == (1, 4)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
