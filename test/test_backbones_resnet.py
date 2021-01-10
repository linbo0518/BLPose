"""BLPose Test: ResNet

Author: Bo Lin (@linbo0518)
Date: 2020-12-08
"""

import sys
import torch
from test import test_backbone

sys.path.append("../")
from blpose.utils.profiler import Profiler
from blpose.backbones.resnet import *

p = Profiler()
x = torch.randn(8, 3, 224, 224, dtype=torch.float32)

net = ResNet18()
with p.profiling("Test ResNet18"):
    test_backbone(net, x, 8, 16, 32)

net = ResNet34()
with p.profiling("Test ResNet34"):
    test_backbone(net, x, 8, 16, 32)

net = ResNet50()
with p.profiling("Test ResNet50"):
    test_backbone(net, x, 8, 16, 32)

net = ResNet101()
with p.profiling("Test ResNet101"):
    test_backbone(net, x, 8, 16, 32)

net = ResNet152()
with p.profiling("Test ResNet152"):
    test_backbone(net, x, 8, 16, 32)
