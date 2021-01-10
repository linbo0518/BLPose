"""BLPose Test: SE ResNet

Author: Bo Lin (@linbo0518)
Date: 2020-12-09
"""

import sys
import torch
from test import test_backbone

sys.path.append("../")
from blpose.utils.profiler import Profiler
from blpose.backbones.seresnet import *

p = Profiler()
x = torch.randn(8, 3, 224, 224, dtype=torch.float32)

net = SEResNet50()
with p.profiling("Test SEResNet50"):
    test_backbone(net, x, 8, 16, 32)

net = SEResNet101()
with p.profiling("Test SEResNet101"):
    test_backbone(net, x, 8, 16, 32)

net = SEResNet152()
with p.profiling("Test SEResNet152"):
    test_backbone(net, x, 8, 16, 32)
