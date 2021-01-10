"""BLPose Test: VGG

Author: Bo Lin (@linbo0518)
Date: 2020-12-07
"""

import sys
import torch
from test import test_backbone

sys.path.append("../")
from blpose.utils.profiler import Profiler
from blpose.backbones.vgg import *

p = Profiler()
x = torch.randn(8, 3, 224, 224, dtype=torch.float32)

net = VGG11()
with p.profiling("Test VGG11"):
    test_backbone(net, x, 8, 16)

net = VGG13()
with p.profiling("Test VGG13"):
    test_backbone(net, x, 8, 16)

net = VGG16()
with p.profiling("Test VGG16"):
    test_backbone(net, x, 8, 16)

net = VGG19()
with p.profiling("Test VGG19"):
    test_backbone(net, x, 8, 16)
