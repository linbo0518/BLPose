"""BLPose Test: MobileNet

Author: Bo Lin (@linbo0518)
Date: 2020-12-29
"""

import sys
import torch
from test import test_backbone

sys.path.append("../")
from blpose.utils.profiler import Profiler
from blpose.backbones.mobilenet import *

p = Profiler()
x = torch.randn(8, 3, 224, 224)

net = MobileNetV1()
with p.profiling("Test MobileNet v1"):
    test_backbone(net, x, 8, 16, 32)

net = MobileNetV2()
with p.profiling("Test MobileNet v2"):
    test_backbone(net, x, 8, 16, 32)
