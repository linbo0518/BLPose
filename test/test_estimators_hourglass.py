"""BLPose Test: Test Stacked Hourglass

Author: Bo Lin (@linbo0518)
Date: 2020-12-23
"""

import sys
import torch

sys.path.append("../")
from blpose.utils.profiler import Profiler
from blpose.estimators.hourglass import Hourglass, StackedHourglass

p = Profiler()

x = torch.randn(1, 256, 256, 256)
net = Hourglass(4, 256)
with p.profiling("Test Hourglass"):
    y = net(x)
print(f"Input shape: {x.shape}, Output shape: {y.shape}")
torch.onnx.export(net, x, "outputs/hourglass.onnx", opset_version=11)

x = torch.randn(1, 3, 256, 256)
net = StackedHourglass(n_keypoints=15)
with p.profiling("Test Stacked Hourglass"):
    y = net(x)
print(f"Input shape: {x.shape}, Output shape: {y.shape}")
torch.onnx.export(net, x, "outputs/stacked_hourglass.onnx", opset_version=11)
