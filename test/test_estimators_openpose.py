"""BLPose Test: OpenPose

Author: Bo Lin (@linbo0518)
Date: 2020-12-11
"""

import sys
import torch

sys.path.append("../")
from blpose.utils.profiler import Profiler
from blpose.estimators.openpose import *

p = Profiler()

x = torch.randn(8, 3, 368, 368)

net = OpenPoseV1(19, 18)
with p.profiling("Test OpenPose v1"):
    y = net(x)
print(f"Input shape: {x.shape}, Output shape: {y[0].shape}, {y[1].shape}")
torch.onnx.export(net, x, "outputs/openpose_v1.onnx", opset_version=11)

net = OpenPoseV2(19, 18)
with p.profiling("Test OpenPose v2"):
    y = net(x)
print(f"Input shape: {x.shape}, Output shape: {y[0].shape}, {y[1].shape}")
torch.onnx.export(net, x, "outputs/openpose_v2.onnx", opset_version=11)
