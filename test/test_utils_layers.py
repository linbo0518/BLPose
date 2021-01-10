"""BLPose Test: Utils Layers

Author: Bo Lin (@linbo0518)
Date: 2020-12-08
"""

import sys
from torch import nn

sys.path.append("../")
from blpose.utils.profiler import Profiler
from blpose.utils.layers import *

p = Profiler()

with p.profiling("Test get_conv1x1"):
    print(f"ref: {nn.Conv2d(3, 64, kernel_size=1, stride=1, bias=False)}")
    print(f"act: {get_conv1x1(3, 64, stride=1)}")

    print(f"ref: {nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False)}")
    print(f"act: {get_conv1x1(64, 128, stride=2)}")

with p.profiling("Test get_conv3x3"):
    print(f"ref: {nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)}")
    print(f"act: {get_conv3x3(3, 64, stride=1)}")

    print(f"ref: {nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)}")
    print(f"act: {get_conv3x3(64, 128, stride=2)}")

with p.profiling("Test get_depthwise_separable_conv"):
    print(
        f"act: {get_depthwise_separable_conv(3, 64, stride=1, relu6=False, last_nonlinear=False)}"
    )

    print(f"act: {get_depthwise_separable_conv(64, 128, stride=2, has_last_norm=True)}")

with p.profiling("Test get_norm_layer"):
    print(f"ref: {nn.BatchNorm2d(64)}")
    print(f"act: {get_norm_layer(nn.BatchNorm2d, 64)}")

    print(f"ref: {nn.GroupNorm(4, 64)}")
    print(f"act: {get_norm_layer(nn.GroupNorm, 64, 4)}")
