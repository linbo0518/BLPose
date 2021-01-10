"""BLPose Test: Utils Helper

Author: Bo Lin (@linbo0518)
Date: 2020-12-08
"""

import sys
from torch import nn

sys.path.append("../")
from blpose.utils.helper import *

just = 15
conv = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
bn = nn.BatchNorm2d(64)
last_bn = nn.BatchNorm2d(64)
last_bn.last_norm = True

print(
    f"{'before'.rjust(just)}: mean={conv.weight.data.mean()} var={conv.weight.data.var()}"
)

init_params(conv, "kaiming", "normal")
print(
    f"{'kaiming normal'.rjust(just)}: mean={conv.weight.data.mean()} var={conv.weight.data.var()}"
)

init_params(conv, "kaiming", "uniform")
print(
    f"{'kaiming uniform'.rjust(just)}: mean={conv.weight.data.mean()} var={conv.weight.data.var()}"
)

init_params(conv, "xavier", "normal")
print(
    f"{'xavier normal'.rjust(just)}: mean={conv.weight.data.mean()} var={conv.weight.data.var()}"
)

init_params(conv, "xavier", "uniform")
print(
    f"{'xavier uniform'.rjust(just)}: mean={conv.weight.data.mean()} var={conv.weight.data.var()}"
)

init_params(bn)
print(
    f"{'normal bn'.rjust(just)}: mean={bn.weight.data.mean()} var={bn.weight.data.var()}"
)

init_params(last_bn)
print(
    f"{'last bn'.rjust(just)}: mean={last_bn.weight.data.mean()} var={last_bn.weight.data.var()}"
)
