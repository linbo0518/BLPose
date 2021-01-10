"""BLPose Backbone: Hourglass

Author: Bo Lin (@linbo0518)
Date: 2020-12-23

Stacked Hourglass Networks for Human Pose Estimation
https://arxiv.org/abs/1603.06937
"""

from torch import nn
from torch.nn import functional as F
from ..utils.validators import check_aliquot

__all__ = ["StackedHourglass"]


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        check_aliquot(out_ch, 2)
        mid_ch = out_ch // 2
        super().__init__()

        self._is_identity = in_ch == out_ch

        self.features = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=1),
        )

        if not self._is_identity:
            self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        residual = x

        x = self.features(x)

        if not self._is_identity:
            residual = self.residual(residual)

        x += residual
        return x


class Hourglass(nn.Module):
    def __init__(self, depth, in_out_ch):
        super().__init__()

        self.up1 = ResidualBlock(in_out_ch, in_out_ch)
        self.low1 = ResidualBlock(in_out_ch, in_out_ch)

        if depth > 1:
            self.low2 = Hourglass(depth - 1, in_out_ch)
        else:
            self.low2 = ResidualBlock(in_out_ch, in_out_ch)

        self.low3 = ResidualBlock(in_out_ch, in_out_ch)

    def forward(self, x):
        residual = self.up1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.low1(x)
        x = self.low2(x)
        x = self.low3(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x += residual
        return x


class StackedHourglass(nn.Module):
    def __init__(self, n_keypoints, hg_ch=256, n_stack=8):
        super().__init__()

        out_ch = n_keypoints + 1
        self.n_stack = n_stack
        self.intro = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, hg_ch),
        )
        self.hourglasses = nn.ModuleList([Hourglass(4, hg_ch) for _ in range(n_stack)])
        self.features = nn.ModuleList(
            [
                nn.Sequential(
                    ResidualBlock(hg_ch, hg_ch),
                    nn.Conv2d(hg_ch, hg_ch, kernel_size=1),
                    nn.BatchNorm2d(hg_ch),
                    nn.ReLU(inplace=True),
                )
                for _ in range(n_stack)
            ]
        )
        self.outs = nn.ModuleList(
            [nn.Conv2d(hg_ch, out_ch, kernel_size=1) for _ in range(n_stack)]
        )
        self.feature_trans = nn.ModuleList(
            [nn.Conv2d(hg_ch, hg_ch, kernel_size=1) for _ in range(1, n_stack)]
        )
        self.out_trans = nn.ModuleList(
            [nn.Conv2d(out_ch, hg_ch, kernel_size=1) for _ in range(1, n_stack)]
        )

    def forward_train(self, x):
        heatmaps = []
        x = self.intro(x)
        for i in range(self.n_stack):
            hg_out = self.hourglasses[i](x)
            feature = self.features[i](hg_out)
            heatmap = self.outs[i](feature)
            heatmaps.append(heatmap)
            if i < self.n_stack - 1:
                x += self.feature_trans[i](feature) + self.out_trans[i](heatmap)
        return heatmaps

    def forward(self, x):
        x = self.intro(x)
        for i in range(self.n_stack):
            hg_out = self.hourglasses[i](x)
            feature = self.features[i](hg_out)
            heatmap = self.outs[i](feature)
            if i < self.n_stack - 1:
                x += self.feature_trans[i](feature) + self.out_trans[i](heatmap)
        return heatmap
