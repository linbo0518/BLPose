"""BLPose Backbone: SE ResNet

Author: Bo Lin (@linbo0518)
Date: 2020-12-21

Squeeze-and-Excitation Networks
https://arxiv.org/abs/1709.01507
"""
from abc import ABC
from torch import nn
from torch.nn import functional as F
from .base import BackboneBase
from ..utils.layers import get_conv1x1, get_conv3x3
from ..utils.validators import (
    check_type,
    check_gt,
    check_aliquot,
    check_len,
    check_oneof,
)

__all__ = ["SEResNet50", "SEResNet101", "SEResNet152"]


def _get_stage_list(in_ch, out_ch, stride, n_repeat):
    check_type(n_repeat, int)
    check_gt(n_repeat, 0)
    layers = [SEResBlock(in_ch, out_ch, stride)]
    for _ in range(1, n_repeat):
        layers.append(SEResBlock(out_ch, out_ch, 1))
    return layers


class SEResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        check_aliquot(out_ch, 4)
        mid_ch = int(out_ch / 4)
        r = 16
        super().__init__()

        self._is_identity = in_ch == out_ch and stride == 1

        self.features = nn.Sequential(
            get_conv1x1(in_ch, mid_ch),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            get_conv3x3(mid_ch, mid_ch, stride=stride),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            get_conv1x1(mid_ch, out_ch),
            nn.BatchNorm2d(out_ch),
        )

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            get_conv1x1(out_ch, out_ch // r),
            nn.ReLU(inplace=True),
            get_conv1x1(out_ch // r, out_ch),
            nn.Sigmoid(),
        )

        if not self._is_identity:
            self.residual = nn.Sequential(
                get_conv1x1(in_ch, out_ch, stride=stride),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        residual = x

        x = self.features(x)

        se = self.se(x)
        x *= se

        if not self._is_identity:
            residual = self.residual(residual)

        x += residual
        x = F.relu(x, inplace=True)
        return x


class SEResNet(BackboneBase, ABC):
    channels = (64, 64, 128, 256, 512)
    strides = (2, 4, 8, 16, 32)

    def __init__(self, n_repeats):
        check_type(n_repeats, (tuple, list))
        check_len(n_repeats, 4)
        super().__init__()

        self.features = nn.Sequential(
            # stage 0
            nn.Conv2d(
                3, self.channels[0], kernel_size=7, stride=2, padding=3, bias=False
            ),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(inplace=True),
            # stage 1
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *_get_stage_list(
                self.channels[0], self.channels[1], stride=1, n_repeat=n_repeats[0]
            ),
            # stage 2
            *_get_stage_list(
                self.channels[1], self.channels[2], stride=2, n_repeat=n_repeats[1]
            ),
            # stage 3
            *_get_stage_list(
                self.channels[2], self.channels[3], stride=2, n_repeat=n_repeats[2]
            ),
            # stage 4
            *_get_stage_list(
                self.channels[3], self.channels[4], stride=2, n_repeat=n_repeats[3]
            )
        )

        self.init_params(method="kaiming", distribution="normal")

    def forward(self, x):
        x = self.features(x)
        return x


class SEResNet50(SEResNet):
    channels = (64, 256, 512, 1024, 2048)

    def __init__(self):
        super().__init__((3, 4, 6, 3))

    def change_stride(self, stride):
        check_type(stride, int)
        check_oneof(stride, (8, 16, 32))

        if stride == 8:
            self.strides = (2, 4, 6, 8, 8, 8)
            self.features[11].features[3].stride = (1, 1)
            self.features[11].residual[0].stride = (1, 1)
            self.features[17].features[3].stride = (1, 1)
            self.features[17].residual[0].stride = (1, 1)
        elif stride == 16:
            self.strides = (2, 4, 6, 8, 16, 16)
            self.features[11].features[3].stride = (2, 2)
            self.features[11].residual[0].stride = (2, 2)
            self.features[17].features[3].stride = (1, 1)
            self.features[17].residual[0].stride = (1, 1)
        elif stride == 32:
            self.strides = (2, 4, 6, 8, 16, 32)
            self.features[11].features[3].stride = (2, 2)
            self.features[11].residual[0].stride = (2, 2)
            self.features[17].features[3].stride = (2, 2)
            self.features[17].residual[0].stride = (2, 2)


class SEResNet101(SEResNet):
    channels = (64, 256, 512, 1024, 2048)

    def __init__(self):
        super().__init__((3, 4, 23, 3))

    def change_stride(self, stride):
        check_type(stride, int)
        check_oneof(stride, (8, 16, 32))

        if stride == 8:
            self.strides = (2, 4, 6, 8, 8, 8)
            self.features[11].features[3].stride = (1, 1)
            self.features[11].residual[0].stride = (1, 1)
            self.features[34].features[3].stride = (1, 1)
            self.features[34].residual[0].stride = (1, 1)
        elif stride == 16:
            self.strides = (2, 4, 6, 8, 16, 16)
            self.features[11].features[3].stride = (2, 2)
            self.features[11].residual[0].stride = (2, 2)
            self.features[34].features[3].stride = (1, 1)
            self.features[34].residual[0].stride = (1, 1)
        elif stride == 32:
            self.strides = (2, 4, 6, 8, 16, 32)
            self.features[11].features[3].stride = (2, 2)
            self.features[11].residual[0].stride = (2, 2)
            self.features[34].features[3].stride = (2, 2)
            self.features[34].residual[0].stride = (2, 2)


class SEResNet152(SEResNet):
    channels = (64, 256, 512, 1024, 2048)

    def __init__(self):
        super().__init__((3, 8, 36, 3))

    def change_stride(self, stride):
        check_type(stride, int)
        check_oneof(stride, (8, 16, 32))

        if stride == 8:
            self.strides = (2, 4, 6, 8, 8, 8)
            self.features[15].features[3].stride = (1, 1)
            self.features[15].residual[0].stride = (1, 1)
            self.features[51].features[3].stride = (1, 1)
            self.features[51].residual[0].stride = (1, 1)
        elif stride == 16:
            self.strides = (2, 4, 6, 8, 16, 16)
            self.features[15].features[3].stride = (2, 2)
            self.features[15].residual[0].stride = (2, 2)
            self.features[51].features[3].stride = (1, 1)
            self.features[51].residual[0].stride = (1, 1)
        elif stride == 32:
            self.strides = (2, 4, 6, 8, 16, 32)
            self.features[15].features[3].stride = (2, 2)
            self.features[15].residual[0].stride = (2, 2)
            self.features[51].features[3].stride = (2, 2)
            self.features[51].residual[0].stride = (2, 2)
