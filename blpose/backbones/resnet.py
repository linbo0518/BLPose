"""BLPose Backbone: ResNet

Author: Bo Lin (@linbo0518)
Date: 2020-09-13

Deep Residual Learning for Image Recognition
https://arxiv.org/abs/1512.03385
"""
from abc import ABC
from torch import nn
from torch.nn import functional as F
from .base import BackboneBase
from ..utils.layers import get_conv1x1, get_conv3x3
from ..utils.validators import (
    check_type,
    check_gt,
    check_len,
    check_aliquot,
    check_oneof,
)

__all__ = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]


def _get_stage_list(block, in_ch, out_ch, stride, n_repeat):
    check_type(n_repeat, int)
    check_gt(n_repeat, 0)
    layers = [block(in_ch, out_ch, stride)]
    for _ in range(1, n_repeat):
        layers.append(block(out_ch, out_ch, 1))
    return layers


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()

        self._is_identity = in_ch == out_ch and stride == 1

        self.features = nn.Sequential(
            get_conv3x3(in_ch, out_ch, stride=stride),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            get_conv3x3(out_ch, out_ch),
            nn.BatchNorm2d(out_ch),
        )
        self.features[4].last_norm = True

        if not self._is_identity:
            self.residual = nn.Sequential(
                get_conv3x3(in_ch, out_ch, stride=stride),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        residual = x

        x = self.features(x)

        if not self._is_identity:
            residual = self.residual(residual)

        x += residual
        x = F.relu(x, inplace=True)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        check_aliquot(out_ch, 4)
        mid_ch = out_ch // 4
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

        if not self._is_identity:
            self.residual = nn.Sequential(
                get_conv1x1(in_ch, out_ch, stride=stride),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        residual = x

        x = self.features(x)

        if not self._is_identity:
            residual = self.residual(residual)

        x += residual
        x = F.relu(x, inplace=True)
        return x


class ResNet(BackboneBase, ABC):
    channels = (64, 64, 128, 256, 512)
    strides = (2, 4, 8, 16, 32)

    def __init__(self, blocks, n_repeats):
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
                blocks,
                self.channels[0],
                self.channels[1],
                stride=1,
                n_repeat=n_repeats[0],
            ),
            # stage 2
            *_get_stage_list(
                blocks,
                self.channels[1],
                self.channels[2],
                stride=2,
                n_repeat=n_repeats[1],
            ),
            # stage 3
            *_get_stage_list(
                blocks,
                self.channels[2],
                self.channels[3],
                stride=2,
                n_repeat=n_repeats[2],
            ),
            # stage 4
            *_get_stage_list(
                blocks,
                self.channels[3],
                self.channels[4],
                stride=2,
                n_repeat=n_repeats[3],
            )
        )

        self.init_params(method="kaiming", distribution="normal")

    def forward(self, x):
        x = self.features(x)
        return x


class ResNet18(ResNet):
    channels = (64, 64, 128, 256, 512)

    def __init__(self):
        super().__init__(BasicBlock, (2, 2, 2, 2))

    def change_stride(self, stride):
        check_type(stride, int)
        check_oneof(stride, (8, 16, 32))

        if stride == 8:
            self.strides = (2, 4, 6, 8, 8, 8)
            self.features[8].features[0].stride = (1, 1)
            self.features[8].residual[0].stride = (1, 1)
            self.features[10].features[0].stride = (1, 1)
            self.features[10].residual[0].stride = (1, 1)
        elif stride == 16:
            self.strides = (2, 4, 6, 8, 16, 16)
            self.features[8].features[0].stride = (2, 2)
            self.features[8].residual[0].stride = (2, 2)
            self.features[10].features[0].stride = (1, 1)
            self.features[10].residual[0].stride = (1, 1)
        elif stride == 32:
            self.strides = (2, 4, 6, 8, 16, 32)
            self.features[8].features[0].stride = (2, 2)
            self.features[8].residual[0].stride = (2, 2)
            self.features[10].features[0].stride = (2, 2)
            self.features[10].residual[0].stride = (2, 2)


class ResNet34(ResNet):
    channels = (64, 64, 128, 256, 512)

    def __init__(self):
        super().__init__(BasicBlock, (3, 4, 6, 3))

    def change_stride(self, stride):
        check_type(stride, int)
        check_oneof(stride, (8, 16, 32))

        if stride == 8:
            self.strides = (2, 4, 6, 8, 8, 8)
            self.features[11].features[0].stride = (1, 1)
            self.features[11].residual[0].stride = (1, 1)
            self.features[17].features[0].stride = (1, 1)
            self.features[17].residual[0].stride = (1, 1)
        elif stride == 16:
            self.strides = (2, 4, 6, 8, 16, 16)
            self.features[11].features[0].stride = (2, 2)
            self.features[11].residual[0].stride = (2, 2)
            self.features[17].features[0].stride = (1, 1)
            self.features[17].residual[0].stride = (1, 1)
        elif stride == 32:
            self.strides = (2, 4, 6, 8, 16, 32)
            self.features[11].features[0].stride = (2, 2)
            self.features[11].residual[0].stride = (2, 2)
            self.features[17].features[0].stride = (2, 2)
            self.features[17].residual[0].stride = (2, 2)


class ResNet50(ResNet):
    channels = (64, 256, 512, 1024, 2048)

    def __init__(self):
        super().__init__(ResidualBlock, (3, 4, 6, 3))

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


class ResNet101(ResNet):
    channels = (64, 256, 512, 1024, 2048)

    def __init__(self):
        super().__init__(ResidualBlock, (3, 4, 23, 3))

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


class ResNet152(ResNet):
    channels = (64, 256, 512, 1024, 2048)

    def __init__(self):
        super().__init__(ResidualBlock, (3, 8, 36, 3))

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
