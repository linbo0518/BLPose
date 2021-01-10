"""BLPose Backbone: VGG

Author: Bo Lin (@linbo0518)
Date: 2020-09-11

Very Deep Convolutional Networks for Large-Scale Image Recognition
https://arxiv.org/abs/1409.1556
"""
from abc import ABC
from torch import nn
from .base import BackboneBase
from ..utils.validators import check_type, check_gt, check_len, check_oneof

__all__ = ["VGG11", "VGG13", "VGG16", "VGG19"]


def _get_stage_list(in_ch, out_ch, n_repeat, has_pool=True):
    check_type(n_repeat, int)
    check_gt(n_repeat, 0)

    layers = []
    if has_pool:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))

    layers.extend(
        (
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
    )
    for _ in range(1, n_repeat):
        layers.extend(
            (
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            )
        )
    return layers


class VGG(BackboneBase, ABC):
    channels = (64, 128, 256, 512, 512)
    strides = (1, 2, 4, 8, 16)

    def __init__(self, n_repeats):
        check_type(n_repeats, (tuple, list))
        check_len(n_repeats, 5)
        super().__init__()

        self.features = nn.Sequential(
            # stage 0
            *_get_stage_list(3, self.channels[0], n_repeats[0], False),
            # stage 1
            *_get_stage_list(self.channels[0], self.channels[1], n_repeats[1]),
            # stage 2
            *_get_stage_list(self.channels[1], self.channels[2], n_repeats[2]),
            # stage 3
            *_get_stage_list(self.channels[2], self.channels[3], n_repeats[3]),
            # stage 4
            *_get_stage_list(self.channels[3], self.channels[4], n_repeats[4]),
        )

        self.init_params(method="xavier", distribution="uniform")

    def forward(self, x):
        x = self.features(x)
        return x


class VGG11(VGG):
    def __init__(self):
        super().__init__((1, 1, 2, 2, 2))

    def change_stride(self, stride):
        check_type(stride, int)
        check_oneof(stride, (8, 16))

        if stride == 8:
            self.strides = (1, 2, 4, 8, 8)
            self.features[10] = nn.Identity()
        elif stride == 16:
            self.strides = (1, 2, 4, 8, 16)
            self.features[10] = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)


class VGG13(VGG):
    def __init__(self):
        super().__init__((2, 2, 2, 2, 2))

    def change_stride(self, stride):
        check_type(stride, int)
        check_oneof(stride, (8, 16))

        if stride == 8:
            self.strides = (1, 2, 4, 8, 8)
            self.features[19] = nn.Identity()
        elif stride == 16:
            self.strides = (1, 2, 4, 8, 16)
            self.features[19] = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)


class VGG16(VGG):
    def __init__(self):
        super().__init__((2, 2, 3, 3, 3))

    def change_stride(self, stride):
        check_type(stride, int)
        check_oneof(stride, (8, 16))

        if stride == 8:
            self.strides = (1, 2, 4, 8, 8)
            self.features[23] = nn.Identity()
        elif stride == 16:
            self.strides = (1, 2, 4, 8, 16)
            self.features[23] = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)


class VGG19(VGG):
    def __init__(self):
        super().__init__((2, 2, 4, 4, 4))

    def change_stride(self, stride):
        check_type(stride, int)
        check_oneof(stride, (8, 16))

        if stride == 8:
            self.strides = (1, 2, 4, 8, 8)
            self.features[27] = nn.Identity()
        elif stride == 16:
            self.strides = (1, 2, 4, 8, 16)
            self.features[27] = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
