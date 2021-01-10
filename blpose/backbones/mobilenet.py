"""BLPose Backbone: MobileNet

Author: Bo Lin (@linbo0518)
Date: 2020-12-22

MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
https://arxiv.org/abs/1704.04861

MobileNetV2: Inverted Residuals and Linear Bottlenecks
https://arxiv.org/abs/1801.04381
"""

from torch import nn
from .base import BackboneBase
from ..utils.layers import get_conv1x1, get_conv3x3, get_depthwise_separable_conv
from ..utils.validators import check_type, check_gt, check_oneof

__all__ = ["MobileNetV1", "MobileNetV2"]


def _get_stage_list_v1(in_ch, out_ch, stride, n_repeat):
    check_type(n_repeat, int)
    check_gt(n_repeat, 0)
    layers = [get_depthwise_separable_conv(in_ch, out_ch, stride=stride, relu6=False)]
    for _ in range(1, n_repeat):
        layers.append(
            get_depthwise_separable_conv(out_ch, out_ch, stride=1, relu6=False)
        )
    return layers


class MobileNetV1(BackboneBase):
    channels = (64, 128, 256, 512, 1024)
    strides = (2, 4, 8, 16, 32)

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # stage 0
            get_conv3x3(3, 32, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            *_get_stage_list_v1(32, self.channels[0], stride=1, n_repeat=1),
            # stage 1
            *_get_stage_list_v1(
                self.channels[0], self.channels[1], stride=2, n_repeat=2
            ),
            # stage 2
            *_get_stage_list_v1(
                self.channels[1], self.channels[2], stride=2, n_repeat=2
            ),
            # stage 3
            *_get_stage_list_v1(
                self.channels[2], self.channels[3], stride=2, n_repeat=6
            ),
            # stage 4
            *_get_stage_list_v1(
                self.channels[3], self.channels[4], stride=2, n_repeat=2
            ),
        )

        self.init_params(method="xavier", distribution="uniform")

    def forward(self, x):
        x = self.features(x)
        return x

    def change_stride(self, stride):
        check_type(stride, int)
        check_oneof(stride, (8, 16, 32))

        if stride == 8:
            self.strides = (2, 4, 6, 8, 8, 8)
            self.features[8][0].stride = (1, 1)
            self.features[14][0].stride = (1, 1)
        elif stride == 16:
            self.strides = (2, 4, 6, 8, 16, 16)
            self.features[8][0].stride = (2, 2)
            self.features[14][0].stride = (1, 1)
        elif stride == 32:
            self.strides = (2, 4, 6, 8, 16, 32)
            self.features[8][0].stride = (2, 2)
            self.features[14][0].stride = (2, 2)


class LinearBottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expansion=6):
        check_type(expansion, int)
        check_gt(expansion, 0)
        mid_ch = in_ch * expansion
        super().__init__()

        self._is_identity = in_ch == out_ch and stride == 1

        layers = []
        if expansion != 1:
            layers.extend(
                [
                    get_conv1x1(in_ch, mid_ch),
                    nn.BatchNorm2d(mid_ch),
                    nn.ReLU6(inplace=True),
                ]
            )
        layers.append(
            get_depthwise_separable_conv(
                mid_ch, out_ch=out_ch, stride=stride, last_nonlinear=False
            )
        )
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        residual = x

        x = self.features(x)

        if self._is_identity:
            x += residual
        return x


def _get_stage_list_v2(in_ch, out_ch, stride, n_repeat):
    check_type(n_repeat, int)
    check_gt(n_repeat, 0)
    layers = [LinearBottleneck(in_ch, out_ch, stride=stride)]
    for _ in range(1, n_repeat):
        layers.append(LinearBottleneck(out_ch, out_ch, stride=1))
    return layers


class MobileNetV2(BackboneBase):
    channels = (16, 24, 32, 96, 1280)
    strides = (2, 4, 6, 8, 16, 32)

    def __init__(self):
        super().__init__()

        channels = (32, 16, 24, 32, 64, 96, 160, 320, 1280)

        self.features = nn.Sequential(
            # stage 0
            get_conv3x3(3, channels[0], stride=2),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU6(inplace=True),
            LinearBottleneck(channels[0], channels[1], stride=1, expansion=1),
            # stage 1
            *_get_stage_list_v2(channels[1], channels[2], stride=2, n_repeat=2),
            # stage 2
            *_get_stage_list_v2(channels[2], channels[3], stride=2, n_repeat=3),
            # stage 3
            *_get_stage_list_v2(channels[3], channels[4], stride=2, n_repeat=4),
            *_get_stage_list_v2(channels[4], channels[5], stride=1, n_repeat=3),
            # stage 4
            *_get_stage_list_v2(channels[5], channels[6], stride=2, n_repeat=3),
            *_get_stage_list_v2(channels[6], channels[7], stride=1, n_repeat=1),
            get_conv1x1(channels[7], channels[8], stride=1),
            nn.BatchNorm2d(channels[8]),
            nn.ReLU6(inplace=True),
        )

        self.init_params(method="kaiming", distribution="normal")

    def forward(self, x):
        x = self.features(x)
        return x

    def change_stride(self, stride):
        check_type(stride, int)
        check_oneof(stride, (8, 16, 32))

        if stride == 8:
            self.strides = (2, 4, 6, 8, 8, 8)
            self.features[9].features[3][0].stride = (1, 1)
            self.features[16].features[3][0].stride = (1, 1)
        elif stride == 16:
            self.strides = (2, 4, 6, 8, 16, 16)
            self.features[9].features[3][0].stride = (2, 2)
            self.features[16].features[3][0].stride = (1, 1)
        elif stride == 32:
            self.strides = (2, 4, 6, 8, 16, 32)
            self.features[9].features[3][0].stride = (2, 2)
            self.features[16].features[3][0].stride = (2, 2)
