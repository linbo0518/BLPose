"""BLPose Estimator: OpenPose

Author: Bo Lin (@linbo0518)
Date: 2020-09-11

Convolutional Pose Machines
https://arxiv.org/abs/1602.00134

Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields
https://arxiv.org/abs/1611.08050

OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields
https://arxiv.org/abs/1812.08008
"""

import math
import torch
from torch import nn

from ..backbones import get_backbone
from ..utils.helper import init_params, get_module_output_shape
from ..utils.validators import check_aliquot

__all__ = ["OpenPoseV1", "OpenPoseV2"]


def _get_intro(in_ch, out_ch):
    in_index = math.log2(in_ch)
    out_index = math.log2(out_ch)
    mid_index = round((in_index + out_index) / 2)
    mid_ch = 2 ** mid_index

    return nn.Sequential(
        nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


# OpenPose v1
class Stage1V1(nn.Module):
    def __init__(self, in_ch, out_ch1, out_ch2):
        super().__init__()
        self.paf = nn.Sequential(
            nn.Conv2d(in_ch, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, out_ch1, kernel_size=1),
        )
        self.heat = nn.Sequential(
            nn.Conv2d(in_ch, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, out_ch2, kernel_size=1),
        )

    def forward(self, x):
        pafmap = self.paf(x)
        heatmap = self.heat(x)
        return pafmap, heatmap


class StageTV1(nn.Module):
    def __init__(self, in_ch, out_ch1, out_ch2):
        super().__init__()
        self.paf = nn.Sequential(
            nn.Conv2d(in_ch, 128, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_ch1, kernel_size=1),
        )
        self.heat = nn.Sequential(
            nn.Conv2d(in_ch, 128, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_ch2, kernel_size=1),
        )

    def forward(self, inputs):
        feature = torch.cat(inputs, dim=1)
        pafmap = self.paf(feature)
        heatmap = self.heat(feature)
        return pafmap, heatmap


class OpenPoseV1(nn.Module):
    def __init__(self, n_limbs, n_keypoints, backbone="vgg19_first10"):
        super().__init__()
        paf_ch = n_limbs * 2
        heat_ch = n_keypoints + 1
        n_stages = 6
        self.backbone = get_backbone(backbone)
        _, b_ch, _, _ = get_module_output_shape(self.backbone)
        self.intro = _get_intro(b_ch, 128)
        self.stage1 = Stage1V1(128, paf_ch, heat_ch)
        self.stages = nn.ModuleList(
            [
                StageTV1(128 + paf_ch + heat_ch, paf_ch, heat_ch)
                for _ in range(1, n_stages)
            ]
        )

        init_params(self)

    def forward_train(self, x):
        pafmaps = []
        heatmaps = []
        x = self.backbone(x)
        x = self.intro(x)
        pafmap, heatmap = self.stage1(x)
        pafmaps.append(pafmap)
        heatmap.append(heatmap)
        for stage in self.stages:
            pafmap, heatmap = stage((pafmap, heatmap, x))
            pafmaps.append(pafmap)
            heatmap.append(heatmap)
        return pafmaps, heatmaps

    def forward(self, x):
        x = self.backbone(x)
        x = self.intro(x)
        pafmap, heatmap = self.stage1(x)
        for stage in self.stages:
            pafmap, heatmap = stage((pafmap, heatmap, x))
        return pafmap, heatmap


# OpenPose v2
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        check_aliquot(out_ch, 3)
        mid_ch = out_ch // 3
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride=1, padding=1)
        self.prelu1 = nn.PReLU(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1)
        self.prelu2 = nn.PReLU(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1)
        self.prelu3 = nn.PReLU(mid_ch)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.prelu1(x1)
        x2 = self.conv2(x1)
        x2 = self.prelu2(x2)
        x3 = self.conv3(x2)
        x3 = self.prelu3(x3)
        x = torch.cat((x1, x2, x3), dim=1)
        return x


class Stage1V2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_ch, 288),
            ConvBlock(288, 288),
            ConvBlock(288, 288),
            ConvBlock(288, 288),
            ConvBlock(288, 288),
            nn.Conv2d(288, 256, kernel_size=1),
            nn.PReLU(256),
            nn.Conv2d(256, out_ch, kernel_size=1),
        )

    def forward(self, inputs):
        x = torch.cat(inputs, dim=1)
        x = self.features(x)
        return x


class StageTV2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_ch, 384),
            ConvBlock(384, 384),
            ConvBlock(384, 384),
            ConvBlock(384, 384),
            ConvBlock(384, 384),
            nn.Conv2d(384, 512, kernel_size=1),
            nn.PReLU(512),
            nn.Conv2d(512, out_ch, kernel_size=1),
        )

    def forward(self, inputs):
        x = torch.cat(inputs, dim=1)
        x = self.features(x)
        return x


class OpenPoseV2(nn.Module):
    def __init__(self, n_limbs, n_keypoints, backbone="vgg19_first10"):
        super().__init__()
        paf_ch = n_limbs * 2
        heat_ch = n_keypoints + 1
        paf_stages = 4
        heat_stages = 2
        self.backbone = get_backbone(backbone)
        _, b_ch, _, _ = get_module_output_shape(self.backbone)
        self.intro = _get_intro(b_ch, 128)
        self.paf_stage1 = Stage1V2(128, paf_ch)
        self.paf_stages = nn.ModuleList(
            StageTV2(128 + paf_ch, paf_ch) for _ in range(1, paf_stages)
        )
        self.heat_stage1 = Stage1V2(128 + paf_ch, heat_ch)
        self.heat_stages = nn.ModuleList(
            StageTV2(128 + paf_ch + heat_ch, heat_ch) for _ in range(1, heat_stages)
        )

        init_params(self)

    def forward_train(self, x):
        pafmaps = []
        heatmaps = []
        x = self.backbone(x)
        x = self.intro(x)
        pafmap = self.paf_stage1((x,))
        pafmaps.append(pafmap)
        for stage in self.paf_stages:
            pafmap = stage((x, pafmap))
            pafmaps.append(pafmap)
        heatmap = self.heat_stage1((x, pafmap))
        heatmaps.append(heatmap)
        for stage in self.heat_stages:
            heatmap = stage((x, heatmap, pafmap))
            heatmaps.append(heatmap)
        return pafmaps, heatmaps

    def forward(self, x):
        x = self.backbone(x)
        x = self.intro(x)
        pafmap = self.paf_stage1((x,))
        for stage in self.paf_stages:
            pafmap = stage((x, pafmap))
        heatmap = self.heat_stage1((x, pafmap))
        for stage in self.heat_stages:
            heatmap = stage((x, heatmap, pafmap))
        return pafmap, heatmap
