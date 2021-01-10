"""BLPose Utils: Layers

Author: Bo Lin (@linbo0518)
Date: 2020-09-11
"""

from torch import nn

__all__ = [
    "get_conv1x1",
    "get_conv3x3",
    "get_depthwise_separable_conv",
    "get_norm_layer",
]


def get_conv1x1(in_ch, out_ch, stride=1):
    """1x1 convolution

    Args:
        in_ch (int): input channels
        out_ch (int): output channels
        stride (int, optional): stride of kernel. Defaults to 1.

    Returns:
        nn.Conv2d: convolution 2d with 1x1 kernel
    """
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)


def get_conv3x3(in_ch, out_ch, stride=1):
    """3x3 convolution with padding

    Args:
        in_ch (int): input channels
        out_ch (int): output channels
        stride (int, optional): stride of kernel. Defaults to 1.

    Returns:
        nn.Conv2d: convolution 2d with 3x3 kernel and 1 padding
    """
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


def get_depthwise_separable_conv(
    in_ch, out_ch, stride, relu6=True, last_nonlinear=True, has_last_norm=False
):
    """

    Args:
        in_ch (int):
        out_ch (int):
        stride (int):
        relu6 (bool):
        last_nonlinear (bool):
        has_last_norm (bool):

    Returns:
        nn.Sequential: depthwise separable convolution 2d with batch norm and non-linear function
    """
    nonlinear = nn.ReLU6 if relu6 else nn.ReLU

    features = nn.Sequential(
        nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_ch,
            bias=False,
        ),
        nn.BatchNorm2d(in_ch),
        nonlinear(inplace=True),
        nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_ch),
    )
    if has_last_norm:
        features[-1].last_norm = True
    if last_nonlinear:
        features.add_module("5", nonlinear(inplace=True))
    return features


def get_norm_layer(norm_layer, num_channels, num_groups=None):
    if norm_layer == nn.BatchNorm2d:
        return nn.BatchNorm2d(num_channels)
    elif norm_layer == nn.GroupNorm:
        return nn.GroupNorm(num_groups, num_channels)
    elif norm_layer == nn.SyncBatchNorm:
        return nn.SyncBatchNorm(num_channels)
    else:
        NotImplementedError(
            f"'norm_layer' should be BatchNorm2d, GroupNorm or SyncBatchNorm, {norm_layer} is not supported."
        )
