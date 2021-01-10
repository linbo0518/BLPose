"""BLPose Utils: Helper

Author: Bo Lin (@linbo0518)
Date: 2020-09-11
"""

import torch
from torch import nn
from .validators import check_oneof, check_type

__all__ = ["init_params", "get_module_output_shape"]


def init_params(module, method="kaiming", distribution="normal", zero_gamma=True):
    """TODO: docs

    Args:
        module ([type]): [description]
        method (str, optional): [description]. Defaults to "kaiming".
        distribution (str, optional): [description]. Defaults to "normal".
        zero_gamma (bool, optional): [description]. Defaults to True.
    """
    method = method.lower()
    distribution = distribution.lower()
    check_oneof(method, ("xavier", "kaiming"))
    check_oneof(distribution, ("uniform", "normal"))

    init_func = f"{method}_{distribution}_"

    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            getattr(nn.init, init_func)(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        if isinstance(m, nn.BatchNorm2d):
            is_zero_gamma = zero_gamma and hasattr(m, "last_norm")
            nn.init.constant_(m.weight, 0.0 if is_zero_gamma else 1.0)
            nn.init.constant_(m.bias, 0.0)


@torch.no_grad()
def get_module_output_shape(module, input_shape=(1, 3, 64, 64), infer_func="forward"):
    check_type(input_shape, (tuple, list))
    dummy_input = torch.randn(*input_shape)
    if infer_func == "forward":
        out = module(dummy_input)
    else:
        out = getattr(module, infer_func)(dummy_input)
    return out.shape
