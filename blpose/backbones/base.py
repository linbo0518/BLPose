"""BLPose Backbone: Base

Author: Bo Lin (@linbo0518)
Date: 2020-01-06
"""

from abc import ABCMeta, abstractmethod, abstractproperty
from torch import nn
from ..utils.validators import check_oneof


class BackboneBase(nn.Module, metaclass=ABCMeta):
    def init_params(self, method="kaiming", distribution="normal", zero_gamma=True):
        """TODO: docs

        Args:
            method (str, optional): [description]. Defaults to "kaiming".
            distribution (str, optional): [description]. Defaults to "normal".
            zero_gamma (bool, optional): [description]. Defaults to True.
        """
        method = method.lower()
        distribution = distribution.lower()
        check_oneof(method, ("xavier", "kaiming"))
        check_oneof(distribution, ("uniform", "normal"))

        init_func = f"{method}_{distribution}_"

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                getattr(nn.init, init_func)(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            if isinstance(m, nn.BatchNorm2d):
                is_zero_gamma = zero_gamma and hasattr(m, "last_norm")
                nn.init.constant_(m.weight, 0.0 if is_zero_gamma else 1.0)
                nn.init.constant_(m.bias, 0.0)

    @abstractmethod
    def forward(self, x):
        """Forward function of module

        Args:
            x (torch.Tensor | tuple[torch.Tensor]): x could be a torch.Tensor or a tuple
                of torch.Tensor for forward computation
        """

    @abstractmethod
    def change_stride(self, stride):
        """

        Args:
            stride (int):

        Returns:

        """
