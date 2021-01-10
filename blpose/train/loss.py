"""BLPose Train: Loss

Author: Bo Lin (@linbo0518)
Date: 2021-01-01
"""

import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

__all__ = [
    "L1LossWithOHEM",
    "MSELossWithOHEM",
    "SmoothL1LossWithOHEM",
    "KLDivLossWithOHEM",
]


@torch.no_grad()
def _ohem_mask(loss: torch.Tensor, ratio: float):
    values, _ = torch.topk(loss.reshape(-1), int(loss.numel() * ratio))
    mask = loss >= values[-1]
    return mask


class _OHEMLoss(_Loss):
    __constants__ = ["ohem_ratio", "eps", "reduction"]
    ohem_ratio: float
    eps: float

    def __init__(
        self,
        ohem_ratio: float = 0.7,
        eps: float = 1e-7,
        size_average=None,
        reduce=None,
        reduction: str = "none",
    ):
        super(_OHEMLoss, self).__init__(size_average, reduce, reduction)
        self.ohem_ratio = ohem_ratio
        self.eps = eps

    def set_ohem_ratio(self, ratio):
        self.ohem_ratio = ratio


class L1LossWithOHEM(_OHEMLoss):
    __constants__ = ["ohem_ratio", "eps", "reduction"]

    def __init__(self, ohem_ratio: float = 0.7, eps: float = 1e-7):
        super().__init__(ohem_ratio, eps)

    def forward(self, input, target):
        loss = F.l1_loss(input, target, reduction=self.reduction)
        mask = _ohem_mask(loss, self.ohem_ratio)
        loss *= mask
        return loss.sum() / (mask.sum() + self.eps)


class MSELossWithOHEM(_OHEMLoss):
    __constants__ = ["ohem_ratio", "eps", "reduction"]

    def __init__(self, ohem_ratio: float = 0.7, eps: float = 1e-7):
        super().__init__(ohem_ratio, eps)

    def forward(self, input, target):
        loss = F.mse_loss(input, target, reduction=self.reduction)
        mask = _ohem_mask(loss, self.ohem_ratio)
        loss *= mask
        return loss.sum() / (mask.sum() + self.eps)


class SmoothL1LossWithOHEM(_OHEMLoss):
    __constants__ = ["ohem_ratio", "eps", "reduction"]

    def __init__(self, ohem_ratio: float = 0.7, eps: float = 1e-7, beta: float = 1.0):
        super().__init__(ohem_ratio, eps)
        self.beta = beta

    def forward(self, input, target):
        loss = F.smooth_l1_loss(input, target, reduction=self.reduction, beta=self.beta)
        mask = _ohem_mask(loss, self.ohem_ratio)
        loss *= mask
        return loss.sum() / (mask.sum() + self.eps)


class KLDivLossWithOHEM(_OHEMLoss):
    __constants__ = ["ohem_ratio", "eps", "reduction"]

    def __init__(
        self, ohem_ratio: float = 0.7, eps: float = 1e-7, log_target: bool = False
    ):
        super().__init__(ohem_ratio, eps)
        self.log_target = log_target

    def forward(self, input, target):
        loss = F.kl_div(
            input, target, reduction=self.reduction, log_target=self.log_target
        )
        mask = _ohem_mask(loss, self.ohem_ratio)
        loss *= mask
        return loss.sum() / (mask.sum() + self.eps)
