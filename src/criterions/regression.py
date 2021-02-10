import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class MAEWeighted(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
        super().__init__(size_average, reduce, reduction)

    def forward(self, inputs, targets, weights):
        mae = torch.abs(inputs - targets) * weights
        if self.reduction == "mean":
            return mae.mean()
        else:
            return mae.sum()
