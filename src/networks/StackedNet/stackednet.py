from .helper import CNN, Bottleneck
from torch import nn
import torch


class StackedNN(nn.Module):
    def __init__(self, batch_size=64):
        super(StackedNN, self).__init__()

        self.models = [
            CNN(out_channels=32, kernel_size=3),
            CNN(out_channels=16, kernel_size=5),
            Bottleneck(kernel_size=3),
        ]
        head_meta = [
            nn.Linear(batch_size * len(self.models), 1),
        ]
        self.head_meta = nn.Sequential(*head_meta)

    def forward(self, x):
        results = [model(x) for model in self.models]
        result_layer = torch.cat(results, 0).squeeze()
        out = self.head_meta(result_layer)
        return out
