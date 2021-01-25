from .helper import CNN, Bottleneck
from torch import nn
import torch


class StackedNN(nn.Module):
    def __init__(self, batch_size=64):
        super(StackedNN, self).__init__()

        # todo REDO
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.models = [
            CNN(out_channels=32, kernel_size=3).to(device),
            CNN(out_channels=16, kernel_size=5).to(device),
            Bottleneck(kernel_size=3).to(device),
        ]
        head_meta = [
            nn.Linear(batch_size * len(self.models), batch_size),
        ]
        self.head_meta = nn.Sequential(*head_meta)

    def forward(self, x):
        results = [model(x) for model in self.models]
        result_layer = torch.cat(results, 0).squeeze()
        out = self.head_meta(result_layer)
        return out
