import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNCastellano(nn.Module):
    def __init__(self, in_channels=3):
        super(FCNCastellano, self).__init__()

        # TODO parametrize feature numbers
        features = [
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(5, 5), stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.BatchNorm2d(num_features=32, eps=0.001, momentum=0.99),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.PReLU(),

            nn.Dropout2d(0.5),
            nn.AdaptiveAvgPool2d((1, 1))
        ]

        head = [
            nn.Linear(in_features=64, out_features=1),
        ]

        self.features = nn.Sequential(*features)
        self.head = nn.Sequential(*head)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1)  # TODO make a correct flatten
        x = self.head(x)

        return x
