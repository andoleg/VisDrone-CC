from torch import nn


class CNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, kernel_size=3):
        super().__init__()
        features = [
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.99),
            nn.Conv2d(out_channels, out_channels * 2, kernel_size),
            nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size),

            nn.Dropout2d(0.5),
            nn.AdaptiveAvgPool2d((1, 1))
        ]
        head = [
            nn.Linear(in_features=out_channels * 2, out_features=1),
        ]

        self.features = nn.Sequential(*features)
        self.head = nn.Sequential(*head)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, kernel_size=3):
        super().__init__()

        features = [
            nn.Conv2d(in_channels, out_channels // 2, kernel_size),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.BatchNorm2d(num_features=out_channels // 2, eps=0.001, momentum=0.99),

            nn.Conv2d(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=(1, 1), stride=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1), stride=1),
            nn.PReLU(),

            nn.Dropout2d(0.5),
            nn.AdaptiveAvgPool2d((1, 1))
        ]

        head = [
            nn.Linear(in_features=out_channels, out_features=1),
        ]

        self.features = nn.Sequential(*features)
        self.head = nn.Sequential(*head)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x