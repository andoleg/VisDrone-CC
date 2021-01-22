import torch.nn as nn


class FCNCastellanoBN(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(FCNCastellanoBN, self).__init__()

        features = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2, kernel_size=(5, 5), stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.BatchNorm2d(num_features=out_channels//2, eps=0.001, momentum=0.99),
            nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels * 2, kernel_size=(3, 3), stride=1),
            nn.PReLU(),

            # Bottleneck
            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels // 4, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(num_features=out_channels // 4),
            nn.PReLU(),
            nn.Conv2d(in_channels=out_channels // 4, out_channels=out_channels // 4, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(num_features=out_channels // 4),
            nn.PReLU(),
            nn.Conv2d(in_channels=out_channels // 4, out_channels=out_channels * 2, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(num_features=out_channels * 2),
            nn.PReLU(),

            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=(3, 3), stride=1),
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


class FCNCastellanoBNDeeper(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()

        features = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2, kernel_size=(5, 5), stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.BatchNorm2d(num_features=out_channels//2, eps=0.001, momentum=0.99),
            nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels, kernel_size=(3, 3), stride=1),
            nn.PReLU(),

            # Bottleneck
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels // 4, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(num_features=out_channels // 4),
            nn.PReLU(),
            nn.Conv2d(in_channels=out_channels // 4, out_channels=out_channels // 4, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(num_features=out_channels // 4),
            nn.PReLU(),
            nn.Conv2d(in_channels=out_channels // 4, out_channels=out_channels, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.PReLU(),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=1),
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

class FCNCastellanoBNSmall(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()

        features = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2, kernel_size=(5, 5), stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.BatchNorm2d(num_features=out_channels//2, eps=0.001, momentum=0.99),
            nn.Conv2d(in_channels=out_channels // 2, out_channels=out_channels // 2, kernel_size=(3, 3), stride=1),
            nn.PReLU(),

            # Bottleneck
            nn.Conv2d(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=(1, 1), stride=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=(3, 3), stride=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=(1, 1), stride=1),
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
        print(x.shape)
        x = self.head(x)
        return x