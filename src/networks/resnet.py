from .resnet_helper import *


class ResNet(nn.Module):
    """
    ResNet encoder composed by layers with increasing features.
    """
    def __init__(self, initial_size=128, in_channels=3, blocks_sizes=tuple([32, 64, 128, 256]), deepths=tuple([1, 1, 1, 1]),
                 activation='relu', block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation_func(activation),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation,
                        block=block, *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion,
                          out_channels, n=n, activation=activation,
                          block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]
        ])

        self.head = nn.Sequential(
            nn.Linear(in_features=self.blocks_sizes[-1] * (initial_size // 32)**2, out_features=128),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

