import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning.core.lightning import LightningModule


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
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


class ExtendedFCNCastellano(FCNCastellano, LightningModule):
    def __init__(self):
        super().__init__()

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x.float())
        preds = torch.reshape(preds, (-1,))
        loss = F.l1_loss(preds, y.float())  # mae loss
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x.float())
        preds = torch.reshape(preds, (-1,))
        loss = F.l1_loss(preds, y.float())  # mae loss
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), momentum=0.9, lr=10e-5)
        return optimizer
