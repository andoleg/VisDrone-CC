from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning.core.lightning import LightningModule


class FCNCastellanoBN(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()

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


class ExtendedFCNCastellanoBN(FCNCastellanoBN, LightningModule):
    def __init__(self):
        super().__init__()
        self.train_losses = list()
        self.val_losses = list()

    def training_step(self, batch, batch_idx):
        loss = self.run_batch(batch, batch_idx)
        self.train_losses.append(loss.item())
        self.log('train_loss', loss)
        return {'loss': loss}

    def training_epoch_end(self, outputs) -> None:
        average_loss = torch.mean(torch.stack([x["loss"] for x in outputs]))
        self.log('train_epoch_loss', average_loss)  # log mean losses on epoch end
        self.train_losses = list()

    def validation_step(self, batch, batch_idx):
        loss = self.run_batch(batch, batch_idx)
        metrics = {'val_loss': loss}
        self.val_losses.append(loss.item())
        self.log_dict(metrics)
        return metrics

    def validation_epoch_end(self, outputs) -> None:
        average_loss = torch.mean(torch.stack([x["val_loss"] for x in outputs]))
        self.log('val_epoch_loss', average_loss)  # log mean losses on epoch end
        self.val_losses = list()

    def test_step(self, batch, batch_idx):
        loss = self.run_batch(batch, batch_idx)
        metrics = {'test_loss': loss}
        self.log_dict(metrics, on_step=True, on_epoch=True)
        return metrics

    def test_epoch_end(self, outputs) -> None:
        average_loss = torch.mean(torch.stack([x["test_loss"] for x in outputs]))
        self.log('test_loss_average', average_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), momentum=0.9, lr=10e-5)
        return optimizer

    def run_batch(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x.float())
        preds = torch.reshape(preds, (-1,))
        loss = F.l1_loss(preds, y.float())  # mae loss
        return loss
