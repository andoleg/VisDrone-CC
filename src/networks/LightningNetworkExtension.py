import torch
from torch.nn import functional as F
from pytorch_lightning.core.lightning import LightningModule


class PLNetworkExtension(LightningModule):
    def __init__(self):
        super().__init__()
        # todo allow criterion change

    def training_step(self, batch, batch_idx):
        loss = self._run_batch(batch, batch_idx)
        self.log('train_loss', loss)
        return {'loss': loss}

    def training_epoch_end(self, outputs) -> None:
        average_loss = torch.mean(torch.stack([x["loss"] for x in outputs]))
        self.log('train_epoch_loss', average_loss)  # log mean losses on epoch end

    def validation_step(self, batch, batch_idx):
        loss = self._run_batch(batch, batch_idx)
        metrics = {'val_loss': loss}
        self.log_dict(metrics)
        return metrics

    def validation_epoch_end(self, outputs) -> None:
        average_loss = torch.mean(torch.stack([x["val_loss"] for x in outputs]))
        self.log('val_epoch_loss', average_loss)  # log mean losses on epoch end

    def test_step(self, batch, batch_idx):
        loss = self._run_batch(batch, batch_idx)
        metrics = {'test_loss': loss}
        self.log_dict(metrics, on_step=True, on_epoch=True)
        return metrics

    def test_epoch_end(self, outputs) -> None:
        average_loss = torch.mean(torch.stack([x["test_loss"] for x in outputs]))
        self.log('test_loss_average', average_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), momentum=0.9, lr=10e-5)
        return optimizer

    def _run_batch(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x.float())
        preds = torch.reshape(preds, (-1,))
        loss = F.l1_loss(preds, y.float())  # mae loss
        return loss