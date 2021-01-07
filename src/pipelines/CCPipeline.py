import torch
from src.pipelines import Pipeline


class CCPipeline(Pipeline):
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss = self._run_batch(batch, batch_idx)
        metrics = {'loss': loss}
        self.log_dict(metrics, on_epoch=True, on_step=False)
        return metrics

    def validation_step(self, batch, batch_idx):
        loss = self._run_batch(batch, batch_idx)
        metrics = {'val_loss': loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        mae_loss, mse_loss = self._run_batch(batch, batch_idx, test=True)
        metrics = {
            'test_mae_loss': mae_loss,
            'test_mse_loss': mse_loss,
        }
        return metrics

    def training_epoch_end(self, outputs) -> None:
        self._run_batch_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs) -> None:
        self._run_batch_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs) -> None:
        self._run_batch_epoch_end(outputs, 'test')

    def _run_batch(self, batch, batch_idx, test=False):
        x, y = batch
        preds = self.forward(x.float())
        preds = torch.reshape(preds, (-1,))
        loss = self.criterions.mae(preds, y.float())  # mae loss
        if test:
            mse_loss = self.criterions.mse(preds, y.float())
            return loss, mse_loss
        return loss

    def _run_batch_epoch_end(self, outputs, mode='train'):
        if mode == 'test':
            average_loss = torch.mean(torch.stack([x["test_mae_loss"] for x in outputs]))
            average_mse_loss = torch.mean(torch.stack([x["test_mse_loss"] for x in outputs]))
            metrics = {
                'test_loss': average_loss,
                'test_mse_loss_average': average_mse_loss,
                'step': self.current_epoch
            }
            self.logger.experiment.add_scalar('test_loss/mae', average_loss, self.current_epoch)
            self.logger.experiment.add_scalar('test_loss/mse', average_mse_loss, self.current_epoch)
        elif mode == 'train':
            average_loss = torch.mean(torch.stack([x['loss'] for x in outputs]))
            metrics = {f'{mode}_loss': average_loss,
                       'step': self.trainer.current_epoch}
            self.logger.experiment.add_scalar('average_loss/train', average_loss, self.current_epoch)
        elif mode == 'val':
            average_loss = torch.mean(torch.stack([x[f'val_loss'] for x in outputs]))
            metrics = {f'val_loss': average_loss,
                       'step': self.trainer.current_epoch}
            self.logger.experiment.add_scalar('average_loss/val', average_loss, self.current_epoch)
