import torch
from src.pipelines import Pipeline
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report
import numpy as np


class CCPipeline(Pipeline):
    def forward(self, x):
        return self.model(x)

    def on_test_epoch_start(self) -> None:
        self.predictions = list()
        self.ground_truth = list()

    def training_step(self, batch, batch_idx):
        loss = self._run_batch(batch, batch_idx)
        metrics = {'loss': loss}
        self.log_dict(metrics)
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
        self.log_dict(metrics)
        return metrics

    def training_epoch_end(self, outputs) -> None:
        self._run_batch_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs) -> None:
        self._run_batch_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs) -> None:
        self._run_batch_epoch_end(outputs, 'test')
        self._precision_recall()

    def _precision_recall(self, class_divider=10):
        gt_classes = list()
        pred_classes = list()
        for i in range(len(self.ground_truth)):
            gt_classes.append(0 if self.ground_truth[i] < class_divider else 1)
            pred_classes.append(0 if self.predictions[i] < class_divider else 1)

        gt_classes = np.array(gt_classes)
        pred_classes = np.array(pred_classes)

        print(f'Classification report (0: class < {class_divider}; 1: class >= {class_divider}):')
        print(classification_report(gt_classes, pred_classes))
        # print(f'Precision/Recall: {precision_score(gt_classes, pred_classes)}, {recall_score(gt_classes, pred_classes)}')
        # print(f'Precision/Recall: {precision_score(1 - gt_classes, 1 - pred_classes)}, '
        #       f'{recall_score(1 - gt_classes, 1 - pred_classes)}')

        # self.logger.experiment.add_hparams({'class1_precision': precision_score(1 - gt_classes, 1 - pred_classes),
        #                                     'class1_recall': recall_score(1 - gt_classes, 1 - pred_classes),
        #                                     'class2_precision': precision_score(gt_classes, pred_classes),
        #                                     'class2_recall': recall_score(gt_classes, pred_classes)}, {})

    def _run_batch(self, batch, batch_idx, test=False):
        x, y = batch
        y = y.float()
        preds = self.forward(x.float())
        preds = torch.reshape(preds, (-1,))
        loss = self.criterions.mae(preds, y)  # mae loss
        if test:
            self.predictions.extend(preds)
            self.ground_truth.extend(y)
            mse_loss = self.criterions.mse(preds, y)
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
            print('\nResults:')
            print(f'\tMAE loss: {average_loss}, \n\t MSE loss: {average_mse_loss}')
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
