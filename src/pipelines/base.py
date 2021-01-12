from collections import namedtuple

from pytorch_lightning import LightningModule
from torch import load as load_model
from src.utils import print_dataset_info


class Pipeline(LightningModule):
    def __init__(
            self,
            model_params,
            criterions_params,
            optimizers_params,
            schedulers_params,
            data_params,
            test=False
    ):
        super(Pipeline, self).__init__()
        self.data_params = data_params
        self.optimizers_params = optimizers_params
        self.schedulers_params = schedulers_params

        from src.config import ClassBox

        self.model = ClassBox.models[model_params.name](**model_params.params)
        if test:
            state_dict = load_model(model_params.load_model)['state_dict']
            state_dict = {k[6:]: v for k, v in state_dict.items()}  # remove 'model.' from keys
            self.model.load_state_dict(state_dict)
            self.model.eval()

            print(f'- Model loaded: {model_params.load_model}')

        criterions = {
            criterion_name: ClassBox.criterions[criterion.name](**criterion.params)
            for criterion_name, criterion in criterions_params.items()
        }
        self.criterions = namedtuple("criterions", criterions.keys())(**criterions)

    def configure_optimizers(self):
        from src.config import ClassBox

        assert len(self.optimizers_params) >= len(self.schedulers_params), \
            'Optimizer and scheduler numbers are not matched'

        optimizers = [
            ClassBox.optimizers[optimizer.name](self.model.parameters(), **optimizer.params)
            for optimizer in self.optimizers_params
        ]

        schedulers = [
            ClassBox.schedulers[scheduler.name](optimizer, **scheduler.params)
            for scheduler, optimizer in zip(self.schedulers_params, optimizers)
        ]
        return optimizers, schedulers

    def configure_data(self, datasets_params, dataloader_params, transforms=None, name='train'):
        from src.config import ClassBox
        from torch.utils.data import ConcatDataset, DataLoader

        if transforms is not None:
            transforms = [ClassBox.transforms[transform.name](**transform.params)
                          for transform in transforms]

        datasets = [ClassBox.datasets[dataset.name](transforms=transforms, **dataset.params)
                    for dataset in datasets_params]
        dataset = ConcatDataset(datasets)
        dataloader = DataLoader(dataset, **dataloader_params.dict())

        print_dataset_info(dataset, dataloader, name=name)

        return dataloader

    def train_dataloader(self):
        train_datasets_params = self.data_params.train.datasets
        train_dataloader_params = self.data_params.train.dataloader
        transform_params = self.data_params.train.transforms
        return self.configure_data(train_datasets_params, train_dataloader_params, transform_params, name='Train')

    def val_dataloader(self):
        val_datasets_params = self.data_params.val.datasets
        val_dataloader_params = self.data_params.val.dataloader
        return self.configure_data(val_datasets_params, val_dataloader_params, name='Validation')

    def test_dataloader(self):
        if self.data_params.test is None:
            return self.val_dataloader()
        test_datasets_params = self.data_params.test.datasets
        test_dataloader_params = self.data_params.test.dataloader
        return self.configure_data(test_datasets_params, test_dataloader_params, name='Test')
