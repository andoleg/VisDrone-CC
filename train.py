# For GoogleColab/console running
# import sys
# sys.path.append("/content/VisDrone-CC")

import yaml
import argparse

import torch
from torch.utils.data import DataLoader, ConcatDataset

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.networks import PLNetworkExtension
from src.utils.print_info import print_dataset_info
from src.config import TrainerConfig, ClassBox, PipelineConfig, Data

torch.manual_seed(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_path', type=str, help='Path to .yml file with configuration parameters .'
    )
    config_path = parser.parse_args().config_path
    config_yaml = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    trainer_params = TrainerConfig(**config_yaml['trainer'])
    pipeline_config = PipelineConfig(**config_yaml['pipeline'])
    data_params = Data(**config_yaml['data'])

    # todo optional validation
    # Load dataset
    data_params.train.datasets = [ClassBox.datasets[dataset.name](**dataset.params)
                                  for dataset in data_params.train.datasets]
    data_params.val.datasets = [ClassBox.datasets[dataset.name](**dataset.params)
                                for dataset in data_params.val.datasets]
    train_dataset = ConcatDataset(data_params.train.datasets)
    val_dataset = ConcatDataset(data_params.val.datasets)
    train_dataloader = DataLoader(train_dataset, **data_params.train.dataloader.dict())
    val_dataloader = DataLoader(val_dataset, **data_params.val.dataloader.dict())

    # Dataset info
    print_dataset_info(train_dataset, train_dataloader)
    print_dataset_info(val_dataset, val_dataloader, name='val')

    # Train
    if not isinstance(trainer_params.checkpoint_callback, bool):
        trainer_params.checkpoint_callback = ModelCheckpoint(**trainer_params.checkpoint_callback)

    trainer_callbacks_list = trainer_params.callbacks
    if trainer_callbacks_list:
        trainer_params.callbacks = [ClassBox.callbacks[trainer_callback.name](**trainer_callback.params)
                                    for trainer_callback in trainer_callbacks_list]

    trainer = Trainer(**trainer_params.dict())

    model_params = pipeline_config.model
    ExtendedNetwork = type('ExtendedNet', (ClassBox.models[model_params.name], PLNetworkExtension), {})
    model = ExtendedNetwork(**model_params.params)
    print(f'Loaded model: {model_params.name}')

    trainer.fit(model, train_dataloader, val_dataloader)
