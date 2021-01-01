# For GoogleColab/console running
# import sys
# sys.path.append("/content/VisDrone-CC")

import os
import yaml
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, ConcatDataset

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.networks import FCNCastellano, FCNCastellanoBN, PLNetworkExtension
from src.utils.print_info import print_dataset_info
from src.config import TrainerConfig, ClassBox, DataloaderConfig, VisDroneDataConfig, PipelineConfig, Data

torch.manual_seed(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_path', type=str, help='Path to .yml file with configuration parameters .'
    )
    config_path = parser.parse_args().config_path
    config_yaml = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    train_dataloader_params = DataloaderConfig(**config_yaml['dataloader']['train'])
    test_dataloader_params = DataloaderConfig(**config_yaml['dataloader']['test'])
    trainer_params = TrainerConfig(**config_yaml['trainer'])
    pipeline_config = PipelineConfig(**config_yaml['pipeline'])

    # todo optional validation
    # Load dataset
    data_params = Data(**config_yaml['data'])
    data_params.train = [ClassBox.datasets[dataset.name](**dataset.params)
                         for dataset in data_params.train]
    data_params.val = [ClassBox.datasets[dataset.name](**dataset.params)
                       for dataset in data_params.val]

    train_dataset = ConcatDataset(data_params.train)
    val_dataset = ConcatDataset(data_params.val)
    train_dataloader = DataLoader(train_dataset, **train_dataloader_params.dict())
    val_dataloader = DataLoader(val_dataset, **test_dataloader_params.dict())

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
