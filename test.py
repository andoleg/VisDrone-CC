import yaml
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data import VisDroneDatasetCC
from src.data import visdrone_read_train_test, train_val_split
from src.networks.FCNCastellano import FCNCastellano, ExtendedFCNCastellano
from src.utils.print_info import print_dataset_info
from src.config import TrainerConfig, VisDroneDataConfig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_path', type=str, help='Path to .yml file with configuration parameters .'
    )
    config_path = parser.parse_args().config_path
    config_yaml = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    # Load data
    dataset_params = VisDroneDataConfig(**config_yaml['data'])
    dataset_params.data_root = Path(dataset_params.data_root)
    # dataset_root = Path(config_yaml['data']['dataset_root'])
    train_split, test_split = visdrone_read_train_test(data_root=dataset_params.data_root)
    train_split, val_split = train_val_split(train_split)

    train_dataset = VisDroneDatasetCC(train_split, **dataset_params.dict())
    val_dataset = VisDroneDatasetCC(val_split, **dataset_params.dict())
    # test_dataset = VisDroneDatasetCC(test_split, data_root=dataset_root, resize=(512, 512), train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=5, shuffle=True, )
    val_dataloader = DataLoader(val_dataset, batch_size=64, num_workers=5)
    # test_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=5)

    # Dataset info
    # print_dataset_info(train_dataset, train_dataloader)
    print_dataset_info(val_dataset, val_dataloader, name='val')

    # Load model
    model = ExtendedFCNCastellano.load_from_checkpoint(checkpoint_path=config_yaml['test']['checkpoint_path'])

    # Trainer
    trainer_params = TrainerConfig(**config_yaml['trainer'])
    if not isinstance(trainer_params.checkpoint_callback, bool):
        trainer_params.checkpoint_callback = ModelCheckpoint(**trainer_params.checkpoint_callback)
    trainer = Trainer(**trainer_params.dict())

    # Test
    trainer.test(model, test_dataloaders=val_dataloader, verbose=True)
    # todo check if logging works correctly with tests
