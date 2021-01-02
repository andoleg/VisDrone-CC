import yaml
import argparse

from torch.utils.data import DataLoader, ConcatDataset

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.networks import PLNetworkExtension
from src.utils.print_info import print_dataset_info
from src.utils.metrics import computeTime
from src.config import TrainerConfig, PipelineConfig, ClassBox, Data

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

    # Load dataset
    data_params.val.datasets = [ClassBox.datasets[dataset.name](**dataset.params)
                                for dataset in data_params.val.datasets]
    val_dataset = ConcatDataset(data_params.val.datasets)
    val_dataloader = DataLoader(val_dataset, **data_params.val.dataloader.dict())

    # Dataset info
    # print_dataset_info(train_dataset, train_dataloader)
    print_dataset_info(val_dataset, val_dataloader, name='val')

    # Load model
    model_params = pipeline_config.model
    ExtendedNetwork = type('Extended', (ClassBox.models[model_params.name], PLNetworkExtension), {})
    model = ExtendedNetwork.load_from_checkpoint(checkpoint_path=config_yaml['test']['checkpoint_path'])
    model.eval()
    print(f'Loaded model: {model_params.name}')

    # Trainer
    if not isinstance(trainer_params.checkpoint_callback, bool):
        trainer_params.checkpoint_callback = ModelCheckpoint(**trainer_params.checkpoint_callback)
    trainer = Trainer(**trainer_params.dict())

    # Test
    trainer.test(model, test_dataloaders=val_dataloader, verbose=True)

    if config_yaml['test']['benchmark']:
        average_time, fps = computeTime(model, device=config_yaml['trainer']['gpus'],
                                        input_size=config_yaml['test']['resize'])
        print(f'\n - Average time for model {config_yaml["pipeline"]["model"]["name"]} is {average_time}, FPS is {fps}')
