import yaml
import argparse

from torch.utils.data import DataLoader, ConcatDataset

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.networks import PLNetworkExtension
from src.utils.print_info import print_dataset_info
from src.utils.metrics import computeTime
from src.config import TrainerConfig, PipelineConfig, ClassBox, Data
from src.pipelines import CCPipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_path', type=str, help='Path to .yml file with configuration parameters .'
    )
    config_path = parser.parse_args().config_path
    config_yaml = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    trainer_params = TrainerConfig(**config_yaml['trainer'])
    pipeline_params = PipelineConfig(**config_yaml['pipeline'])

    pipeline = CCPipeline(
        pipeline_params.model,
        pipeline_params.criterions,
        pipeline_params.optimizers,
        pipeline_params.schedulers,
        pipeline_params.data
    )

    # Trainer
    if not isinstance(trainer_params.checkpoint_callback, bool):
        trainer_params.checkpoint_callback = ModelCheckpoint(**trainer_params.checkpoint_callback)
    trainer_params.logger = ClassBox.loggers[trainer_params.logger.name](**trainer_params.logger.params)
    trainer = Trainer(**trainer_params.dict())

    # Test
    trainer.test(pipeline)

    if config_yaml['test']['benchmark']:
        average_time, fps = computeTime(pipeline.model, device=config_yaml['trainer']['gpus'],
                                        input_size=config_yaml['test']['resize'])
        print(f'\n - Average time for model {config_yaml["pipeline"]["model"]["name"]} is {average_time}, FPS is {fps}')
