from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import BaseModel


class BaseParams(BaseModel):
    name: str
    params: Optional[dict] = {}


class DataParams(BaseModel):
    dataset: BaseParams
    dataloader: BaseParams


class VisDroneDataConfig(BaseModel):
    data_root: str = None
    im_folder: str = 'sequences'
    an_folders: str = 'annotations'
    resize: tuple = (128, 128)
    normalize: bool = True
    train: bool = True


class DataloaderConfig(BaseModel):
    batch_size: int = 1
    shuffle: bool = False
    sampler: Optional[dict] = None
    batch_sampler: Optional[dict] = None
    num_workers: int = 0
    collate_fn: Optional[dict] = None
    pin_memory: bool = False
    drop_last: bool = False
    timeout: int = 0
    worker_init_fn: Optional[dict] = None
    multiprocessing_context: Optional[dict] = None
    generator: Optional[dict] = None


class TrainerConfig(BaseModel):
    logger: Union[BaseParams, bool] = True  ###
    callbacks: Optional[List[BaseParams]] = None
    # checkpoint_callback: dict = dict()
    checkpoint_callback: Union[dict, bool] = True
    # early_stop_callback: bool = False  ### RODO
    default_root_dir: Optional[str] = None
    gradient_clip_val: float = 0
    process_position: int = 0
    num_nodes: int = 1
    num_processes: int = 1
    gpus: Optional[Union[List[int], str, int]] = None
    auto_select_gpus: bool = False
    tpu_cores: Optional[Union[List[int], str, int]] = None
    log_gpu_memory: Optional[str] = None
    progress_bar_refresh_rate: int = 1
    overfit_batches: Union[int, float] = 0.0
    track_grad_norm: Union[int, float, str] = -1
    check_val_every_n_epoch: int = 1
    fast_dev_run: bool = False
    accumulate_grad_batches: Union[int, Dict[int, int], List[list]] = 1
    max_epochs: int = 1000
    min_epochs: int = 1
    max_steps: Optional[int] = None
    min_steps: Optional[int] = None
    limit_train_batches: Union[int, float] = 1.0
    limit_val_batches: Union[int, float] = 1.0
    limit_test_batches: Union[int, float] = 1.0
    val_check_interval: Union[int, float] = 1.0
    flush_logs_every_n_steps: int = 100  # log_save_interval: int = 100
    log_every_n_steps: int = 50  # row_log_interval: int = 50
    distributed_backend: Optional[str] = None
    sync_batchnorm: bool = False
    precision: int = 32
    weights_summary: Optional[str] = "top"
    weights_save_path: Optional[str] = None
    num_sanity_val_steps: int = 2
    truncated_bptt_steps: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None
    profiler: bool = None
    benchmark: bool = False
    deterministic: bool = False
    reload_dataloaders_every_epoch: bool = False
    auto_lr_find: Union[bool, str] = False
    replace_sampler_ddp: bool = True
    terminate_on_nan: bool = False
    auto_scale_batch_size: Union[str, bool] = False
    prepare_data_per_node: bool = True
    amp_backend: str = 'native'
