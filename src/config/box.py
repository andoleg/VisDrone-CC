from src import networks
from src import criterions
from src.data import transforms
from src.data import datasets

from torch import optim
from torch.nn import Module
from torch.nn.modules import loss
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

from pytorch_lightning import callbacks
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import LightningLoggerBase

from albumentations.core.composition import BaseCompose
from albumentations.core.transforms_interface import BasicTransform


class ClassBox:
    models: dict = {
        k: v
        for k, v in networks.__dict__.items()
        if isinstance(v, type) and issubclass(v, Module)
    }

    datasets: dict = {
        k: v
        for k, v in datasets.__dict__.items()
        if isinstance(v, type) and issubclass(v, Dataset)
    }

    callbacks: dict = {
        k: v
        for k, v in callbacks.__dict__.items()
        if isinstance(v, type) and issubclass(v, Callback)
    }

    optimizers: dict = {
        k: v
        for k, v in optim.__dict__.items()
        if isinstance(v, type) and issubclass(v, optim.Optimizer)
    }

    schedulers: dict = {
        k: v
        for k, v in lr_scheduler.__dict__.items()
        if isinstance(v, type) and issubclass(v, _LRScheduler)
    }

    dataloaders: dict = {"Default": DataLoader}

    loggers: dict = {
        k: v
        for k, v in loggers.__dict__.items()
        if isinstance(v, type) and issubclass(v, LightningLoggerBase)
    }

    criterions: dict = {
        k: v
        for k, v in criterions.__dict__.items()
        if isinstance(v, type) and issubclass(v, loss._Loss)
    }

    transforms: dict = {
        k: v
        for k, v in transforms.__dict__.items()
        if isinstance(v, type) and (issubclass(v, BasicTransform) or issubclass(v, BaseCompose))
    }
