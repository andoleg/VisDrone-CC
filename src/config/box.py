from src import networks

from torch.nn import Module
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import callbacks

from torch import optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning import loggers
from pytorch_lightning.loggers import LightningLoggerBase


class ClassBox:
    models: dict = {
        k: v
        for k, v in networks.__dict__.items()
        if isinstance(v, type) and issubclass(v, Module)
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
