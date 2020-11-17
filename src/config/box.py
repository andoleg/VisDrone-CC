from pytorch_lightning.callbacks import Callback
from pytorch_lightning import callbacks


class ClassBox:
    callbacks: dict = {
        k: v
        for k, v in callbacks.__dict__.items()
        if isinstance(v, type) and issubclass(v, Callback)
    }