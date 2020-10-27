from pathlib import Path
import torch
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer

from src.data.VisDrone_CC import VisDroneDatasetCC
from src.data.utils import visdrone_read_train_test, train_val_split
from src.networks.FCNCastellano import FCNCastellano, ExtendedFCNCastellano
from src.utils.print_info import print_dataset_info

torch.manual_seed(0)

if __name__ == '__main__':
    dataset_root = Path('/Users/olega/Downloads/VisDrone2020-CC')
    train_split, test_split = visdrone_read_train_test(data_root=dataset_root)
    train_split, val_split = train_val_split(train_split)

    train_dataset = VisDroneDatasetCC(train_split, data_root=dataset_root)
    val_dataset = VisDroneDatasetCC(val_split, data_root=dataset_root)
    train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=5, shuffle=True, )
    val_dataloader = DataLoader(val_dataset, batch_size=64, num_workers=5)

    # Dataset info
    print_dataset_info(train_dataset, train_dataloader)
    print_dataset_info(val_dataset, val_dataloader, name='val')

    model = ExtendedFCNCastellano()
    trainer = Trainer()
    trainer.fit(model, train_dataloader)