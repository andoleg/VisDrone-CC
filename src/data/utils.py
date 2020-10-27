from pathlib import Path
from torch import Generator
from torch.utils.data import random_split


def visdrone_read_train_test(data_root: Path,
                             testlist_name: str = 'testlist.txt',
                             trainlist_name: str = 'trainlist.txt') -> (list, list):
    """
    Read train/test split provided by VisDrone
    :param data_root: path to Root Dataset folder */VisDrone2020-CC
    :param testlist_name: name of file with test split
    :param trainlist_name: name of file with train split
    :return: tuple of train and test lists with folder ids
    """
    data_root = Path(data_root)

    with (data_root / trainlist_name).open() as file:
        train = file.read().split()

    with (data_root / testlist_name).open() as file:
        test = file.read().split()

    return train, test


def train_val_split(dataset: list, train_size: float = 0.9) -> (list, list):
    train_length = int(len(dataset) * train_size)
    test_length = len(dataset) - train_length
    train_split, val_split = random_split(dataset, [train_length, test_length],
                                          generator=Generator().manual_seed(42))

    return train_split, val_split
