import numpy as np
from pathlib import Path
from torch import Generator
from torch.utils.data import random_split
from cv2 import resize, INTER_CUBIC

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


def gen_discrete_map(im_size, points, resize_shape=None):
    """
    Generate the discrete map.
    :param points: [num_gt, 2]
    :return:
    """
    points = np.array(points)
    discrete_map = np.zeros(im_size, dtype=np.float32)
    h, w = discrete_map.shape[:2]
    num_gt = points.shape[0]
    if num_gt == 0:
        return discrete_map
    for p in points:
        p = np.round(p).astype(int)
        p[0], p[1] = min(h - 1, p[1]), min(w - 1, p[0])
        discrete_map[p[0], p[1]] += 1

    # print(f'W: {w}, H: {h}, map: {discrete_map.shape}')
    if type(resize_shape) is tuple:
        # discrete_map = np.resize(discrete_map, resize)
        # down_w = (w // 2) - 8
        # down_h = (h // 2) - 8
        # discrete_map = discrete_map.reshape([down_h, 2, down_w, 2]).sum(axis=(1, 3))
        print(f'Before: {discrete_map.shape}')
        discrete_map = resize(discrete_map, dsize=resize_shape)
        print(f'After: {discrete_map.shape}, GT: {num_gt}, New: {np.sum(discrete_map)}')

        # todo solve reshaping issue

    assert np.sum(discrete_map) == num_gt
    return discrete_map