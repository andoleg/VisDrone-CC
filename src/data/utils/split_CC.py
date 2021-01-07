from pathlib import Path
import shutil
from src.data import visdrone_read_train_test, train_val_split


def split(path, new_path):
    path, new_path = Path(path), Path(new_path)
    train_split, test_split = visdrone_read_train_test(data_root=path)
    train_split, val_split = train_val_split(train_split)
    (new_path / 'val' / 'sequences').mkdir(parents=True, exist_ok=True)
    (new_path / 'val' / 'annotations').mkdir(parents=True, exist_ok=True)
    (new_path / 'train' / 'sequences').mkdir(parents=True, exist_ok=True)
    (new_path / 'train' / 'annotations').mkdir(parents=True, exist_ok=True)

    for f_id in train_split:
        shutil.copytree((path / 'sequences' / f_id), (new_path / 'train'/ 'sequences' / f_id))
        shutil.copy((path / 'annotations' / f_id).with_suffix('.txt'), (new_path / 'train' / 'annotations' / f_id).with_suffix('.txt'))

    for f_id in val_split:
        shutil.copytree((path / 'sequences' / f_id), (new_path / 'val'/ 'sequences' / f_id))
        shutil.copy((path / 'annotations' / f_id).with_suffix('.txt'), (new_path / 'val' / 'annotations' / f_id).with_suffix('.txt'))


if __name__ == '__main__':
    path = '/Users/olega/Downloads/VisDrone2020-CC'
    new_path = '/Users/olega/Downloads/VisDrone2020-CC-split'
    split(path, new_path)
