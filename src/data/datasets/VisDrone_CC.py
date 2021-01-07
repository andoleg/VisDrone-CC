import cv2
from torch.utils.data import Dataset
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, List

from src.data.utils.utils import gen_discrete_map


class VisDroneDatasetCC(Dataset):
    def __init__(self, data_root: Path,
                 im_folder: str = 'images',
                 an_folder: str = 'annotations',
                 resize: tuple = (128, 128),
                 normalize: bool = True,
                 train: bool = True,
                 n_points: bool = True,
                 transforms: Optional[List] = None) -> None:
        """
        :param folder_ids: list of ids of dataset tracks
        :param data_root: path to root "VisDrone2020-CC" folder
        :param im_folder: name of image folder
        :param an_folder: name of annotation folder
        :param resize: desired image size
        :param normalize: image normalization
        :param train: if false, uses test data (that has no annotations)
        :param n_points: if true will use number of people as label, otherwise generate density map
        :param transforms: list of Albumentation transforms
        """
        self.img_paths = list()  # list of tuples: (image_path, people_count)
        self.resize = resize
        self.normalize = normalize
        self.train = train
        self.n_points = n_points
        self.transforms = transforms

        data_root = Path(data_root)
        folder_ids = [x.name.split('.')[0] for x in (data_root / an_folder).glob('*.txt')]

        if self.train:
            for f_id in folder_ids:
                imgs = (data_root / im_folder / f_id).glob('*')

                annotation_path = (data_root / an_folder / f_id).with_suffix('.txt')
                annotations = self.read_annotation_file(annotation_path)

                for img in imgs:
                    self.img_paths.append((img, annotations[img.stem.lstrip('0')]))
        else:
            for f_id in folder_ids:
                imgs = (data_root / im_folder / f_id).glob('*')
                for img in imgs:
                    self.img_paths.append((img, -1))

    def __getitem__(self, item):
        image_path, label = self.img_paths[item]

        image = cv2.imread(str(image_path))
        in_shape = image.shape[:2]
        image = cv2.resize(image, self.resize)
        if self.normalize:
            image = image / 255.0

        if self.transforms is not None:
            for transform in self.transforms:
                image = transform(image=image)['image']

        image = image.transpose(2, 0, 1)
        if not self.n_points:
            label = gen_discrete_map(in_shape, label, tuple(x // 2 - 8 for x in self.resize))

        if self.train:
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.img_paths)

    def __str__(self):
        return [x[0] for x in self.img_paths]

    def read_annotation_file(self, filename: Path) -> dict:
        """
        :param filename: path to annotation file
        :param return_points: return # of points or list of points (True for list)
        :return: dict {'image_id': # of people}
        """
        if self.n_points:
            with filename.open() as file:
                annotations = [x.split(',') for x in file.read().split()]

            return Counter([x[0] for x in annotations])
        else:
            with filename.open() as file:
                annotations = [x.split(',') for x in file.read().split()]

            points = defaultdict(list)
            for point in annotations:
                points[point[0]].append((int(point[1]), int(point[2])))
            return points
