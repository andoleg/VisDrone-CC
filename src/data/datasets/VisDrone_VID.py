from collections import defaultdict

import cv2
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, List

from src.data.utils.utils import generate_weight_distribution


class VisDroneDatasetVID(Dataset):
    def __init__(self, data_root: Path,
                 im_folder: str = 'sequences',
                 an_folder: str = 'annotations',
                 resize: tuple = (128, 128),
                 normalize: bool = True,
                 n_points: bool = True,
                 transforms: Optional[List] = None,
                 weighted: bool = False) -> None:
        """
        :param folder_ids: list of ids of dataset tracks
        :param data_root: path to root "VisDrone2020-VID" folder
        :param im_folder: name of image folder
        :param an_folder: name of annotation folder
        :param resize: desired image size
        :param normalize: image normalization
        :param n_points: if true will use number of people as label, otherwise generate density map
        :param weighted: True to create weights for less represented samples
        """
        self.img_paths = list()  # list of tuples: (image_path, people_count)
        self.resize = resize
        self.normalize = normalize
        self.n_points = n_points
        self.transforms = transforms

        self.weighted = weighted

        data_root = Path(data_root)
        folders = (data_root / im_folder).glob('uav*')

        for subfolder in folders:
            annotation_path = (data_root / an_folder / subfolder.name).with_suffix('.txt')
            annotations = self.read_annotation_file(annotation_path)

            for img in subfolder.glob('*'):
                img_id = int(img.stem.lstrip('0'))
                img_annotation = annotations[img_id]
                if img_annotation:
                    self.img_paths.append((img, img_annotation))

        if weighted:
            self.weight_distribution = generate_weight_distribution(self.img_paths, self.weighted)

    def __getitem__(self, item):
        image_path, label = self.img_paths[item]

        image = cv2.imread(str(image_path)).astype('float32')

        image = cv2.resize(image, self.resize)
        if self.normalize:
            image = image / 255.0

        if self.transforms is not None:
            for transform in self.transforms:
                image = transform(image=image)['image']

        image = image.transpose(2, 0, 1)

        if self.weighted:
            quotient = label // self.weighted
            closest_biggest = quotient * self.weighted + self.weighted
            weight = self.weight_distribution[closest_biggest]
            return image, label, weight
        return image, label

    def __len__(self):
        return len(self.img_paths)

    def __str__(self):
        return [x[0] for x in self.img_paths]

    def read_annotation_file(self, filename: Path, categories: tuple = (1,2)) -> defaultdict:
        """
        :param filename: path to annotation file
        :param categories: categories of objects of VisDroneDET to use
        :return: dict {'image_id': # of people}

        annotation style:
        <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
        """
        if self.n_points:
            with filename.open() as file:
                annotations = [[int(r) for r in x.split(',')] for x in file.read().split()]

            filtered = defaultdict(list)
            for ann in annotations:
                if ann[7] in categories:
                    filtered[ann[0]].append(ann)

            for key, value in filtered.items():
                filtered[key] = len(value)
            return filtered
        else:
            raise NotImplementedError
