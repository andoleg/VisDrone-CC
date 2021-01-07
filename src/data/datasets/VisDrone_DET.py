import cv2
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, List


class VisDroneDatasetDET(Dataset):
    def __init__(self, data_root: Path,
                 im_folder: str = 'images',
                 an_folder: str = 'annotations',
                 resize: tuple = (128, 128),
                 normalize: bool = True,
                 n_points: bool = True,
                 transforms: Optional[List] = None) -> None:
        """
        :param folder_ids: list of ids of dataset tracks
        :param data_root: path to root "VisDrone2020-CC" folder
        :param im_folder: name of image folder
        :param an_folder: name of annotation folder
        :param resize: desired image size
        :param normalize: image normalization
        :param n_points: if true will use number of people as label, otherwise generate density map
        """
        self.img_paths = list()  # list of tuples: (image_path, people_count)
        self.resize = resize
        self.normalize = normalize
        self.n_points = n_points
        self.transforms = transforms

        data_root = Path(data_root)
        imgs = (data_root / im_folder).glob('*')

        for img in imgs:
            annotation_path = (data_root / an_folder / img.name).with_suffix('.txt')
            annotations = self.read_annotation_file(annotation_path)

            # print(img)
            # image = cv2.imread(str(img))
            # for ann in annotations:
            #     image = cv2.rectangle(image, (int(ann[0]), int(ann[1])),
            #                           (int(ann[0]) + int(ann[2]), int(ann[1]) + int(ann[3])), (255, 0, 0), 1)
            # cv2.imwrite(f'/Users/olega/Downloads/test/{img.name}.jpg', image)

            self.img_paths.append((img, annotations))

    def __getitem__(self, item):
        image_path, label = self.img_paths[item]

        image = cv2.imread(str(image_path))

        image = cv2.resize(image, self.resize)
        if self.normalize:
            image = image / 255.0

        if self.transforms is not None:
            for transform in self.transforms:
                image = transform(image=image)['image']

        image = image.transpose(2, 0, 1)

        return image, label

    def __len__(self):
        return len(self.img_paths)

    def __str__(self):
        return [x[0] for x in self.img_paths]

    def read_annotation_file(self, filename: Path, categories: tuple = (1,2)) -> int:
        """
        :param filename: path to annotation file
        :param categories: categories of objects of VisDroneDET to use
        :return: dict {'image_id': # of people}
        """
        if self.n_points:
            with filename.open() as file:
                annotations = [x.split(',') for x in file.read().split()]
                filtered = list()
                for ann in annotations:
                    if int(ann[5]) in categories:
                        filtered.append(ann)

            return len(filtered)
        else:
            raise NotImplementedError
