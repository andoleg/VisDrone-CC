import cv2
from torch.utils.data import Dataset
from collections import Counter

from pathlib import Path


class VisDroneDatasetCC(Dataset):
    def __init__(self, folder_ids: list, data_root: Path,
                 im_folder: str = 'sequences',
                 an_folders: str = 'annotations',
                 resize: tuple = (128, 128),
                 normalize: bool = True,
                 train: bool = True) -> None:
        """
        :param folder_ids: list of ids of dataset tracks
        :param data_root: path to root "VisDrone2020-CC" folder
        :param im_folder: name of image folder
        :param an_folders: name of annotation folder
        :param resize: desired image size
        :param normalize: image normalization
        :param train: if false, uses test data (that has no annotations)
        """
        self.img_paths = list()  # list of tuples: (image_path, people_count)
        self.resize = resize
        self.normalize = normalize
        self.train = train

        data_root = Path(data_root)
        if self.train:
            for f_id in folder_ids:
                imgs = (data_root / im_folder / f_id).glob('*')

                annotation_path = (data_root / an_folders / f_id).with_suffix('.txt')
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
        image = cv2.resize(image, self.resize)
        if self.normalize:
            image = image / 255.0

        image = image.transpose(2, 0, 1)

        if self.train:
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.img_paths)

    @staticmethod
    def read_annotation_file(filename: Path) -> dict:
        """
        :param filename: path to annotation file
        :return: dict {'image_id': # of people}
        """
        with filename.open() as file:
            annotations = [x.split(',') for x in file.read().split()]

        return Counter([x[0] for x in annotations])


# a = VisDroneDatasetCC(['00001'], '/Users/olega/Downloads/VisDrone2020-CC', resize=(700, 700))
# image, label = a[0]
# print(label)
# cv2.imshow('image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
