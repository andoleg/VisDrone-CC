import cv2
import numpy as np
import albumentations as A
from albumentations import ImageOnlyTransform


class Grayscale(ImageOnlyTransform):
    def __init__(self, num_output_channels=1, always_apply=False, p=1):
        super().__init__(always_apply, p)
        self.num_output_channels = num_output_channels

    def apply(self, img, **params):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if self.num_output_channels == 1:
            return np.expand_dims(gray, -1)
        else:
            return cv2.merge([gray] * self.num_output_channels)
