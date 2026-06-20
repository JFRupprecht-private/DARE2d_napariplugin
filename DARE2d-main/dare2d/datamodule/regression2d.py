from albumentations import (
    Compose,
    Flip,
    GaussianBlur,
    GaussNoise,
    HorizontalFlip,
    OneOf,
    RandomBrightnessContrast,
    ShiftScaleRotate,
    VerticalFlip,
    Equalize,
    KeypointParams
)
import albumentations as A
import cv2
from dare2d.datamodule.augmentation import ElasticHistogram
from dare2d.datamodule.datamodule import Datamodule


class Regression2dDatamodule(Datamodule):
    def __init__(self, train, val, test, **kwargs) -> None:
        super().__init__(train, val, test, **kwargs)

    def get_augmentations(self):
        D4 = A.Compose([A.Flip(p=1),
                        A.RandomRotate90(p=1)])
        D4_with_noise = A.Compose([D4,
                                   A.OneOf([
                                       A.GaussianBlur(blur_limit=23),
                                       A.GaussNoise(var_limit=30. / 255),
                                       A.MedianBlur(blur_limit=5)
                                   ], p=1)
                                   ])
        brighten = A.RandomBrightnessContrast(
            p=0.5, brightness_limit=[.3, .7], contrast_limit=[.3, .7])
        rotate = A.ShiftScaleRotate(
            p=0.5, shift_limit=0.0425, scale_limit=[0, 0.6], rotate_limit=180, border_mode=cv2.BORDER_CONSTANT)
        elastic = ElasticHistogram(num_control_points=5, max_repeat=3, p=0.5)
        return A.Compose([D4_with_noise, rotate, brighten, elastic], p=0.5, keypoint_params=KeypointParams(format='yx'))
