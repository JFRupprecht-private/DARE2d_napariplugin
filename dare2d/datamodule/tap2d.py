import albumentations as A
from dare2d.datamodule.datamodule import Datamodule
import cv2
from dare2d.datamodule.augmentation import ElasticHistogram


class Tap2Datamodule(Datamodule):
    def __init__(self, train, val, test, **kwargs) -> None:
        super().__init__(train, val, test, **kwargs)

    def get_augmentations(self):
        brighten = A.RandomBrightnessContrast(
            p=0.5, brightness_limit=[.3, .5], contrast_limit=[.3, .5])
        flip = A.Flip(p=0.5)
        flip_with_noise = A.Compose([flip,
                                     A.OneOf([
                                         A.GaussianBlur(blur_limit=23),
                                         A.GaussNoise(var_limit=30. / 255),
                                         A.MedianBlur(blur_limit=5)
                                     ], p=0.5)
                                     ])
        rotate = A.ShiftScaleRotate(
            p=0.5, shift_limit=0, scale_limit=0, rotate_limit=180, border_mode=cv2.BORDER_CONSTANT)
        distort = A.OneOf([
            A.GridDistortion(p=0.5),
            A.ElasticTransform(sigma=0.5, alpha_affine=10, p=0.5)
        ], p=0.5)
        elastic = ElasticHistogram(num_control_points=5, max_repeat=3, p=0.5)
        return A.Compose([flip_with_noise, rotate, distort], p=0.5, additional_targets={'image': 'image', 'image2': 'image'})
        # return None