from albumentations.core.transforms_interface import DualTransform
from scipy import interpolate
import numpy as np
import copy


def histogram_voodoo(image, num_control_points: int = 3):
    """
    TAKEN FROM: https://gitlab.com/dunloplab/delta/-/blob/main/delta/data.py

    This function kindly provided by Daniel Eaton from the Paulsson lab.
    It performs an elastic deformation on the image histogram to simulate
    changes in illumination

    Parameters
    ----------
    image : 2D numpy array
        Input image.
    num_control_points : int, optional
        Number of inflection points to use on the histogram conversion curve.
        The default is 3.

    Returns
    -------
    2D numpy array
        Modified image.

    """
    control_points = np.linspace(0, 1, num=num_control_points + 2)
    sorted_points = copy.copy(control_points)
    random_points = np.random.uniform(
        low=0.1, high=0.9, size=num_control_points)
    sorted_points[1:-1] = np.sort(random_points)
    mapping = interpolate.PchipInterpolator(control_points, sorted_points)

    if image.dtype == np.uint8:
        image = image.astype(np.float32)
        image /= 255.
        result = mapping(image)
        result *= 255.
        return result.astype(np.uint8)

    return mapping(image)


class ElasticHistogram(DualTransform):
    """Perform elastic deformation on the histogram.
    Args:
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8
    """

    def __init__(
        self,
        num_control_points=3,
        max_repeat=1,
        always_apply=False,
        p=0.5,
    ):
        super(ElasticHistogram, self).__init__(always_apply, p)
        self.num_control_points = num_control_points
        self.max_repeat = max_repeat

    def apply(self, img, repeat, **params):
        transform = img
        for _ in range(repeat):
            transform = histogram_voodoo(transform, self.num_control_points)
        return transform

    def apply_to_mask(self, img, angle=0, scale=0, dx=0, dy=0, **params):
        return img

    def apply_to_keypoint(self, keypoint, angle=0, scale=0, dx=0, dy=0, rows=0, cols=0, **params):
        return keypoint

    def get_params(self):
        return {
            "repeat": np.random.randint(1, self.max_repeat) if self.max_repeat > 1 else 1
        }

    def apply_to_bbox(self, bbox, angle, scale, dx, dy, **params):
        return bbox

    def get_transform_init_args(self):
        return {}
