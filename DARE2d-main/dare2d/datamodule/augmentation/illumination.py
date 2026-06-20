import numpy as np
from scipy import interpolate
from albumentations.core.transforms_interface import DualTransform


def illumination_voodoo(image, num_control_points: int = 5):
    """
    This function inspired by the one above.
    It simulates a variation in illumination along the length of the chamber

    Parameters
    ----------
    image : 2D numpy array
        Input image.
    num_control_points : int, optional
        Number of inflection points to use on the illumination multiplication
        curve.
        The default is 5.

    Returns
    -------
    newimage : 2D numpy array
        Modified image.

    """

    # Create a random curve along the length of the chamber:
    control_points = np.linspace(0, image.shape[0] - 1, num=num_control_points)
    random_points = np.random.uniform(
        low=0.1, high=0.9, size=num_control_points)
    mapping = interpolate.PchipInterpolator(control_points, random_points)
    curve = mapping(np.linspace(0, image.shape[0] - 1, image.shape[0]))
    # Apply this curve to the image intensity along the length of the chamebr:
    newimage = np.multiply(
        image,
        np.reshape(
            np.tile(np.reshape(curve, curve.shape + (1,)),
                    (1, image.shape[1])),
            image.shape,
        ),
    )
    # Rescale values to original range:
    newimage = np.interp(
        newimage, (newimage.min(), newimage.max()), (image.min(), image.max())
    )

    return newimage.astype(np.uint8)


class Illumination(DualTransform):
    """Perform random illumination on the image.
    Args:
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8
    """

    def __init__(
        self,
        num_control_points=5,
        max_repeat=1,
        always_apply=False,
        p=0.5,
    ):
        super(Illumination, self).__init__(always_apply, p)
        self.num_control_points = num_control_points
        self.max_repeat = max_repeat

    def apply(self, img, repeat, **params):
        transform = img
        for _ in range(repeat):
            for i in range(img.shape[-1]):
                transform[:, :, i] = illumination_voodoo(
                    transform[:, :, i], self.num_control_points)
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
