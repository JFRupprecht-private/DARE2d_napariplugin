import logging

from typing import Union, Tuple, List, Dict
import numpy as np
import tensorflow as tf

from .cell_2dataset import Cell2Dataset
from dare2d.typing import Point, Bipoint

log = logging.getLogger(__name__)


class Regress2Dataset(Cell2Dataset):
    def __init__(
        self,
        data_folder: str,
        input_channels: List[int],
        scale: Union[float, List[float]] = [1.0, 1.0, 1.0],
        crop_size: int = 64,
        renorm: str = "min-max",
    ):
        """Create the Regress 2D Cell Dataset.
        This class is intended to be used for angle and length regression.

        Args:
            data_folder (str): The folder where the data is stored
            input_channels (List[int]): The index of the input channels to use
            scale (Union[float, List[float]]): The scale used to rescale the bipoints and input images.
            Can either be a single float for all dimensions or a list of floats for each dimension.
            crop_size (int, optional): Crop size used to perform the regression. Defaults to 64.
            renorm (str): The name of the renormalization strategy: 'min-max'
        """
        self.crop_size = crop_size
        self.half_crop_size = self.crop_size // 2
        super(Regress2Dataset, self).__init__(data_folder, input_channels, scale, renorm)

        log.info(
            f"Final crop mask 0 shape: {self.crops[0].shape}, dtype {self.crops[0].dtype}, min-max: {self.crops[0].min()}-{self.crops[0].max()}"
        )

    def __len__(self) -> int:
        """Number of crops or manual limit.

        Returns:
            int: number of crops
        """
        return min(len(self.crops), self._sample_limit)

    def prepare_data_after_norm(self) -> None:
        """Crops all division from the prepared data.
        Precompute all cosine and sine values.
        """
        self.crops, self.bipoint_crops = self.make_crop_all_division()
        self.polar_crop_all = self.compute_polar_for_all_crops()

    def make_crop_all_division(
        self,
    ) -> Tuple[List[np.ndarray], List[Bipoint]]:
        """Crop regions around each division.

        Returns:
            Tuple[List[np.ndarray], List[Bipoint]]: List of crop image with associated bipoint.
            One image for one bipoint.
        """
        crops = []
        bipoint_crops = []
        no_division_images_index = []
        for idx, (img, bipoints) in enumerate(zip(self.images, self.bipoints)):
            if len(bipoints) == 0:
                no_division_images_index.append(idx)
            else:
                ccrops, cbipoint_crops = self.crop_one_img_division(img, bipoints)
                crops.extend(ccrops)
                bipoint_crops.extend(cbipoint_crops)

        log.info(f" CROPED {len(crops)} crops from all images with one division")
        log.info(f" ->  {len(no_division_images_index)} images with no divisions")

        return crops, bipoint_crops

    def crop_img_from_center(self, img: np.ndarray, center: Point) -> np.ndarray:
        """Crop the image at the given center point.

        Args:
            img (np.ndarray): The image to crop
            center (Point): The center of the crop in the image

        Returns:
            np.ndarray: The crop
        """
        translat_half_crop = np.array([self.half_crop_size, self.half_crop_size])
        center_pad = (center + translat_half_crop).astype(int)
        start_x, start_y = center_pad - self.half_crop_size
        end_x, end_y = center_pad + self.half_crop_size
        crop = img[start_x:end_x, start_y:end_y, :]
        assert crop.shape[0] > 0 and crop.shape[1] > 0, (
            f"Expected crop to be non zero shape but found {crop.shape}"
            f"Initial image shape {img.shape} and division point {center}"
        )
        return crop

    def pad_img(self, img: np.ndarray, padding_border: str = "constant") -> np.ndarray:
        """Zero pad image with equal amount at the XY borders.

        Args:
            img (np.ndarray): Image to pad
            padding_border (str, optional): pad strategy. Defaults to "constant".

        Returns:
            np.ndarray: Padded image
        """
        return np.pad(
            img,
            pad_width=(
                (self.half_crop_size, self.half_crop_size),
                (self.half_crop_size, self.half_crop_size),
                (0, 0),
            ),
            mode=padding_border,
            constant_values=0,
        )

    def crop_one_img_division(
        self,
        img: np.ndarray,
        bipoints: List[Bipoint],
        padding_border: str = "constant",
    ) -> Tuple[List[np.ndarray], List[Bipoint]]:
        """Crop a single image for each division it contains.

        Args:
            img (np.ndarray): The image to process
            bipoints (List[Bipoint]): The list of division in the image.
            padding_border (str, optional): pad strategy. Defaults to "constant".

        Returns:
            Tuple[List[np.ndarray], List[Bipoint]]: List of crops and bipoint for every divisions.
        """
        assert len(img.shape) == 3

        img_pad = self.pad_img(img, padding_border)
        crops = []
        bipoint_crops = []

        translat_half_crop = np.array([self.half_crop_size, self.half_crop_size])
        for bipoint in bipoints:
            p1, p2 = bipoint
            center = (p1 + p2) / 2
            p1_center, p2_center = p1 - center, p2 - center

            if (
                center[0] < 0
                or center[1] < 0
                or center[0] >= img_pad.shape[0] - self.crop_size
                or center[1] >= img_pad.shape[1] - self.crop_size
            ):
                continue

            crop = self.crop_img_from_center(img_pad, center)
            assert crop.shape[0] == self.crop_size and crop.shape[1] == self.crop_size
            crops.append(crop)

            new_p1_crop = p1_center + translat_half_crop
            new_p2_crop = p2_center + translat_half_crop

            bipoint_crops.append(tuple([new_p1_crop, new_p2_crop]))

        return crops, bipoint_crops

    def degree_to_twice_cosin(self, angle_degree: float) -> List[float]:
        """Use the twice angle tricks.
        The goal is to compute twice the angle in radians of the given angle in degree to make
        sur that angles that are near 180° and 0° are close.
        By computing twice the angle we have 180°x2=360° and 0°x2=0° making effectively 180° and 0° close as
        360°=0°.
        Then we regress the cosine and sine values instead of the angle in radians or in degree.

        Args:
            angle_degree (float): The angle to convert

        Returns:
            List[float, float]: Cosine and sine values of twice the angle
        """
        # Degrees to radians
        angle_rad = angle_degree / 180.0 * np.pi
        # From [-pi; pi] range to [0; pi]
        angle_rad = angle_rad % np.pi
        # Compute twice the angle
        angle_rad = 2 * angle_rad
        return [np.cos(angle_rad), np.sin(angle_rad)]

    def compute_polar_for_all_crops(self) -> List[dict]:
        """Compute cosine and sine of twice the angle and distance for all crops.

        Returns:
            List[dict]: Length and cosine and sine of twice the angle for each crop.
        """
        # of dict{"length_output, angle_output}
        polar_all = []
        for bipoint in self.bipoint_crops:
            distance, cosin = self.distance_angle_from_bipoint(bipoint)
            y = {"length_output": np.array([distance]), "angle_output": cosin}
            polar_all.append(y)
        return polar_all

    def distance_angle_from_bipoint(self, bipoint: Bipoint) -> Tuple[float, Tuple[float, float]]:
        """Compute cosine and sine of twice the angle and distance for a single crop.

        Args:
            bipoint (Bipoint): The bipoint to compute angle and distance from

        Returns:
            Tuple[float, Tuple[float, float]]: distance and cosine and sine of twice the angle.
        """
        distance, angle = self.compute_polar_from_one_bipoint(bipoint)
        cosin = self.degree_to_twice_cosin(angle)
        distance = distance / self.crop_size
        return distance, cosin

    def augment_sample(self, X: np.ndarray, Y: List[Bipoint]) -> Tuple[np.ndarray, List[Bipoint]]:
        """Augment a pair of input image and bipoint.

        Args:
            X (np.ndarray): Image to augment
            Y (List[Bipoint]): Bipoint to augment

        Returns:
            Tuple[np.ndarray, List[Bipoint]]: Augmented image and bipoint
        """
        keypoints = Y
        augmented = self._augmentations(image=X, keypoints=keypoints)
        X = augmented["image"]
        Y = [np.array(p) for p in augmented["keypoints"]]
        return X, Y

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict]:
        """Return the sample at given index.
        Apply data augmentation if enabled.

        Args:
            idx (int): Index of the sample

        Returns:
            Tuple[np.ndarray, Dict]: Pair of image and length/cosin values in a dict
        """
        X, bipoints = self.crops[idx], self.bipoint_crops[idx]
        if self._augmentations:
            n_X, n_bipoints = self.augment_sample(X, bipoints)
            # In case something went wrong with the augmentation...
            # Skip augmentation for this sample
            if len(n_bipoints) == 2:
                X, bipoints = n_X, n_bipoints

        distance, cosin = self.distance_angle_from_bipoint(bipoints)
        Y = {"length_output": np.array([distance]), "angle_output": cosin}
        X = self.normalise(X, "min-max")
        return X, Y

    def display(self, X: np.ndarray, bipoints: List[Bipoint]) -> None:
        """Display a sample

        Args:
            X (np.ndarray): Input image
            bipoints (List[Bipoint]): List of bipoints in the image
        """
        p1, p2 = bipoints
        import matplotlib

        matplotlib.use("TKAgg")
        import matplotlib.pyplot as plt

        plt.imshow(X[:, :, -1], cmap="gray")
        plt.plot([p1[1], p2[1]], [p1[0], p2[0]], color="red")
        plt.title("Crop n° à t")
        plt.pause(0)
        plt.clf()

    def get_output_types(self):
        return (tf.float32, {"length_output": tf.float32, "angle_output": tf.float32})

    def get_output_shapes(self):
        return (self.crop_size, self.crop_size, self.n_input_channels), {
            "length_output": (1,),
            "angle_output": (2,),
        }
