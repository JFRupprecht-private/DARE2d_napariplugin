import logging
from typing import Union, Tuple, List
import cv2
import numpy as np


log = logging.getLogger(__name__)

from dare2d.typing import Point
from dare2d.datamodule.generator.abstract_celldataset import AbstractCellDataset


class Cell2Dataset(AbstractCellDataset):
    def __init__(
        self,
        data_folder: str,
        input_channels: List[int],
        scale: Union[float, List[float]] = [1.0, 1.0, 1.0],
        renorm: str = "min-max",
    ):
        """Create the Cell 2D dataset.
        This class is not intended to be use as is but should be subclassed.

        Args:
            data_folder (str): The folder where the data is stored
            input_channels (List[int]): The index of the input channels to use
            scale (Union[float, List[float]]): The scale used to rescale the bipoints and input images.
            Can either be a single float for all dimensions or a list of floats for each dimension.
            renorm (str): The name of the renormalization strategy: 'min-max'
        """
        super(Cell2Dataset, self).__init__(
            data_folder=data_folder,
            input_channels=input_channels,
            scale=scale,
            renorm=renorm,
        )

        log.info(
            f"Final image 0 shape: {self.images[0].shape}, dtype {self.images[0].dtype}, min-max: {self.images[0].min()}-{self.images[0].max()}"
        )

    def _load_image(self, path: str) -> np.ndarray:
        """Load a single image from path.
        Apply histogram equalization after the image was loaded.

        Args:
            path (str): The path to the image.

        Returns:
            np.ndarray: The image loaded
        """
        img_ = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_ = cv2.equalizeHist(img_)
        return img_

    def _normalize(self, renorm: str) -> None:
        for idx, img in enumerate(self.images):
            self.images[idx] = self.normalise(img, renorm)
        img0 = self.images[0]
        log.info(
            f"NORMALISED all original images with renorm ={renorm}:\n"
            f" -> image 0 shape: {img0.shape}, dtype {img0.dtype}, min-max: {img0.min()}-{img0.max()} "
        )

    def normalise(self, img: np.ndarray, renorm: str = None) -> np.ndarray:
        """Normalize the image using the renorm strategy.

        Args:
            img (np.ndarray): The image to normalize
            renorm (str, optional): The selected renorm strategy from: 'min-max'. Defaults to None.

        Returns:
            np.ndarray: The normalized image
        """
        if renorm == "min-max":
            ptp = img.ptp()
            if ptp > 0:
                img = (img - img.min()) / img.ptp()
            return img.astype(np.float32)
        else:
            return img

    def compute_polar_from_one_bipoint(self, bipoint: Tuple[Point, Point]) -> Tuple[float, float]:
        """Compute polar coordinates from a single bipoint.

        Args:
            bipoint (Tuple[Point, Point]): The bipoint used.

        Returns:
            Tuple[float, float]: distance and angle (deg)
        """
        p1, p2 = bipoint
        diff = p2 - p1
        distance = np.linalg.norm(diff)
        angle = np.arctan2(diff[1], diff[0]) * 180 / np.pi
        return distance, angle

    def get_output_types(self):
        raise NotImplementedError

    def get_output_shapes(self):
        raise NotImplementedError
