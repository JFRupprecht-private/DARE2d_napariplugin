import logging
from random import randrange
from typing import Union, Tuple, List

import cv2
import numpy as np
import tensorflow as tf

from .cell_2dataset import Cell2Dataset
from dare2d.typing import Bipoint

log = logging.getLogger(__name__)


class Segmentation2Dataset(Cell2Dataset):
    def __init__(
        self,
        data_folder: str,
        input_channels: List[int],
        crop_size: int,
        cell_radius: int = 25,
        type_mask: str = "circle",
        scale: Union[float, List[float]] = [1.0, 1.0, 1.0],
        renorm: str = "min-max",
    ):
        """Create the Segmentation 2D Cell Dataset.
        This class is intended to be used for semantic segmentation.

        Args:
            data_folder (str): The folder where the data is stored
            input_channels (List[int]): The index of the input channels to use
            crop_size (int): _description_
            cell_radius (int, optional): _description_. Defaults to 25.
            type_mask (str, optional): The type of semantic mask to make.
            It can be one of : "ellipse", "center", "circle". Defaults to "circle".
            scale (Union[float, List[float]], optional): The scale used to rescale the bipoints and input images.
            Can either be a single float for all dimensions or a list of floats for each dimension.
            renorm (str, optional): The name of the renormalization strategy: 'min-max'
        """
        self.cell_radius = cell_radius
        self.type_mask = type_mask
        self.crop_size = crop_size
        super().__init__(data_folder, input_channels, scale, renorm)

        log.info(
            f"Final mask 0 shape: {self.masks[0].shape}, dtype {self.masks[0].dtype}, min-max: {self.masks[0].min()}-{self.masks[0].max()}"
        )

    def prepare_data(self) -> None:
        """Create all masks beforehand."""
        self.masks = self.get_all_mask()

    def get_all_mask(self) -> List[np.ndarray]:
        """Create all semantic segmentation masks from bipoints.

        Returns:
            List[np.ndarray]: List of semantic segmentation masks.
        """
        masks = []
        for i, bipoints in enumerate(self.bipoints):
            mask_shape = self.images[i].shape[0:2]
            mask = self.make_a_mask(bipoints, mask_shape)
            mask = (mask / 255).astype(np.uint8)
            masks.append(mask)
        log.info(f"CONSTRUCTED {len(masks)} mask  with {self.type_mask}")
        mask0 = masks[0]
        log.info(
            f" -> mask 0 shape: {mask0.shape}, dtype {mask0.dtype}, min-max: {mask0.min()}-{mask0.max()} "
        )
        return masks

    def make_a_mask(self, bipoints: List[Bipoint], mask_shape: tuple) -> np.ndarray:
        """Create a single semantic segmentation mask based on bipoints.

        Args:
            bipoints (List[Bipoint]): The list of bipoints to use to make the mask
            mask_shape (tuple): The size of the mask

        Returns:
            np.ndarray: The semantic segmentation binary mask.
        """
        mask = np.zeros(mask_shape, dtype=np.uint8)
        for bipoint in bipoints:
            p1, p2 = bipoint
            center = (p1 + p2) / 2
            if self.type_mask == "ellipse":
                distance, angle = self.compute_polar_from_one_bipoint(bipoint)
                cv2.ellipse(
                    mask,
                    center=np.flip(center).astype(int),
                    axes=(int(distance), int(self.cell_radius / 2)),
                    angle=angle,
                    startAngle=0,
                    endAngle=360,
                    color=255,
                    thickness=4,
                )
            elif self.type_mask == "circle":
                cv2.circle(mask, np.flip(p1).astype(int), self.cell_radius, 255, -1)
                cv2.circle(mask, np.flip(p2).astype(int), self.cell_radius, 255, -1)
            elif self.type_mask == "center":
                cv2.circle(mask, np.flip(center).astype(int), self.cell_radius, 255, -1)
        return mask

    def augment_sample(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation to an image and its corresponding mask.

        Args:
            X (np.ndarray): The input image
            Y (np.ndarray): The ground truth mask

        Returns:
            Tuple[np.ndarray, np.ndarray]: The augmented input image and groundtruth mask.
        """
        augmented = self._augmentations(image=X, mask=Y)
        augm_image = augmented["image"]
        augm_mask = augmented["mask"]
        return augm_image, augm_mask

    def random_crop(
        self, X: np.ndarray, Y: np.ndarray, width: int, height: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform a random crop on input image and mask

        Args:
            X (np.ndarray): input image
            Y (np.ndarray): groundtruth mask
            width (int): Width of the crop
            height (int): Height of the crop

        Returns:
            Tuple[np.ndarray, np.ndarray]: crop of input image and groundtruth mask.
        """
        if X.shape[0] <= height or X.shape[1] <= width:
            return X, Y
        x = randrange(X.shape[1] - width)
        y = randrange(X.shape[0] - height)
        return X[y : y + height, x : x + width], Y[y : y + height, x : x + width]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return a pair of input image and groundtruth mask.
        It performs a crop and apply data augmentation if enabled.

        Args:
            idx (int): The index of the sample.

        Returns:
            Tuple[np.ndarray, np.ndarray]: input image and groundtruth mask
        """
        X, Y = self.images[idx], self.masks[idx]

        X, Y = self.random_crop(X, Y, self.crop_size, self.crop_size)

        if self._augmentations:
            X, Y = self.augment_sample(X, Y)
        return X, Y

    def get_output_types(self):
        return (tf.float32, tf.float32)

    def get_output_shapes(self):
        return (
            (self.crop_size, self.crop_size, self.n_input_channels),
            (self.crop_size, self.crop_size),
        )
