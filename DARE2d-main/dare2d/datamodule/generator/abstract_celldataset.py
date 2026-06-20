"""
Contains an abstract class that can be used to built 2D and 3D cell datasets.
This class provides interfaces that can be both used for segmentation and regression.
"""

import os
import re
from pathlib import Path
from typing import Union, Tuple, List

import numpy as np
import skimage.transform as skt
import tensorflow as tf
from tqdm import tqdm


def atoi(text: str) -> int:
    """Alphanumerical to integer.

    Args:
        text (str): Text to convert

    Returns:
        int: The text cast as integer if it is and integer.
    """
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]


class AbstractCellDataset(tf.keras.utils.Sequence):
    def __init__(
        self,
        data_folder: str,
        input_channels: List[int],
        scale: Union[float, List[float]],
        renorm: str,
    ):
        """Create the cell dataset

        You can modulate the input channels to use. At time 't' the index
        is 0 so if you want to use t-1 aswell you can make a list with [-1, 0].
        Only -1, 0, 1 time steps are available.

        Args:
            data_folder (str): The folder where the data is stored
            input_channels (List[int]): The index of the input channels to use
            scale (Union[float, List[float]]): The scale used to rescale the bipoints and input images.
            Can either be a single float for all dimensions or a list of floats for each dimension.
            renorm (str): The name of the renormalization strategy: 'min-max'
        """
        self.data_folder = data_folder
        self.input_channels = input_channels
        self.n_input_channels = len(self.input_channels)
        self.scale = np.array(scale)
        self._augmentations = None
        self._sample_limit = np.inf
        self.idx = 0

        # Load data samples paths
        self._list_data()
        # Load images and associated bipoints
        # Data are also rescaled based on the given scale
        self.images, self.bipoints = self._load_data()

        self.prepare_data()

        if renorm:
            self._normalize(renorm)

        self.prepare_data_after_norm()

    def prepare_data_after_norm(self) -> None:
        """Simple hook function to format the data after it has been normalized."""
        pass

    def set_sample_limit(self, sample_limit: float) -> None:
        """Set the maximum index to use by the given amount.
        This is useful when you want to use a subset of the data.

        Args:
            sample_limit (float): The quantity to use within range [0,len(self.images)].
            0 is no data and len(self.images) is all data.
        """
        if sample_limit == np.inf:
            sample_limit = len(self.images)
        assert sample_limit >= 0.0 and sample_limit <= len(
            self.images
        ), f"Expected sample limit to be within [0;{len(self.images)}] but found {sample_limit}"
        self.sample_limit = sample_limit

    def resize(self, img: np.ndarray, target_img_shape: tuple) -> np.ndarray:
        """Resize the image to the target shape.
        If the image has already the target shape do nothing.

        Args:
            img (np.ndarray): The image to resize
            target_img_shape (tuple): The target shape of the image

        Returns:
            np.ndarray: The resized image.
        """
        if img.shape == target_img_shape:
            return img
        return skt.resize(
            img,
            target_img_shape,
            order=1,
            mode="reflect",
            cval=0,
            clip=True,
            anti_aliasing=True,
        )

    def prepare_data(self) -> None:
        """Hook function to prepare data after loading and before normalization."""
        pass

    def _normalize(self, norm_type: str) -> None:
        """Function where the data must be normalized following the given norm_type

        Args:
            norm_type (str): Normalization strategy to use.
        """
        raise NotImplementedError

    def _load_image(self, path: str) -> np.ndarray:
        """Load a single image from path.

        Args:
            path (str): The path to the image.

        Returns:
            np.ndarray: The image loaded
        """
        raise NotImplementedError

    def get_channel_folder(self, channel_index: int) -> str:
        """Retrieve the correct channel folder given a channel index.

        Args:
            channel_index (int): The channel index of the folder.

        Raises:
            ValueError: The channel index is not in [-1; 1]

        Returns:
            str: The name of the channel folder for the channel index.
        """
        if channel_index == -1:
            return "previmg"
        if channel_index == 0:
            return "currimg"
        if channel_index == 1:
            return "nextimg"
        raise ValueError(f"Channel index {channel_index} is not in valid range [-1; 1]")

    def _load_data(
        self,
    ) -> Tuple[List[np.ndarray], List[Tuple[np.float16, np.float16]]]:
        """Load the image and the bipoints at the same time.
        It loads dynamically the parametrized image channels (concatenated) then resize the image.
        The bipoints are then loaded and also transformed to be in par with the image size.

        Returns:
            Tuple[List[np.ndarray], List[Tuple[np.float16, np.float16]]]: _description_
        """
        images = []
        bipoints = []

        for img_file in tqdm(self.img_name_list, total=len(self.img_name_list)):
            stack = []
            # Dynamic loading with flexible number of input time steps
            for i in range(len(self.input_channels)):
                channel_folder = self.get_channel_folder(self.input_channels[i])
                img_path = os.path.join(self.data_folder, channel_folder, img_file)
                img_ = self._load_image(img_path)
                stack.append(img_)

            # Convert image stack array to numpy array
            im = np.stack(stack, axis=-1)

            # Resize the image if needed: based on given scale
            target_shape = self._compute_target_shape(im.shape)

            im = self.resize(im, target_shape)

            bipoints_path = os.path.join(
                self.data_folder, "div_location", img_file.replace(".tif", ".npy")
            )

            bipoints_img = self._load_bipoints(bipoints_path)
            images.append(im)
            bipoints.append(bipoints_img)
        print(f" LOADED {len(images)} images with {self.n_input_channels} timestamps")
        img0 = images[0]
        print(
            f" -> image 0 shape: {img0.shape}, dtype {img0.dtype}, min-max: {img0.min()}-{img0.max()} "
        )

        return images, bipoints

    def _list_data(self) -> None:
        """List all the image data in the data folder.
        Throws assertion error when there is no images.
        """
        img_file_list = os.listdir(self.data_folder + "/currimg")
        times = sorted([Path(x).stem for x in img_file_list], key=natural_keys)
        self.img_name_list = [x + ".tif" for x in times]

        print(self.img_name_list)
        assert len(self.img_name_list) > 0, f"Found no images in {img_file_list}"

    def _load_bipoints(self, bipoints_path: str) -> List[Tuple[np.float16, np.float16]]:
        """Load the bipoints from a bipoints path.

        Args:
            bipoints_path (str): The path to the bipoints for a single image.

        Raises:
            ValueError: The bipoints file is corrupted since len(bipoints)%2 != 0

        Returns:
            List[Tuple[np.float16, np.float16]]: The list of loaded bipoints
        """
        center_coords = np.load(bipoints_path)
        nb_points = center_coords.shape[0]
        if nb_points % 2 != 0:
            raise ValueError(
                f" file {bipoints_path.split('/')[-1]} contains {nb_points} points. Not a pair number "
            )

        bipoints = []
        # Bipoints are stored in a flat "list"
        for k in range(nb_points // 2):
            # p1_coords = center_coords[2 * k]   dimension [.., .., .. , 8]
            # coordinate[2 * k][1], coordinate[2* k][0], coordinate[2 * k][2]
            # (x,y,z, 8)
            p1 = np.round(np.array(center_coords[2 * k][:3]) * self.scale)
            p2 = np.round(np.array(center_coords[2 * k + 1][:3]) * self.scale)

            # p1 = np.round(np.array(self.zxy_xyz(p1)) * self.scale)
            # p2 = np.round(np.array(self.zxy_xyz(p2)) * self.scale)

            bipoints.append((p1.astype(np.int16), p2.astype(np.int16)))

        return bipoints

    def zxy_yxz(self, pt):
        z, x, y = pt
        return y, x, z

    def _compute_target_shape(self, im_shape: tuple) -> tuple:
        """Compute the target shape using the scale parameter.

        Args:
            im_shape (tuple): The current image shape

        Returns:
            tuple: The new target image shape.
        """
        # Shape is : D0, D1, ..., T
        assert len(self.scale) == len(im_shape[:-1])
        new_shape = self.scale * im_shape[:-1]
        new_shape = new_shape.astype(np.int)

        return tuple(new_shape) + (im_shape[-1],)

    def set_augmentations(self, augmentations: any) -> None:
        """Set the augmentations to use for this dataset.

        Args:
            augmentations (any): The augmentations object from albumentation.
        """
        self._augmentations = augmentations

    def __next__(self) -> List[Tuple[np.ndarray, Tuple[np.float16, np.float16]]]:
        """Retrieve the next sample.

        Returns:
            Tuple[np.ndarray, List[Tuple[np.float16, np.float16]]]: The image and the list of bipoints in the image
        """
        sample = self.__getitem__(self.idx)
        self.idx = (self.idx + 1) % len(self)
        return sample

    def __iter__(self):
        """Return the instance object as iterator.

        Returns:
            AbstractCellDataset: The instance as iterator
        """
        return self

    def __len__(self) -> int:
        """Return the number of samples.

        Returns:
            int: The number of samples.
        """
        return min(len(self.images), self._sample_limit)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, List[Tuple[np.float16, np.float16]]]:
        """Return the sample at given index.

        Args:
            idx (int): The index of the sample.

        Returns:
            Tuple[np.ndarray, List[Tuple[np.float16, np.float16]]]: The image and the list of bipoints in the image
        """
        return self.images[idx], self.bipoints[idx]

    def get_output_types(self):
        """Output types for the Tensorflow DataLoader"""
        raise NotImplementedError

    def get_output_shapes(self):
        """Output shapes for the Tensorflow DataLoader"""
        raise NotImplementedError
