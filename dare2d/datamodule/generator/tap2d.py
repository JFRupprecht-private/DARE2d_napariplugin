import logging
import os

import glob
import random
from random import randrange
import numpy as np
import tensorflow as tf
import skimage.io as io
from tqdm import tqdm

log = logging.getLogger(__name__)


class Tap2d(tf.keras.utils.Sequence):
    def __init__(self, data_folder, crop_size, renorm=None):
        self.data_folder = data_folder
        self.crop_size = crop_size
        data_path = os.path.join(self.data_folder, "*.tif")
        movies = glob.glob(data_path)
        stacks = []
        for movie in movies:
            data = io.imread(movie)
            stacks.append(data)
        self.images = np.concatenate(stacks, axis=0)
        self.images = self.images.astype(np.float32)
        for i in tqdm(range(self.images.shape[0])):
            self.images[i] = (self.images[i] - self.images[i].min()) / self.images[i].ptp()                                
        print(self.images.max())

        self.idx = 0
        self._augmentations = None
        self.sample_limit = np.inf
            
    def pad_image(self, image, target_shape):
        current_shape = image.shape[:2]
        target_height, target_width = target_shape

        if current_shape[0] >= target_height and current_shape[1] >= target_width:
            return image

        pad_height = max(target_height - current_shape[0], 0)
        pad_width = max(target_width - current_shape[1], 0)

        padded_image = np.pad(image, [(0, pad_height), (0, pad_width), (0, 0)], mode='constant')
        return padded_image

    def set_sample_limit(self, sample_limit):
        self.sample_limit = sample_limit

    def __iter__(self):
        return self

    def __next__(self):
        sample = self.__getitem__(self.idx)
        self.idx = (self.idx + 1) % (len(self)-2)
        return sample

    def __len__(self):
        return len(self.images) - 2

    def random_crop(self, X, crop_size):
        if X.shape[0] <= crop_size and X.shape[1] <= crop_size:
            return X 
               
        y_diff = X.shape[0] - crop_size
        if y_diff == 0:
            y = 0
        else:
            y = randrange(y_diff)
            
        x_diff = X.shape[1] - crop_size
        if x_diff == 0:
            x = 0
        else:
            x = randrange(x_diff)
        
        # Random shift between the two channels
        max_shift_length = 5
        random_shift_x = randrange(max_shift_length)
        random_shift_y = randrange(max_shift_length)
        
        x_2 = min(max(0, x + random_shift_x), X.shape[1] - crop_size)
        y_2 = min(max(0, y + random_shift_y), X.shape[0] - crop_size)
        
        crop_a = X[y: y+crop_size, x: x+crop_size, 0]
        crop_b = X[y_2: y_2+crop_size, x_2: x_2+crop_size, 1]
        return np.stack([crop_a, crop_b], axis=-1)

    def augment_sample(self, a, b):
        augmented = self._augmentations(image=a, image2=b)
        na = augmented['image']
        nb = augmented['image2']
        return na, nb

    def __getitem__(self, idx):
        idx = idx % len(self.images)
        # Select 3 timestamps
        left = self.images[idx]
        center = self.images[idx+1]
        # right = self.images[idx+2]

        frames = []
        if random.random() > 0.5:
            class_value = [0.0, 1.0]
            # swap first and last
            # frames = [right, center, left]
            frames = [center, left]
        else:
            class_value = [1.0, 0.0]
            # frames = [left, center, right]
            frames = [left, center]

        frames = np.stack(frames, axis=-1)
        
        frames = self.pad_image(frames, (self.crop_size, self.crop_size))

        crop = np.zeros((self.crop_size, self.crop_size, 2))
        while np.sum(crop) == 0.0:
            crop = self.random_crop(frames, self.crop_size)

        a = np.expand_dims(crop[..., 0], axis=-1)
        b = np.expand_dims(crop[..., 1], axis=-1)
        
        if self._augmentations:
            a, b = self.augment_sample(a, b)

        return {"input_a": a, "input_b": b}, {"classification_head": np.array(class_value), "feature_head": np.array([0])}

    def set_augmentations(self, augmentations):
        self._augmentations = augmentations

    def get_output_types(self):
        return {"input_a": tf.float32, "input_b": tf.float32}, {"classification_head": tf.float32, "feature_head": tf.float32}

    def get_output_shapes(self):
        return {"input_a": (self.crop_size, self.crop_size, 1), "input_b": (self.crop_size, self.crop_size, 1)}, {"classification_head": (2,), "feature_head": (1,)}
