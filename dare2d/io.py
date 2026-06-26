# Author: Romain and Marc

import os
from skimage.io import imread
import skimage.transform as trans
import numpy as np


def readreshape(filename, total_nb_channel, target_size):
    """
    Read image from disk and format it

    Parameters
    ----------
    filename : string
        Path to file. Only TIFF files
    total_nb_channel : int
        Total number of channel in the entry of the network
        3 for grey, 9 for RGB
    target_size : tuple of int
        Size to reshape the image.

    Returns
    -------
    im : numpy 2d array of floats

    """
    if type(target_size) is not tuple:
        target_size = (target_size, target_size)

    fext = os.path.splitext(filename)[1].lower()
    im = imread(filename, plugin="pil")
    im = trans.resize(im, target_size, anti_aliasing=True, order=1)
    if total_nb_channel == 3:
        im = np.reshape(im, im.shape + (1,))
    im = (im - np.min(im)) / np.ptp(im)
    return im


def find_rgb_generator(Directory_name):
    """
    Read image from disk and find if RGB or no

    Parameters
    ----------
    Directory_name : string
        Path to the directory that contain the training data

    Returns
    -------
    bool_ : bool

    """

    img_train_folder = os.path.join(Directory_name, "currimg/")
    image_name_arr = glob.glob(os.path.join(img_train_folder, "*.tif"))

    im = imread(image_name_arr[0])
    if len(im.shape) != 3:
        bool_ = False
    else:
        bool_ = True

    return bool_

import sys

class Stdout2file():

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass    
