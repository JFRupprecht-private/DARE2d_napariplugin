#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: karnatmarc
"""

from functions_v2 import Annotator

"""
This function will launch napari to create the training set.
Click where the division happen between the frames
Press "m" to pick a new image
Press "w" to save the image and the points positions
Works with RGB and Grey level .tif files
If you want to create an ellipse set, you have to create
2 points for each divisions where the daughters cells are
"""

# Set these to the image to annotate and the ground-truth folder to read/write.
img_path = ""
gt_folder = ""
frame = "1"


def main(img_path, gt_folder, frame):
    Annotator(img_path, gt_folder, frame)


if __name__ == "__main__":
    main(img_path, gt_folder, frame)
