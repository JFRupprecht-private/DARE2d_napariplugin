#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: karnatmarc
"""

# from functions import selected_division_ground_truth
from functions_v2 import Annotator
import click

"""
This function will launch napari to create the training set.    	
Click where the division happen between the frames     	
Press "m" to pick a new image    	
Press "w" to save the image and the points positions
Works with RGB and Grey level .tif files
If you want to create an ellipse set, you have to create 
2 points for each divisions where the daughters cells are
"""


# img_path=
# gt_folder=
# frame=0

img_path="C:/Users/marck/OneDrive/Bureau/Legacy/Divisions_2D/Epithelium/Data/LiON_NICD_RFP_Siractin/SirAc_LiON-NICD-RFP-ant_t1-100_div-1.tif"
gt_folder="C:/Users/marck/OneDrive/Bureau/Legacy/Divisions_2D/Epithelium/Data/LiON_NICD_RFP_Siractin/SirAc_LiON_NICD_RFP_ant_t1_100_div_1/"

frame = "1"
# @click.command()
# @click.option(
#     "--img_path",
#     required=True,
#     help="Path to image to annotate.",
# )
# @click.option(
#     "--gt_folder",
#     required=True,
#     help="Path to the groundtruth folder to load from and save to.",
# )
# @click.option(
#     "--frame",
#     required=True,
#     default=0,
#     help="Current frame index in tiff",
# )
def main(img_path, gt_folder, frame):
    Annotator(img_path, gt_folder, frame)


if __name__ == "__main__":
    main(img_path, gt_folder, frame)
