#!/bin/bash

shift=0

python convert_gt.py --img_path ../data/2d/raw/set_1/MAX_Image_2_pos1_1_celldivisionlevel.tiff --gt_folder ../data/2d/raw/set_1 --prefix=1 --shift=$shift
python convert_gt.py --img_path ../data/2d/raw/set_2/Image_3_t1-100-Z9_contrast_bis.tiff --gt_folder ../data/2d/raw/set_2 --prefix=2 --shift=$shift
python convert_gt.py --img_path ../data/2d/raw/set_3/Image_1-Z9_contrast.tiff --gt_folder ../data/2d/raw/set_3 --prefix=3 --shift=$shift
python convert_gt.py --img_path ../data/2d/raw/set_4/Image_2_3_4-pos2-1.tiff --gt_folder ../data/2d/raw/set_4 --prefix=4 --shift=$shift
python convert_gt.py --img_path ../data/2d/raw/set_5/Image_6_pos1-ant-z8.tiff --gt_folder ../data/2d/raw/set_5 --prefix=5 --shift=$shift
python convert_gt.py --img_path ../data/2d/raw/set_6/Image_6_pos2-post-z8.tiff --gt_folder ../data/2d/raw/set_6 --prefix=6 --shift=$shift
python convert_gt.py --img_path ../data/2d/raw/set_7/siractinE2_14-03-23_1_ant_z9-celldivisionlevel.tiff --gt_folder ../data/2d/raw/set_7 --prefix=7 --shift=$shift
python convert_gt.py --img_path ../data/2d/raw/set_8/siractinE2_14-03-23_1_post_z9-celldivisionlevel.tiff --gt_folder ../data/2d/raw/set_8 --prefix=8 --shift=$shift
