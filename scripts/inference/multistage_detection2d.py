#!/usr/bin/env python3
"""
Multistage 2D Cell Division Detection Inference Script

This script implements a two-stage pipeline for detecting cell divisions in 2D image sequences:
1. Segmentation stage: Identifies potential division centers using a U-Net-based model
2. Regression stage: Predicts division length and angle for each detected center

The pipeline processes 3D stacks (time series of 2D images) by analyzing triplets of consecutive
frames to capture temporal context for division detection.

Author: Romain and Marc

Usage:
    python -m scripts.inference.multistage_detection2d --regression <reg_model> --segmentation <seg_model> --img <input_image> --output <output_dir>

Arguments:
    --regression: Path to trained regression model checkpoint (.h5 file)
    --segmentation: Path to trained segmentation model checkpoint (.h5 file)
    --img: Path to input image stack (.tif format)
    --output: Directory to save inference results

Output:
    - division_position{i+1}.npy: Detected division centers for each frame
    - {image_name}_result.tiff: Visualization with division lines
    - {image_name}_raw.tiff: Raw segmentation predictions
    - {image_name}.gif: Animated results (optional)
"""

import click
import cv2
import hydra
import numpy as np
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from skimage import io
from tqdm import tqdm
import os
from pathlib import Path

from dare2d.datamodule.post_processing.regression2d_pp import convert_values
from dare2d.datamodule.visualization.regression2d_visualisation import (
    display_length_angle,
    project_point,
)
from dare2d.evaluation.center_metrics import extract_centers


def draw_objects_on_img(final_centers, currimg):
    """
    Visualize detected cell divisions on an image by drawing lines representing division axes.

    Args:
        final_centers (list): List of dictionaries containing division center information
                             Each dict has keys: 'x', 'y', 'length', 'angle'
        currimg (np.ndarray): Input image array (grayscale)

    Returns:
        tuple: (visualized_image, line_endpoints)
               - visualized_image: RGB image with division lines drawn
               - line_endpoints: List of [x1,y1,x2,y2] coordinates for line endpoints
    """
    viz = cv2.cvtColor(currimg, cv2.COLOR_GRAY2RGB)
    points = []
    for center in final_centers:
        cx, cy = center["x"], center["y"]
        cnt_length = center["length"]
        cnt_angle = center["angle"]

        # Compute line endpoints representing the division axis
        p1, p2 = project_point(center=[cx, cy], r=cnt_length, theta=cnt_angle)

        points.append([p1[1], p1[0]])
        points.append([p2[1], p2[0]])

        # Draw the division line on the visualization
        cv2.line(viz, p1, p2, (255, 0, 0), 5)
    return viz, points


def inference_strategy(x, seg_model, window_size=256):
    """
    Perform sliding window inference for segmentation using overlapping patches.

    This function divides the input image into overlapping patches, runs segmentation
    on each patch, and reconstructs the full-resolution prediction map. The sliding
    window approach ensures consistent predictions across patch boundaries.

    Args:
        x (np.ndarray): Input image stack of shape (H, W, 3) - three consecutive frames
        seg_model: Loaded segmentation model with .predict() method
        window_size (int): Size of sliding window patches (default: 256)

    Returns:
        np.ndarray: Segmentation prediction map of shape (H, W, 1) with values [0,1]
    """
    size = window_size
    stride = int(np.floor(size / 2))  # 50% overlap between patches

    # Extract sliding windows from the input image
    w = np.lib.stride_tricks.sliding_window_view(x, (size, size, 3))[
        ::stride, ::stride, 0
    ]

    row, col, h_size, w_size, n_chan = w.shape

    # Reshape for batch prediction
    w = np.reshape(w, (row * col, h_size, w_size, n_chan))
    w = seg_model.model.predict(w, verbose=0)
    w = np.reshape(w, (row, col, h_size, w_size, 1))

    # Initialize accumulation arrays for averaging overlapping predictions
    m = np.zeros(x.shape[:-1] + (1,), dtype=np.float32)  # Sum of predictions
    n = np.zeros(x.shape[:-1] + (1,), dtype=np.float32)  # Count of contributions

    # Reconstruct full image by averaging overlapping patch predictions
    for i in range(row):
        for j in range(col):
            y = i * stride
            cx = j * stride
            y_max = y + size
            cx_max = cx + size
            m[y:y_max, cx:cx_max] += w[i, j]
            n[y:y_max, cx:cx_max] += 1

    mean_pred = m / n  # Average predictions where patches overlap
    return mean_pred


def crop_img_from_center(img, center, half_crop_size):
    """
    Extract a square crop centered at a given position from an image.

    Args:
        img (np.ndarray): Input image array
        center (tuple): (y, x) coordinates of crop center
        half_crop_size (int): Half the size of the desired square crop

    Returns:
        np.ndarray: Cropped image patch
    """
    translat_half_crop = np.array([half_crop_size, half_crop_size])
    center_pad = (center + translat_half_crop).astype(int)
    start_x, start_y = center_pad - half_crop_size
    end_x, end_y = center_pad + half_crop_size
    crop = img[start_x:end_x, start_y:end_y, :]
    return crop


@click.command()
@click.option(
    "--regression",
    required=True,
    help="Path to the length/angle regression model.",
)
@click.option(
    "--segmentation",
    required=True,
    help="Path to the cell division center detection model.",
)
@click.option(
    "--img",
    required=True,
    help="Path to the input image.",
)
@click.option(
    "--output",
    required=True,
    help="Path to the output folder.",
)
def main(regression, segmentation, img, output):
    """
    Main inference function for 2D cell division detection.

    This function orchestrates the complete inference pipeline:
    1. Loads pre-trained regression and segmentation models
    2. Processes each frame in the input image stack
    3. Performs segmentation to detect potential division centers
    4. Applies regression to predict division parameters (length, angle)
    5. Generates visualizations and saves results

    Args:
        regression (str): Path to regression model checkpoint (.h5 file)
        segmentation (str): Path to segmentation model checkpoint (.h5 file)
        img (str): Path to input image stack (.tif format)
        output (str): Directory to save inference results
    """
    reg_model = None

    # Load regression model configuration and weights
    with initialize(version_base=None, config_path="../../config/"):
        reg_cfg = compose(
            config_name="train",
            overrides=["experiment=regression2d"],
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(reg_cfg)
        reg_cfg = OmegaConf.create(reg_cfg)
        reg_model = hydra.utils.instantiate(reg_cfg.model)
        reg_model.model.load_weights(regression)

    # Load segmentation model configuration and weights
    initialize(config_path="../../config/", job_name="center_detection_pred")
    seg_cfg = compose(config_name="train", overrides=["experiment=segmentation2d"])
    seg_cfg = OmegaConf.create(seg_cfg)

    # Create and load both models
    seg_model = hydra.utils.instantiate(seg_cfg.model)
    seg_model.model.load_weights(segmentation)

    # Load image
    stack_im = io.imread(img)

    inference_results = []

    print(f"loaded image size {stack_im.shape}")

    name = Path(img).stem

    output = os.path.join(output, name)
    if not os.path.exists(output):
        os.makedirs(output)

    io.imsave(os.path.join(output, f"{name}.tiff"), stack_im)

    segmentation_raws = []
    for i in tqdm(range(stack_im.shape[0])):
        if i == 0:
            previmg = stack_im[i]
        else:
            previmg = stack_im[i - 1]
        currimg = stack_im[i]
        if i == stack_im.shape[0] - 1:
            next_img = stack_im[i]
        else:
            next_img = stack_im[i + 1]

        previmg = cv2.equalizeHist(previmg)
        currimg = cv2.equalizeHist(currimg)
        next_img = cv2.equalizeHist(next_img)

        x = np.stack([previmg, currimg, next_img], axis=-1)

        x = (x / 255).astype(np.float32)

        segmentation_prediction_raw = inference_strategy(x, seg_model, window_size=256)

        # From prediction [0,1] to [0,255]
        segmentation_prediction = np.where(
            segmentation_prediction_raw > 0.5, 255, 0
        ).astype(np.uint8)
        detected_objects = extract_centers(segmentation_prediction)

        # Do the regression
        crop_size = 64
        half_crop_size = int(crop_size // 2)
        x = np.pad(
            x,
            (
                (half_crop_size, half_crop_size),
                (half_crop_size, half_crop_size),
                (0, 0),
            ),
            mode="constant",
            constant_values=0,
        )

        final_centers = []
        for center in detected_objects:
            inverted_center = (center[1], center[0])
            crop = crop_img_from_center(x, inverted_center, half_crop_size)
            length_pred, angle_pred = reg_model.model.predict(
                np.expand_dims(crop, axis=0), verbose=0
            )

            values = convert_values(length_pred, angle_pred, im_size=crop_size)

            final_centers.append(
                {
                    "x": center[0],
                    "y": center[1],
                    "angle": values[0, 1],
                    "length": values[0, 0],
                }
            )

        result_img, points = draw_objects_on_img(final_centers, next_img)
        inference_results.append(result_img)
        segmentation_raws.append(segmentation_prediction_raw)

        points = np.asarray(points)
        np.save(os.path.join(output, f"division_position{i+1}.npy"), final_centers)

    final_img = np.asarray(inference_results)
    segmentation_raws = np.repeat(np.asarray(segmentation_raws), 3, axis=-1)

    io.imsave(os.path.join(output, f"{name}_result.tiff"), final_img)
    io.imsave(os.path.join(output, f"{name}_raw.tiff"), segmentation_raws)

    try:
        fps = 1
        from moviepy.editor import ImageSequenceClip

        clip = ImageSequenceClip(list(final_img), fps=fps)
        clip.write_gif(os.path.join(output, f"{name}.gif"), fps=fps)
    except:
        pass


if __name__ == "__main__":
    

    main()
