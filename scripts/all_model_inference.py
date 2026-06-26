#!/usr/bin/env python3
"""
Batch Inference Runner for 2D Cell Division Detection

This script automates the inference process for 2D cell division detection across multiple images
and model configurations. It is designed for evaluating model performance on a dataset by running
inference with different trained model pairs (regression and segmentation models).

Author: qazi - 09/12/2025

Usage:
    python scripts/all_model_inference.py

Requirements:
    - Input images in .tif format placed in the INPUT_DIR
    - Pre-trained model checkpoints in REG_DIR and SEG_DIR
    - DARE2d project dependencies installed

The script performs the following steps:
    1. Scans the input directory for .tif images
    2. For each image, runs inference with 8 different model sets
    3. Saves structured outputs for each image-model combination
"""

import os
import subprocess
import glob

# Configuration: Modify these paths according to your project setup
# BASE_DIR should point to the root directory of the DARE2d project
BASE_DIR = r"path/to/your/project/root"  # Example: r"C:\Users\YourName\Projects\DARE2d"

# Define subdirectories relative to BASE_DIR
INPUT_DIR = os.path.join(BASE_DIR, "input")  # Directory containing input .tif images
OUTPUT_DIR = os.path.join(BASE_DIR, "output")  # Directory for saving inference results
REG_DIR = os.path.join(BASE_DIR, "regression_checkpoints")  # Directory with regression model checkpoints
SEG_DIR = os.path.join(BASE_DIR, "segmentation_checkpoints")  # Directory with segmentation model checkpoints

def main():
    """
    Main function to execute batch inference.

    This function:
    - Discovers all .tif images in the input directory
    - Iterates through each image and runs inference with 8 model sets
    - Uses subprocess to call the multistage_detection2d script for each combination
    """

    # Discover input images
    input_images = glob.glob(os.path.join(INPUT_DIR, "*.tif"))

    if not input_images:
        print("Error: No .tif images found in the input directory!")
        print(f"Input directory: {INPUT_DIR}")
        print("Please ensure input images are placed in the correct directory.")
        return

    print(f"Found {len(input_images)} input images for processing.")

    # Process each image
    for img_path in input_images:
        # Extract image name without extension for output organization
        image_name = os.path.splitext(os.path.basename(img_path))[0]

        print(f"\n{'='*50}")
        print(f"Processing image: {image_name}")
        print(f"{'='*50}")

        # Run inference with each of the 8 model sets
        for n in range(1, 9):  # Model sets numbered 1 through 8
            print(f"\n--- Running inference with model set {n} ---")

            # Construct paths to model checkpoints
            # Assumes checkpoints are organized as: checkpoints_set_{n}_all_but_target/best.h5
            reg_path = os.path.join(REG_DIR, f"checkpoints_set_{n}_all_but_target", "best.h5")
            seg_path = os.path.join(SEG_DIR, f"checkpoints_set_{n}_all_but_target", "best.h5")

            # Verify model paths exist
            if not os.path.exists(reg_path):
                print(f"Warning: Regression model not found: {reg_path}")
                continue
            if not os.path.exists(seg_path):
                print(f"Warning: Segmentation model not found: {seg_path}")
                continue

            # Create output directory for this image-model combination
            output_folder = os.path.join(OUTPUT_DIR, f"{image_name}_{n}")
            os.makedirs(output_folder, exist_ok=True)

            # Construct command for running inference
            # Calls the multistage_detection2d script with appropriate arguments
            cmd = [
                "python",
                "-m", "scripts.inference.multistage_detection2d",
                "--regression", reg_path,
                "--segmentation", seg_path,
                "--img", img_path,
                "--output", output_folder
            ]

            print(f"Executing: {' '.join(cmd)}")

            try:
                # Run the inference command
                subprocess.run(cmd, check=True)
                print(f"Successfully completed inference for {image_name} with model set {n}")
            except subprocess.CalledProcessError as e:
                print(f"Error running inference for {image_name} with model set {n}: {e}")
                continue

    print("\n" + "="*50)
    print("Batch inference completed for all images and model sets!")
    print(f"Results saved in: {OUTPUT_DIR}")
    print("="*50)

if __name__ == "__main__":
    main()
