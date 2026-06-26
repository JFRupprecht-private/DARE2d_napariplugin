import os
import shutil
from glob import glob
from pathlib import Path

import cc3d
import click
import numpy as np
from rich.pretty import pprint
from skimage import io
from tqdm import tqdm

TRACKING_SUMMARY = "man_track.txt"
TRA = "TRA"
SEG = "SEG"
PREFIX_SEG_FILE = "man_seg"


def parse_tracking_summary(summary_path):
    if not os.path.exists(summary_path):
        raise ValueError(
            f"Could not find tracking summary file: {summary_path}")

    lines = []
    with open(summary_path, 'r') as file:
        lines = file.readlines()

    if len(lines) == 0:
        raise ValueError(
            f"Expected tracking summary file to be non empty: {summary_path}")

    divisions = {}
    for line in lines:
        line = line.strip()
        cell_id, start_t, end_t, parent_id = [
            int(val) for val in line.split(" ")]
        # No parent means no division
        if parent_id == 0:
            continue
        if start_t not in divisions:
            divisions[start_t] = {}
        if parent_id not in divisions[start_t]:
            divisions[start_t][parent_id] = []
        divisions[start_t][parent_id].append(cell_id)

    for start_t in divisions.keys():
        for parent_id in divisions[start_t].keys():
            if len(divisions[start_t][parent_id]) != 2:
                divisions[start_t][parent_id] = []

    return divisions


def files_to_dict_id(files, prefix):
    return {int(Path(file).stem.split(prefix)[-1]): file for file in files}


def find_cell_centers(divisions, files, original=None):
    files = files_to_dict_id(files, PREFIX_SEG_FILE)

    frame_divisions = {}

    for frame_id in tqdm(divisions.keys()):
        # load segmentation image
        current_frame_path = files[frame_id]
        current_frame = io.imread(current_frame_path)

        frame_divisions[frame_id] = []
        for cell_ids in divisions[frame_id].values():
            centers = []
            if len(cell_ids) != 2:
                print("skip")
                continue

            for cell_id in cell_ids:
                filtered_image = np.where(current_frame == cell_id, 1, 0)

                # Do connected components labeling
                labels_out, n = cc3d.connected_components(
                    filtered_image, return_N=True)
                stats = cc3d.statistics(labels_out)
                centroids = stats["centroids"]
                center = centroids[-1]
                z, y, x = center
                center = [x, y, z]
                centers.append(center)
                frame_divisions[frame_id].append(center)

                # print(filtered_image.shape)
                # z, y, x = centroids[-1]
                # filtered_image[int(z), int(y), int(x)] = 2
                # io.imsave(f"{frame_id}.tif", filtered_image)

                # ref = io.imread(original[frame_id])
                # io.imsave(f"{frame_id}_original.tif", ref)
                # input()

            assert len(centers) == 2

    return frame_divisions


def read_write_img(path_in, path_out):
    shutil.copy(path_in, path_out)


def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


@click.command()
@click.option(
    "--gt",
    required=True,
    help="Path to the ground truth folder.",
)
@click.option(
    "--output",
    required=False,
    default="output",
    help="Path to the output folder. By default it is gt/output",
)
@click.option(
    "--original",
    required=False,
    default=None,
    help="Path to the original files.",
)
def main(gt: str, output, original=None):
    output_folder = output
    if output == "output":
        output_folder = os.path.join(gt, output)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    tracking_folder = os.path.join(gt, TRA)
    segmentation_folder = os.path.join(gt, SEG)

    if not os.path.exists(tracking_folder):
        raise ValueError(f"Could not find tracking folder: {tracking_folder}")
    if not os.path.exists(segmentation_folder):
        raise ValueError(
            f"Could not find segmentation folder: {segmentation_folder}")

    tracking_summary_file = os.path.join(tracking_folder, TRACKING_SUMMARY)
    divisions = parse_tracking_summary(tracking_summary_file)

    segmentation_files = glob(os.path.join(segmentation_folder, "*.tif"))

    previmg_folder = None
    currimg_folder = None
    nextimg_folder = None

    original_files = None
    if original is not None:
        previmg_folder = os.path.join(output_folder, "previmg")
        make_folder(previmg_folder)
        currimg_folder = os.path.join(output_folder, "currimg")
        make_folder(currimg_folder)
        nextimg_folder = os.path.join(output_folder, "nextimg")
        make_folder(nextimg_folder)
        curr_label_folder = os.path.join(output_folder, "currlabel")
        make_folder(curr_label_folder)
        original_files = glob(os.path.join(original, "*.tif"))
        original_files = files_to_dict_id(original_files, "t")

    divisions = find_cell_centers(
        divisions, segmentation_files, original_files)

    pprint(divisions)

    division_folder = os.path.join(output_folder, "div_location")
    make_folder(division_folder)

    segmentation_files = files_to_dict_id(segmentation_files, PREFIX_SEG_FILE)
    for frame_id in tqdm(segmentation_files.keys()):
        prev_id = frame_id - 1
        next_id = frame_id + 1
        frame_name = "%03d" % frame_id

        if prev_id not in segmentation_files or next_id not in segmentation_files:
            continue

        frame_divisions = []
        if frame_id in divisions:
            frame_divisions = divisions[frame_id]

        output_path = os.path.join(division_folder, frame_name + ".npy")
        np.save(output_path, np.array(frame_divisions))

        if original_files is not None:
            tiff_frame = frame_name + ".tif"
            read_write_img(original_files[prev_id], os.path.join(
                previmg_folder, tiff_frame))
            read_write_img(original_files[frame_id], os.path.join(
                currimg_folder, tiff_frame))
            read_write_img(original_files[next_id], os.path.join(
                nextimg_folder, tiff_frame))

            img_shape = io.imread(original_files[prev_id]).shape
            empty_img = np.zeros(img_shape)
            for i in range(len(frame_divisions)):
                p = frame_divisions[i]
                x, y, z = p
                empty_img[int(z), int(y), int(x)] = 255
            io.imsave(os.path.join(curr_label_folder, tiff_frame), empty_img)


if __name__ == "__main__":
    main()
