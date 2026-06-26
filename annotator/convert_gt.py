import os
from glob import glob
from pathlib import Path

import click
import numpy as np
from skimage import io
from tqdm import tqdm


def get_all_annotated_frames(folder):
    annotated_frames = list(glob(os.path.join(folder, "division_position*.npy")))
    annotated_frames = [
        int(Path(division_position).stem.split("division_position")[-1])
        for division_position in annotated_frames
    ]
    annotated_frames.sort()
    return annotated_frames


def create_folder(folder_path):
    try:
        os.mkdir(folder_path)
    except:
        pass


def point_in_range(point, x_min, x_max, y_min, y_max):
    x, y = point
    return x_min <= x and x <= x_max and y_min <= y and y <= y_max


def shift_points(point, x_origin, y_origin):
    x, y = point
    return (x - x_origin, y - y_origin)


def get_center(p1, p2):
    return int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)


def image_to_crops(x, bipoints, size):
    stride = size

    w = np.lib.stride_tricks.sliding_window_view(x, (size, size, 3))[
        ::stride, ::stride, 0
    ]

    row, col, h_size, w_size, n_chan = w.shape

    crops_with_points = []
    for i in range(row):
        for j in range(col):
            y = i * stride
            cx = j * stride
            y_max = y + size
            cx_max = cx + size

            crop = w[i, j]

            points = []
            for p1, p2 in bipoints:
                if point_in_range(p1, y, y_max, cx, cx_max) or point_in_range(
                    p2,
                    y,
                    y_max,
                    cx,
                    cx_max,
                ):
                    center = get_center(p1, p2)
                    if point_in_range(center, y, y_max, cx, cx_max):
                        new_p1 = shift_points(p1, y, cx)
                        new_p2 = shift_points(p2, y, cx)
                        print(
                            f"point original P1: {p1} vs new point {new_p1} in range x[{cx}, {cx_max}] & y[{y}, {y_max}]"
                        )
                        print(
                            f"point original P2: {p2} vs new point {new_p2} in range x[{cx}, {cx_max}] & y[{y}, {y_max}]"
                        )
                        points.append((new_p1, new_p2))

            crops_with_points.append((crop, points))

    return crops_with_points


def save_data(
    output_folder,
    index,
    images,
    positions,
    prefix
):
    im_name = f"{index}"
    if prefix:
        im_name = f"{prefix}_{im_name}"
    io.imsave(os.path.join(output_folder, "previmg", f"{im_name}.tif"), images[0])
    io.imsave(os.path.join(output_folder, "currimg", f"{im_name}.tif"), images[1])
    io.imsave(os.path.join(output_folder, "nextimg", f"{im_name}.tif"), images[2])

    # Add points
    division_position = np.array([[div[0], div[1]] for div in positions])
    np.save(
        os.path.join(output_folder, "div_location", f"{im_name}.npy"), division_position
    )


@click.command()
@click.option(
    "--img_path",
    required=True,
    help="Path to image to annotate.",
)
@click.option(
    "--gt_folder",
    required=True,
    help="Path to the groundtruth folder to load from and save to.",
)
@click.option(
    "--crop_size",
    required=False,
    default=0,
    help="Size of the crop to use. If crop_size = 0 then no crops are performed.",
)
@click.option(
    "--prefix",
    required=False,
    default=None,
    help="Prefix image names."
)
@click.option(
    "--shift",
    required=False,
    default=-1,
    help="Shift temporal frames by shift amount such that frames = frames[t-1+shift, t+shift, t+1+shift]"
)
def main(img_path, gt_folder, crop_size, prefix, shift):

    stack_im = io.imread(img_path)
    frames = get_all_annotated_frames(gt_folder)

    output_folder = gt_folder
    use_crop = crop_size > 0
    if use_crop:
        output_folder = os.path.join(output_folder, f"crop_{crop_size}")
        create_folder(output_folder)

    create_folder(os.path.join(output_folder, "div_location"))
    create_folder(os.path.join(output_folder, "previmg"))
    create_folder(os.path.join(output_folder, "currimg"))
    create_folder(os.path.join(output_folder, "nextimg"))

    index = 0

    print(f"Processing set {gt_folder}")
    for i, frame in tqdm(enumerate(frames)):
        prev_index = frame - 1 + shift
        curr_index = frame + shift
        next_index = frame + shift + 1
        
        if prev_index < 0 or next_index >= len(frames):
            continue
        
        prev_im, im, next_im = (
            stack_im[prev_index],
            stack_im[curr_index],
            stack_im[next_index],
        )

        division_position = list(
            np.load(os.path.join(gt_folder, f"division_position{frame}.npy"))
        )
        # filter third component which is num frame
        division_position = np.array([[div[0], div[1]] for div in division_position])

        if use_crop:
            x = np.stack([prev_im, im, next_im], axis=-1)
            n_bipoints = len(division_position) // 2
            bipoints = [
                (division_position[i * 2], division_position[i * 2 + 1])
                for i in range(n_bipoints)
            ]
            crops_with_points = image_to_crops(x, bipoints, crop_size)
            for crop, crop_bipoints in crops_with_points:
                crop_prev_im = crop[:, :, 0]
                crop_im = crop[:, :, 1]
                crop_next_im = crop[:, :, 2]
                crop_div_position = [
                    point for points in crop_bipoints for point in points
                ]

                save_data(
                    output_folder=output_folder,
                    index=index,
                    images=[crop_prev_im, crop_im, crop_next_im],
                    positions=crop_div_position,
                    prefix=prefix
                )
                index += 1
        else:
            save_data(
                output_folder=output_folder,
                index=i,
                images=[prev_im, im, next_im],
                positions=division_position,
                prefix=prefix
            )


if __name__ == "__main__":
    main()
