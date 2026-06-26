import logging
from pathlib import Path

import click
import cv2
import hydra
import numpy as np
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from skimage import io
from tqdm import tqdm

from dare2d.datamodule.post_processing.regression2d_pp import convert_values
from dare2d.datamodule.visualization.regression2d_visualisation import (
    display_length_angle,
    project_point,
)
from dare2d.evaluation.angle_degree import angle_degree_abs_mean
from dare2d.evaluation.evaluate_center2d import evaluate_center_detection
from dare2d.evaluation.length_regression import length_abs_mean

log = logging.getLogger()
log.setLevel(logging.INFO)


def evaluate_crop(pred, cosin, length, crop_size):
    pred_cosin = pred[1]
    pred_length = pred[0]
    cosin = np.expand_dims(cosin, axis=0)
    length = np.expand_dims(length, axis=0)
    cosin = cosin.astype(np.float64)
    pred_cosin = pred_cosin.astype(np.float64)

    # Divide by two as we predict 2theta but we want the distance between theta
    angle_error = angle_degree_abs_mean(cosin, pred_cosin) / 2.0
    length_error = length_abs_mean(length, pred_length)
    return angle_error.numpy(), length_error.numpy() * crop_size


def display_crop(crop, true_values, pred_values):
    crop_size = crop.shape[0]
    true_cvalues = convert_values(true_values[0], true_values[1], crop_size)
    pred_cvalues = convert_values(pred_values[0], pred_values[1], crop_size)

    true_mask = display_length_angle(true_cvalues, crop_size, color=(0, 255, 0), thickness=3)[0]
    pred_mask = display_length_angle(pred_cvalues, crop_size, color=(255, 0, 0), thickness=3)[0]

    x = crop[:, :, -1] * 255
    x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB).astype(np.uint8)

    x = cv2.addWeighted(x, 1.0, true_mask.astype(np.uint8), 0.8, 0)
    x = cv2.addWeighted(x, 1.0, pred_mask.astype(np.uint8), 0.8, 0)

    return x


def fig_to_img(fig):
    canvas = fig.canvas
    canvas.draw()  # Draw the canvas, cache the renderer
    image_flat = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")  # (H * W * 3,)
    # NOTE: reversed converts (W, H) from get_width_height to (H, W)
    image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)  # (H, W, 3)
    return image


def angles_lengths_from_bipoints(bipoints, regression_generator):
    cosin_angles, lengths = [], []
    for bipoint in bipoints:
        length, cosin = regression_generator.distance_angle_from_bipoint(bipoint)
        cosin_angles.append(cosin)
        lengths.append(length)
    return cosin_angles, lengths


def single_line(mask, center, cvalues):
    cy, cx = center
    p1, p2 = project_point(center=[cx, cy], r=cvalues[0], theta=cvalues[1])
    cv2.line(mask, p1, p2, 255, 2)
    return mask


def display_angle_on_mask(mask, pred, crop_size):
    pred_cvalues = convert_values(np.array([pred["length"]]), np.array(pred["cosin"]), crop_size)[0]
    single_line(mask, pred["center"], pred_cvalues)
    return mask


@click.command()
@click.option(
    "--regression_weights",
    required=True,
    help="Path to the length/angle regression model.",
)
@click.option(
    "--segmentation_weights",
    required=True,
    help="Path to the cell division center detection model.",
)
@click.option("--test_set", required=False, default=None, help="Folder path to the test set.")
@click.option(
    "--display",
    required=False,
    default=False,
    help="Display each segmentation prediction",
)
@click.option(
    "--output_scores",
    required=False,
    default="scores.json",
    help="Path to the file that will contains the scores",
)
@click.option("--inputs", required=False, default="[-1, 0, 1]")
def main(regression_weights, segmentation_weights, test_set, display, inputs, output_scores):
    reg_datamodule = None
    reg_train = None
    reg_model = None

    # Load regression config
    with initialize(version_base=None, config_path="../../config/"):
        reg_cfg = compose(
            config_name="train",
            overrides=["experiment=regression2d", f"input_channels={inputs}"],
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(reg_cfg)
        reg_cfg = OmegaConf.create(reg_cfg)

        reg_train = hydra.utils.instantiate(reg_cfg.datamodule.train)
        # reg_train = reg_datamodule._train_generator
        reg_model = hydra.utils.instantiate(reg_cfg.model)
        reg_model.model.load_weights(regression_weights)

    # Load center detection config
    initialize(config_path="../../config/", job_name="center_detection_eval")
    seg_overrides = ["experiment=segmentation2d", f"input_channels={inputs}"]
    if test_set:
        seg_overrides += [f"datamodule.test.data_folder={test_set}"]
    seg_cfg = compose(config_name="train", overrides=seg_overrides)
    seg_cfg = OmegaConf.create(seg_cfg)

    # Create and load both models
    seg_model = hydra.utils.instantiate(seg_cfg.model)
    seg_model.model.load_weights(segmentation_weights)

    # Create dataloader from center detection
    test_set = hydra.utils.instantiate(seg_cfg.datamodule.test)

    # On the test set
    # test_set = seg_datamodule._test_generator
    max_distance = 10

    # Make prediction on the whole set
    # Retrieve the centers
    stats, centers_matched = evaluate_center_detection(
        seg_model.model, test_set, max_distance=max_distance, display=display
    )

    print(
        f"Recall {100*stats['recall']:02f} % ({stats['true_center_matched']} / {stats['true_center']}) with #{stats['true_matched_other_time']} matched at another time"
    )
    print(
        f"Precision {100*stats['precision']:02f} % ({stats['pred_center_matched']} / {stats['pred_center']}) with #{stats['pred_matched_other_time']} matched at another time"
    )
    print(f"Fmeasure {100*stats['fmeasure']:02f} %")

    true_centers_matched, pred_centers_matched = centers_matched

    true_angle_errors, true_length_errors = [], []
    pred_angle_errors, pred_length_errors = [], []

    print("Evaluating regression on true and predicted crops...")
    crops = []
    masks = []
    for i in tqdm(range(len(test_set))):
        x = test_set.images[i]
        mask = np.zeros((x.shape[0], x.shape[1], 1))
        x = reg_train.pad_img(x)
        bipoints = test_set.bipoints[i]
        cosin_angles, lengths = angles_lengths_from_bipoints(bipoints, reg_train)
        true_centers = true_centers_matched[i]

        for j, (true_center, matches) in enumerate(true_centers.items()):
            for match in matches:
                # Don't compare orientation at two different detection time
                if match["t"] != i:
                    continue

                if true_center[0] < 0 or true_center[1] < 0:
                    continue

                pred_center = match["position"]
                true_crop = reg_train.crop_img_from_center(x, true_center)
                pred_crop = reg_train.crop_img_from_center(x, pred_center)

                cosin = cosin_angles[j]
                length = lengths[j]

                # Evaluate regression on crop_gt
                true_y_pred = reg_model.model.predict(np.expand_dims(true_crop, axis=0), verbose=0)
                true_angle_error, true_length_error = evaluate_crop(
                    true_y_pred, cosin, length, true_crop.shape[0]
                )

                true_angle_errors.append(true_angle_error)
                true_length_errors.append(true_length_error)

                # Evaluate regression on crop_pred
                pred_y_pred = reg_model.model.predict(np.expand_dims(pred_crop, axis=0), verbose=0)
                pred_angle_error, pred_length_error = evaluate_crop(
                    pred_y_pred, cosin, length, true_crop.shape[0]
                )

                mask = display_angle_on_mask(
                    mask,
                    {"center": pred_center, "cosin": pred_y_pred[1], "length": pred_y_pred[0]},
                    true_crop.shape[0],
                )

                # if display:
                #     crop_viz_true_crop = display_crop(
                #         true_crop,
                #         true_values=[np.array([length]), np.array([cosin])],
                #         pred_values=true_y_pred,
                #     )
                #     crop_viz_pred_crop = display_crop(
                #         pred_crop,
                #         true_values=[np.array([length]), np.array([cosin])],
                #         pred_values=pred_y_pred,
                #     )

                #     import matplotlib.pyplot as plt

                #     fig = plt.figure(constrained_layout=True)
                #     fig.suptitle("Regression results")
                #     subfig1, subfig2 = fig.subfigures(nrows=2, ncols=1)
                #     ax1 = subfig1.subplots(nrows=1, ncols=3)
                #     ax2 = subfig2.subplots(nrows=1, ncols=3)
                #     subfig1.suptitle(
                #         f"True crop: angle error {np.round(true_angle_error, decimals=2)}°, length error {np.round(true_length_error[0], decimals=2)} px"
                #     )
                #     subfig2.suptitle(
                #         f"Pred crop: angle error {np.round(pred_angle_error, decimals=2)}°, length error {np.round(pred_length_error[0], decimals=2)} px"
                #     )

                #     ax1[0].imshow(
                #         cv2.cvtColor(true_crop[..., 0] * 255, cv2.COLOR_GRAY2RGB).astype(np.uint8)
                #     )
                #     ax1[0].set_title("t-2")
                #     ax1[1].set_title("t-1")
                #     ax1[2].set_title("t")
                #     ax1[0].set_axis_off()
                #     ax1[1].set_axis_off()
                #     ax1[2].set_axis_off()
                #     ax1[1].imshow(
                #         cv2.cvtColor(true_crop[..., 1] * 255, cv2.COLOR_GRAY2RGB).astype(np.uint8)
                #     )
                #     ax1[2].imshow(crop_viz_true_crop)

                #     ax2[0].set_axis_off()
                #     ax2[1].set_axis_off()
                #     ax2[2].set_axis_off()
                #     ax2[0].set_title("t-2")
                #     ax2[1].set_title("t-1")
                #     ax2[2].set_title("t")
                #     ax2[0].imshow(
                #         cv2.cvtColor(pred_crop[..., 0] * 255, cv2.COLOR_GRAY2RGB).astype(np.uint8)
                #     )
                #     ax2[1].imshow(
                #         cv2.cvtColor(pred_crop[..., 1] * 255, cv2.COLOR_GRAY2RGB).astype(np.uint8)
                #     )
                #     ax2[2].imshow(crop_viz_pred_crop)
                #     crops.append((fig_to_img(fig), i))
                #     plt.close()

                pred_angle_errors.append(pred_angle_error)
                pred_length_errors.append(pred_length_error)

        masks.append(mask[..., -1])

    io.imsave(f"logs/regression.tif", np.array(masks).astype(np.uint8))
    crops_data = zip(crops, pred_angle_errors, pred_length_errors)
    crops_data = sorted(crops_data, key=lambda x: x[1], reverse=True)

    for i in range(min(50, len(crops))):
        crop_idx, angle_error, length_error = crops_data[i]
        crop, idx = crop_idx
        io.imsave(
            f"logs/crop_error_top_{i}_index_{idx}.png",
            crop.astype(np.uint8),
        )

    mean_true_angle_error, std_true_angle_error = np.mean(true_angle_errors), np.std(
        true_angle_errors
    )
    mean_true_length_error, std_true_length_error = np.mean(true_length_errors), np.std(
        true_length_errors
    )
    mean_pred_angle_error, std_pred_angle_error = np.mean(pred_angle_errors), np.std(
        pred_angle_errors
    )
    mean_pred_length_error, std_pred_length_error = np.mean(pred_length_errors), np.std(
        pred_length_errors
    )

    print(
        f"Mean regression error from groundtruth centers. Angle error {mean_true_angle_error}° +/- {std_true_angle_error} ; length error {mean_true_length_error} px +/- {std_true_length_error}"
    )
    print(
        f"Mean regression error from predicted centers. Angle error {mean_pred_angle_error}° +/- {std_pred_angle_error}; length error {mean_pred_length_error} px +/- {std_pred_length_error}"
    )

    # stats["true_angle_error"] = mean_true_angle_error
    # stats["pred_angle_error"] = mean_pred_angle_error
    # stats[""]

    # set_name = Path(test_set).stem
    # with open(output_scores, "w+") as file:
    #     import json

    #     file.write(json.dumps({set_name: stats}))


if __name__ == "__main__":
    main()
