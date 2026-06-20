"""Plot prediction results"""
import matplotlib.pyplot as plt
import cv2
import math
import numpy as np
from dare2d.datamodule.post_processing.regression2d_pp import (
    convert_values,
)


def project_point(center, r, theta):
    """Project a point at given angle and distance and
    in both directions from the center."""
    center_x, center_y = center

    theta_rad = theta * np.pi / 180
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    vec_y, vec_x = r * cos_theta, r * sin_theta
    norm = math.sqrt(vec_x**2 + vec_y**2)
    norm = norm if norm > 0 else 1
    vec_x /= norm
    vec_y /= norm
    inverse_vec_x, inverse_vec_y = -vec_x, -vec_y

    half_size = r / 2

    p1 = (
        int(center_x + vec_x * half_size),
        int(center_y + vec_y * half_size),
    )
    p2 = (
        int(center_x + inverse_vec_x * half_size),
        int(center_y + inverse_vec_y * half_size),
    )

    return p1, p2


def display_length_angle(values, im_size, color=(255, 0, 0), thickness=5):
    results = []
    for k in range(values.shape[0]):
        result = np.zeros((im_size, im_size, 3))
        cx = im_size // 2
        cy = im_size // 2
        cnt_length = values[k, 0]
        angle_deg = values[k, 1]

        # Compute line to display
        p1, p2 = project_point(center=[cx, cy], r=cnt_length, theta=angle_deg)

        # Draw line
        cv2.line(result, p1, p2, color, 5)
        results.append(result)
    return np.asarray(results)


def plot_prediction(
    model,
    batch,
):
    """Plot the prediction given the model and a batch.

    Args:
        model (_type_): _description_
        batch (_type_): _description_
    """
    X, y = batch
    crop_size = X.shape[1]

    # Real values (ground truth)
    length_y = y["length_output"]
    angle_y = y["angle_output"]

    # Predicted values
    length_pred, angle_pred = model.predict(X)

    # print("Extracting groundtruth objects")
    gt_values = convert_values(length_y, angle_y, crop_size)
    gt_extracted = display_length_angle(gt_values, crop_size)

    # print("Extracting prediction objects")
    pred_values = convert_values(length_pred, angle_pred, crop_size)
    extracted = display_length_angle(pred_values, crop_size)

    for i in range(X.shape[0]):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        ax1.imshow(
            (gt_extracted[i, :, :] + X[i, :, :, 1] * 255),
            cmap="gray",
            alpha=0.8,
        )
        ax1.title("Ground truth mask:" + str(i))
        ax1.colorbar()

        ax2.imshow(
            (extracted[i, :, :] + X[i, :, :, 1] * 255),
            cmap="gray",
            alpha=0.8,
        )
        ax2.title("Extracted shape mask:" + str(i))
        ax2.colorbar()

        fig.show()
