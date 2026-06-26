import tensorflow
import numpy as np


def angle_degree_abs_mean_wrapper():
    return angle_degree_abs_mean


def angle_degree_abs_mean(y_true, y_pred):
    # Convert cos(2*theta), sin(2*theta) to angle in degree
    y_true_rad = tensorflow.experimental.numpy.arctan2(y_true[:, 1], y_true[:, 0])
    # y_true_rad = tensorflow.math.scalar_mul(0.5, y_true_rad)

    y_pred_rad = tensorflow.experimental.numpy.arctan2(y_pred[:, 1], y_pred[:, 0])
    # y_pred_rad = tensorflow.math.scalar_mul(0.5, y_pred_rad)

    # Compute absolute difference
    rad_diff = tensorflow.math.abs(y_true_rad - y_pred_rad)

    # Compute unsigned difference
    rad_diff = tensorflow.math.minimum((2 * np.pi) - rad_diff, rad_diff)

    # Convert difference to degrees
    deg_diff = rad_diff * 180 / np.pi

    # Compute the abs diff batch mean
    return tensorflow.reduce_mean(deg_diff, axis=-1)
