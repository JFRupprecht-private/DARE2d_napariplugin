import numpy as np


def convert_values(length: np.ndarray, angle: np.ndarray, im_size: int) -> np.ndarray:
    """Convert angle and length output from the model back to real values.

    Args:
        length (np.ndarray): shape: B float between 0-1
        angle (np.ndarray): shape: Bx(2) cos and sin values of twice the angle
        im_size (int): Size of the image used to unormalize the length

    Returns:
        np.ndarray: Bx2 which is (length, angle) angle is in degree
    """
    values = np.zeros((length.shape[0], 2), np.float32)
    for k in range(length.shape[0]):
        cnt_length = length[k] * im_size
        cos_cnt_angle, sin_cnt_angle = angle[k]
        angle_deg = np.arctan2(sin_cnt_angle, cos_cnt_angle) * 180 / np.pi
        angle_deg = angle_deg / 2.0
        values[k, 0] = cnt_length
        values[k, 1] = angle_deg

    return np.asarray(values)
