import numpy as np
import matplotlib.pyplot as plt


def angle_error(length):
    v1, v2 = vectors_from_length(length)
    angle_v1 = np.arctan2(v1[1], v1[0]) * 180 / np.pi
    angle_v2 = np.arctan2(v2[1], v2[0]) * 180 / np.pi
    return abs(angle_v1 - angle_v2)


def vectors_from_length(length):
    # Origin is 0, 0

    # vector(P1,O)
    v1 = (length, 0)
    # vector(P2,0) with 1 pixel shift perpendicular to the vector direction
    v2 = (length, 1)
    return v1, v2


def draw_error_by_lengths(max_length):
    lengths = list(range(1, max_length + 1))
    errors = []
    for length in lengths:
        errors.append(angle_error(length))

    specific_lengths = [23, 35]
    specific_errors = [angle_error(length) for length in specific_lengths]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(lengths, errors)
    ax.scatter(
        specific_lengths,
        specific_errors,
    )
    for length, error in zip(specific_lengths, specific_errors):
        error = np.round(error, decimals=2)
        ax.annotate(str(f"{error}°"), xy=(length, error + 0.5))
    ax.set_title("Angle error by vector length")
    ax.set_ylabel("Angle error in degrees")
    ax.set_xlabel("Vector length in pixels")
    plt.xticks(list(range(0, 101, 10)) + specific_lengths)
    plt.savefig("angle_error_by_length.png")


draw_error_by_lengths(100)
