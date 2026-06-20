import tensorflow


def length_abs_mean_wrapper(crop_size):
    def abs_mean(y_true, y_pred):
        return crop_size * length_abs_mean(y_true, y_pred)

    return abs_mean


def length_abs_mean(y_true, y_pred):
    # Compute absolute difference
    length_diff = tensorflow.math.abs(y_true - y_pred)

    # Compute the abs diff batch mean
    return tensorflow.reduce_mean(length_diff, axis=-1)
