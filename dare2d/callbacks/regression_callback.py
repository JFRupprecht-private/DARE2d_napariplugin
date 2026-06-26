from dare2d.callbacks.display_callback import DisplayCallback
from dare2d.datamodule.visualization.regression2d_visualisation import \
    display_length_angle
from dare2d.datamodule.post_processing.regression2d_pp import convert_values
import tensorflow as tf
import numpy as np

class RegressionCallback(DisplayCallback):
    def __init__(self, log_dir):
        super(RegressionCallback, self).__init__(log_dir)

    def batch_to_pred(self, X, Y):
        crop_size = X.shape[1]

        # Real values (ground truth)
        length_y = Y["length_output"]
        angle_y = Y["angle_output"]

        # Predicted values
        length_pred, angle_pred = self.model.predict(X)

        gt_values = convert_values(length_y, angle_y, crop_size)
        Y_true = display_length_angle(gt_values, crop_size) / 255.0
        Y_true = np.expand_dims(Y_true, axis=-1)

        pred_values = convert_values(length_pred, angle_pred, crop_size)
        Y_pred = display_length_angle(pred_values, crop_size) / 255.0
        Y_pred = np.expand_dims(Y_pred, axis=-1)

        return X, Y_true, Y_pred
