""""""

import tensorflow.keras as keras
from keras.losses import MeanSquaredError


class Regression2dCNN:
    def __init__(self, im_size, channels, optimizer, losses, metrics, n_stages=4, n_start_filters=16, evaluations=None) -> None:
        shape = (im_size, im_size, len(channels))
        input_layer = keras.layers.Input(shape=shape)
        x = input_layer

        for _ in range(n_stages):
            x = self.conv_block(x, n_start_filters)
            n_start_filters *= 2

        # Flatten cnn output
        x = keras.layers.Flatten()(x)

        # Create length regression
        len_x = keras.layers.Dense(1)(x)
        len_x = keras.layers.Activation("sigmoid", name="length_output")(len_x)

        # Create angle regression
        angle_x = keras.layers.Dense(2)(x)
        angle_x = keras.layers.Activation("tanh", name="angle_output")(angle_x)

        inputs = [input_layer]
        outputs = [len_x, angle_x]

        self.model = keras.models.Model(
            inputs=inputs, outputs=outputs, name="PropertiesNet")
        self.optimizer = optimizer

        # we need to convert dict config dict to real python dict
        self.losses = {}
        for key, v in losses.items():
            self.losses[key] = v

        self.metrics = {}
        for key, v in metrics.items():
            self.metrics[key] = v

        self.evaluations = evaluations

    def conv_block(self, x, n_filters, filter_size=3):
        """Simple convolution block with max pooling."""
        x = keras.layers.Conv2D(n_filters, filter_size)(x)
        x = keras.layers.Activation("relu")(x)
        # x = keras.layers.LayerNormalization()(x)
        x = keras.layers.MaxPooling2D()(x)
        return x

    def compile(self):
        """ """
        self.model.compile(optimizer=self.optimizer,
                           loss=self.losses, metrics=self.metrics)
