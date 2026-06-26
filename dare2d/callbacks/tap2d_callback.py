import os

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from skimage import io
from skimage.transform import resize

class Tap2DCallback(keras.callbacks.Callback):
    def __init__(self, log_dir="display", n_display=20, crop_size=32, layer_name="model_1"):
        super(Tap2DCallback, self).__init__()
        self.model = None
        self.model_handler = None
        self.epoch = 0
        self.generator = None
        self.mode = None
        self.n_display = n_display
        self.log_dir = log_dir
        self.crop_size = crop_size
        
        # Get intermediate output
        self.layer_name = layer_name
        self.extractor = None

    def check_extractor(self):
        if self.model is not None:
            self.extractor = tf.keras.models.Model(
                   [self.model.inputs], [self.model_handler.feature_model.output, self.model.output] 
            )

    def set_generator(self, generator):
        self.generator = generator
        self.epoch = 0

    def concat(self, var, nvar):
        if isinstance(var, dict):
            for key in var.keys():
                var[key] = np.concatenate(
                    [var[key], nvar[key]], axis=0)
        else:
            var = np.concatenate([var, nvar])
        return var

    def gather_full_set(self, generator):
        X, Y = None, None
        for nX, nY in generator:
            if X is None:
                X = nX
                Y = nY
            else:
                X = self.concat(X, nX)
                Y = self.concat(Y, nY)
        return X, Y

    def gradcam(self, output, grads):
        # Average gradients spatially
        # print(grads.shape)
        # weights = np.mean(grads, axis=(1, 2))
        # print(weights.shape)
        # # Build a ponderated map of filters according to gradients importance
        # cam = np.zeros(output.shape, dtype=np.float32)

        # for index, w in enumerate(weights):
        #     cam += w * output[..., index]

        cam = grads * output
        
        # Sum on temporal axis
        cam = np.sum(cam, axis=0)
        cam = np.sum(cam, axis=-1)
        
        capi=resize(cam,(self.crop_size,self.crop_size))
        capi = np.maximum(capi,0)
        print("Grad min-max: ", capi.min(), capi.max())
        heatmap = (capi - capi.min()) / (capi.max() - capi.min())
        return heatmap

    def batch_to_pred(self, X, Y):
        self.check_extractor()

        if X["input_a"].shape[0] > self.n_display:
            X["input_a"] = X["input_a"][:self.n_display]
            X["input_b"] = X["input_b"][:self.n_display]

        with tf.GradientTape() as tape:
            conv_outputs, output = self.extractor(X)

        grads = tape.gradient(output, conv_outputs).numpy()
        conv_outputs = conv_outputs.numpy()
        
        gradcams = []
        for i in range(X["input_a"].shape[0]):
            gradcams.append(self.gradcam(conv_outputs[i], grads[i]))
        
        return X, gradcams

    def write_batch(self, generator, epoch, name):
        X, Y = self.gather_full_set(generator)
        X, Y_pred = self.batch_to_pred(X, Y)

        # Create folder
        output_folder = os.path.join(self.log_dir, f"{epoch}")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Display images
        for i in range(min(X["input_a"].shape[0], self.n_display)):
            cx = np.concatenate([X["input_a"][i], X["input_b"][i]], axis=-1)
            self.display(cx, Y_pred[i], output_folder, i)

    def display(self, x, y_pred, base_folder, i):
        output_folder = os.path.join(base_folder, f"{i}")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if not isinstance(x, np.ndarray):
            x = x.numpy()

        previm = x[:, :, 0]
        currim = x[:, :, 1]
        # nextim = x[:, :, :, 2]

        io.imsave(os.path.join(output_folder, "previmg.tif"), previm)
        io.imsave(os.path.join(output_folder, "currimg.tif"), currim)
        # io.imsave(os.path.join(output_folder, "nextimg.tif"), nextim)

        # Display predicted gradcam
        io.imsave(os.path.join(output_folder, "grads.tif"), y_pred)

    def on_test_end(self, logs={}):
        self.write_batch(self.generator, self.epoch, self.mode)
        self.epoch += 1
