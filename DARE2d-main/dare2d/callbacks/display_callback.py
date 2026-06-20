import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class DisplayCallback(keras.callbacks.Callback):
    def __init__(self, log_dir: str, n_display: int = 10):
        super(DisplayCallback, self).__init__()
        self.model = None
        self.epoch = 0
        self.generator = None
        self.mode = None
        self.n_display = n_display
        self.writer = tf.summary.create_file_writer(log_dir)

    def set_generator(self, generator):
        self.generator = generator
        self.epoch = 0

    def gather_full_set(self, generator):
        X, Y = None, None
        for nX, nY in generator:
            if X is None:
                X = nX
                Y = nY
            else:
                X = np.concatenate([X, nX], axis=0)
                if isinstance(Y, dict):
                    for key in Y.keys():
                        Y[key] = np.concatenate([Y[key], nY[key]], axis=0)
                else:
                    Y = np.concatenate([Y, nY])
        return X, Y

    def batch_to_pred(self, X, Y):
        Y_pred = self.model.predict(X)
        if len(Y.shape) == 3:
            Y = np.expand_dims(Y, axis=-1)
        return X, Y, Y_pred

    def write_batch(self, generator, epoch, name):
        X, Y = self.gather_full_set(generator)
        X, Y_true, Y_pred = self.batch_to_pred(X, Y)

        # Get current plt backend
        original_backend = plt.get_backend()

        # Switch
        plt.switch_backend("agg")

        figs = self.draw(X, Y_true, Y_pred)
        self.plot_to_tensorboard(name, self.writer, figs, epoch)

        # Go back to original plt backend
        plt.switch_backend(original_backend)

    def on_test_end(self, logs={}):
        self.write_batch(self.generator, self.epoch, self.mode)
        self.epoch += 1

    def plot_to_tensorboard(self, image_name, writer, figs, step):
        """
        Takes a matplotlib figure handle and converts it using
        canvas and string-casts to a numpy array that can be
        visualized in TensorBoard using the add_image function

        Parameters:
            writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
            fig (matplotlib.pyplot.fig): Matplotlib figure handle.
            step (int): counter usually specifying steps/epochs/time.
        """

        imgs = []
        for fig in figs:
            # Draw figure on canvas
            fig.canvas.draw()

            # Convert the figure to numpy array, read the pixel values and reshape the array
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (img.shape[-1],))

            # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
            img = img / 255.0
            plt.close(fig)
            imgs.append(img)

        imgs = np.asarray(imgs)
        # Add figure in numpy "image" to TensorBoard writer
        with writer.as_default():
            tf.summary.image(image_name, imgs, step, max_outputs=10)

    def draw(self, X, Y_true, Y_pred):
        """Draw plots."""
        figs = []
        for idx in range(min(X.shape[0], self.n_display)):
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
            current_Y_true = Y_true[idx, :, :, 0] * 255
            current_X = X[idx, :, :, -1] * 255
            current_Y_pred = Y_pred[idx, :, :, 0] * 255

            ax1.imshow(
                (current_Y_true + current_X),
                cmap="gray",
                alpha=0.8,
            )
            ax1.set_title(f"Ground truth mask overlay: {idx}")

            ax2.imshow(
                (current_Y_pred + current_X),
                cmap="gray",
                alpha=0.8,
            )
            ax2.set_title(f"Predicted mask overlay: {idx}")

            ax3.imshow(current_Y_true, cmap="gray")
            ax3.set_title(f"Ground truth: {idx}")

            ax4.imshow(current_Y_pred, cmap="gray")
            ax4.set_title(f"Prediction: {idx}")

            figs.append(fig)
        return figs
