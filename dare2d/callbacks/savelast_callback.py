import os

from tensorflow import keras


class SaveLastModel(keras.callbacks.Callback):
    def __init__(self):
        super(keras.callbacks.Callback, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        self.model.save(os.path.join(
            'checkpoints', 'last.hdf5'), overwrite=True)
