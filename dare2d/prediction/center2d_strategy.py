import numpy as np


class Center2dInferenceStrategy(object):

    def __init__(self, window_size=256, overlap=0.5):
        self.window_size = window_size
        self.overlap = overlap

    def inference(self, x, model):
        size = self.window_size
        real_overlap = 1.0 - self.overlap
        stride = int(np.floor(size * real_overlap))

        # Cut the image into sliding windows
        w = np.lib.stride_tricks.sliding_window_view(
            x, (size, size, x.shape[-1]))[::stride, ::stride, 0]

        # Sliding window is divided into rows and cols
        row, col, h_size, w_size, n_chan = w.shape

        # We reshape the rows and cols of the sliding windows into
        # the batch size for inference
        w = np.reshape(w, (row * col, h_size, w_size, n_chan))
        w = model.predict(w, verbose=0)
        w = np.reshape(w, (row, col, h_size, w_size, 1))

        m = np.zeros(x.shape[:-1]+(1,), dtype=np.float32)
        n = np.zeros(x.shape[:-1]+(1,), dtype=np.float32)

        for i in range(row):
            for j in range(col):
                y = i * stride
                cx = j * stride
                y_max = y + size
                cx_max = cx + size
                m[y:y_max, cx:cx_max] += w[i, j]
                n[y:y_max, cx:cx_max] += 1

        mean_pred = m / n
        return mean_pred
