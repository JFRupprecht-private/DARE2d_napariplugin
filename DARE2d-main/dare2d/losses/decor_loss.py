import tensorflow as tf

class DecorrelationLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=0.2, scale=0.01, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.scale = scale

    def sub_call(self, x):
        # B, H x W, C
        x = tf.reshape(x, (-1, x.shape[1] * x.shape[2], x.shape[3]))
        x = tf.experimental.numpy.moveaxis(x, 1, -1)
        # B, C, H x W
        transpose = tf.transpose(x, perm=[0, 2, 1])
        dot = tf.linalg.matmul(x, transpose) / self.temperature
        logits = tf.nn.softmax(dot, axis=-1)
        diag = tf.linalg.diag_part(logits)
        loss = -tf.math.log(diag + 1e-10)
        # loss = tf.reduce_mean(loss)
        # return loss * self.scale
        return loss

    def call(self, y, x):
        a = x[:, 0, ...]
        b = x[:, 1, ...]
        return tf.reduce_mean(self.sub_call(a) + self.sub_call(b)) * self.scale
