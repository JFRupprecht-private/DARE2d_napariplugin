import tensorflow as tf
from tensorflow.keras import backend as K


def pixelwise_weighted_binary_crossentropy2(weight_scal):
    def pixelwise_crossentropy(y_true, y_pred):
        weight = (y_true * weight_scal) + 1

        epsilon = tf.constant(tf.keras.backend.epsilon(), dtype=y_pred.dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        y_pred = tf.math.log(y_pred / (1 - y_pred))

        zeros = tf.zeros_like(y_pred, dtype=y_pred.dtype)
        cond = y_pred >= zeros
        relu_logits = tf.where(cond, y_pred, zeros)
        neg_abs_logits = tf.where(cond, -y_pred, y_pred)

        entropy = relu_logits - y_pred * y_true + tf.math.log1p(tf.math.exp(neg_abs_logits))
        loss = tf.reduce_mean(weight * entropy, axis=-1)
        # return tf.reduce_sum(loss)
        return loss

    return pixelwise_crossentropy


def dice_coef(y_true, y_pred, smooth):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def dice_coef_loss(y_true, y_pred, smooth=1):
    return 1 - dice_coef(y_true, y_pred, smooth)


def weighted_dice_focal_loss(gamma=2.0, smooth=1.0):
    focal_loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=gamma, from_logits=False)

    def dice_focal(y_true, y_pred):
        dice = dice_coef_loss(y_true, y_pred, smooth)
        focal = focal_loss(y_true, y_pred)

        avg = (dice + focal) / 2
        return tf.reduce_mean(avg)

    return dice_focal


if __name__ == "__main__":
    import numpy as np
    from dare2d.datamodule.generator.seg_2dataset import SegmentationDataset

    data_folder = "/home/hcourtei/Projects//dare2d/data/train"  # '../data/3D_DivDetect/all_train_10.03.2023/train_10.03.23'
    ds = SegmentationDataset(data_folder, type_mask="circle", cell_size=25, renorm="min-max")
    X0, y0 = ds[0]
    X1, y1 = ds[1]
    y_true = np.array([y0, y1])
    y_true = y_true.reshape(y_true.shape[0], -1)
    y_pred = np.zeros_like(y_true)
    bx = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )

    for W in [1, 20, 50]:
        loss_func = pixelwise_weighted_binary_crossentropy2(W)
        L0 = loss_func(y_true, y_true)
        bc0 = bx(y_true, y_true)
        Lnull = loss_func(y_true, y_pred)
        bxnull = bx(y_true, y_pred)

        print(f"W={W}: L0 {L0}  bc0{bc0} Lnull{Lnull} bxnull{bxnull}")

    # cx = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    # bx = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    #
    # L2 = bx(y_true, y_pred)
