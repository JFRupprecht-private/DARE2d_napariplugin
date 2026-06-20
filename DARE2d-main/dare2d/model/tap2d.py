# fmt: off
import os

import tensorflow.keras as keras
from omegaconf import DictConfig
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm
# fmt: on
import tensorflow as tf


# Define backbone model
class Tap2D:
    def __init__(self, target_size, optimizer, losses, metrics, evaluations=None, features=32) -> None:
        inputs_a = keras.layers.Input(shape=(target_size, target_size, 1), name="input_a")
        inputs_b = keras.layers.Input(shape=(target_size, target_size, 1), name="input_b")
        
        self.backbone = self.create_backbone_model(target_size)
        self.feature_head = ProjectionHead(features)
        self.classification_head = ClassificationHead()

        # Both images go through the same backbone
        features_a = self.backbone(inputs_a)
        features_b = self.backbone(inputs_b)
        
        # Concatenate all features

        # Project features to the final feature space
        features_a = self.feature_head(features_a)
        features_b = self.feature_head(features_b)

        merged_features = StackLayer(axis=1, name="feature_head")([features_a, features_b])
        print("Merged features: ", merged_features)

        # Classify the final features
        result = self.classification_head(merged_features)

        self.model = Model(inputs=[inputs_a, inputs_b], outputs=[merged_features, result], name="tap_model")

        self.feature_model = Model(inputs=[inputs_a, inputs_b], outputs=merged_features, name="feature_model")

        self.optimizer = optimizer

        # we need to convert dict config dict to real python dict
        self.losses = self.dictConfig2dict(losses)
        print(self.losses)
        self.metrics = self.dictConfig2dict(metrics)
        self.evaluations = self.dictConfig2dict(evaluations)

    def create_backbone_model(self, size):
        # input_ = keras.layers.Input(shape=(size, size, 1))
        # x = input_
        # start_filters=8
        # n_stages = 5
        # # Encode
        # for i in range(n_stages):
        #     x = self.conv_block(x, start_filters)
        #     start_filters *= 2
            
        # # Decode
        # for i in range(n_stages):
        #     x = self.deconv_block(x, start_filters)
        #     start_filters = start_filters // 2
        
        # x = keras.layers.Conv2D(32, 3, padding="same")(x)
        # model = Model(inputs=input_, outputs=x, name="backbone")

        model = sm.Unet(
            backbone_name="resnet18",
            input_shape=(size, size, 1),
            classes=32,
            activation="linear",
            encoder_weights=None
        )
        
        return model
    
    def deconv_block(self, x, n_filters, filter_size=3):
        x = keras.layers.UpSampling2D(size=2)(x)
        x = keras.layers.Conv2D(n_filters, filter_size,
                                activation="relu", padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        return x

    def conv_block(self, x, n_filters, filter_size=3):
        """Simple convolution block with max pooling."""
        x = keras.layers.Conv2D(n_filters, filter_size,
                                activation="relu", padding="same")(x)
        x = keras.layers.MaxPooling2D()(x)

        fx = keras.layers.Conv2D(
            n_filters, filter_size, activation="relu", padding="same")(x)
        fx = keras.layers.Conv2D(n_filters, filter_size, padding="same")(fx)

        x = keras.layers.Add()([x, fx])
        x = keras.layers.ReLU()(x)
        x = keras.layers.BatchNormalization()(x)
        return x

    def dictConfig2dict(self, dc):
        new_dict = dc
        if isinstance(dc, DictConfig):
            new_dict = {}
            for key, v in dc.items():
                new_dict[key] = v
        return new_dict

    def compile(self):
        """ """
        self.model.compile(optimizer=self.optimizer,
                           loss=self.losses) # , metrics=self.metrics)
        self.model.summary()

class ClassificationHead(tf.keras.Model):
    def __init__(self):
        super(ClassificationHead, self).__init__()
        
        self._block1 = PermEquivariantBlock(out_features=32, norm=True, norm_before_act=True)
        self._block2 = PermEquivariantBlock(out_features=32, norm=True, norm_before_act=True)
        
        self._average_pooling = MeanLayer(axis=[-2, -3], keepdims=True)
        self._block3 = PermEquivariantBlock(out_features=1, norm=False, activation=Identity)
        self._flatten = Flatten()
        # self._classif = Dense(1, use_bias=False)

    def _layers(self, x):
        x = self._block1(x)
        x = self._block2(x)
        return x

    def _head(self, x):
        x = self._average_pooling(x)
        x = self._block3(x)
        x = self._flatten(x)
        return x        
        
    def call(self, inputs):
        x = self._layers(inputs)
        x = self._head(x)
        # x = self._classif(x)
        return x

class ProjectionHead(tf.keras.Model):
    def __init__(self, out_features):
        super(ProjectionHead, self).__init__()

        self.out_features = out_features
        self._conv = keras.layers.Conv2D(out_features, kernel_size=1, padding="same")
        self._bn = keras.layers.BatchNormalization()

        self.features = {}

    def _layers(self, x):
        x = self._conv(x)
        x = self._bn(x)
        return x

    def call(self, inputs):
        x = inputs
        x = self._layers(x)
        return x

class Identity(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        
    def call(self, inputs):
        return inputs

class MeanLayer(keras.layers.Layer):
    def __init__(self, axis=0, keepdims=True):
        super().__init__()
        self.axis=axis
        self.keepdims=keepdims
        
    def call(self, inputs):
        return tf.math.reduce_mean(inputs, self.axis, self.keepdims)

class StackLayer(keras.layers.Layer):
    def __init__(self, axis=0, name=None):
        super().__init__(name=name)
        self.axis = axis
        
    def call(self, inputs):
        return tf.stack(inputs, self.axis)
        
class TemporalEquivariantBlock(keras.layers.Layer):
    def __init__(self, axis=1, out_features=1, **linear_kwargs):
        super().__init__()
        self.axis = axis
        self.G = Dense(out_features, **linear_kwargs)
        
    def call(self, inputs):
        return tf.math.reduce_max(self.G(inputs), axis=-2, keepdims=True)[0]

        
# class BasicPermEquivariantBlock(keras.layers.Layer):
#     def __init__(self,
#         out_features: int = 1,
#         norm: bool = False,
#         activation= keras.layers.LeakyReLU,
#         norm_before_act: bool = False,
#         **linear_kwargs
#     ):
#         super().__init__()
#         self.out_features = out_features
        
#         self.L = Dense(out_features, **linear_kwargs)
#         self.G = TemporalEquivariantBlock(axis=1, out_features=out_features, **linear_kwargs)
#         self.act = activation()
#         self.sum_layer = keras.layers.Add()
#         self.norm = norm
#         if self.norm:
#             self.norm_layer = PermBatchNorm()
#         else:
#             self.norm_layer = Identity()

#         self.norm_before_act = norm_before_act
        
#     def call(self, inputs):
#         x = inputs
#         x = self.sum_layer([self.L(x), self.G(x)])
        
#         if self.norm_before_act:
#             x = self.norm_layer(x)
#             x = self.act(x)
#         else:
#             x = self.act(x)
#             x = self.norm_layer(x)
#         return x

class BasicPermEquivariantBlock(keras.layers.Layer):
    def __init__(
        self,
        out_features=64,
        reduce_mode="max",
        norm=False,
        activation=tf.keras.layers.LeakyReLU,
        norm_before_act=False,
        **linear_kwargs
    ):
        super(BasicPermEquivariantBlock, self).__init__()

        self.L = keras.layers.Dense(out_features, **linear_kwargs)
        self.G = keras.layers.Dense(out_features, **linear_kwargs)
        self.act = activation()
        self.norm_before_act = norm_before_act

        if norm:
            self.norm = PermBatchNorm()
        else:
            self.norm = Identity()

        self.reduce_mode = reduce_mode
        self.reduce_func = {
            "max": lambda x: tf.reduce_max(x, axis=-2, keepdims=True),
            "mean": lambda x: tf.reduce_mean(x, axis=-2, keepdims=True),
        }[reduce_mode]

    def call(self, inputs):
        x = self.L(inputs) + self.reduce_func(self.G(inputs))

        if self.norm_before_act:
            x = self.norm(x)
            x = self.act(x)
        else:
            x = self.act(x)
            x = self.norm(x)

        return x

class PermEquivariantBlock(keras.layers.Layer):
    """A set of stacked BasicPermEquivBlocks

    input  ->  (Batch, n_objects, in_features,  D0, ..., Dn)
    output ->  (Batch, n_objects, out_features, D0, ..., Dn)

    """

    def __init__(
        self,
        out_features: int = 1,
        activation=keras.layers.LeakyReLU,
        norm: bool = False,
        norm_before_act: bool = True,
        use_bias: bool = True,
    ):
        super().__init__()

        self._block = BasicPermEquivariantBlock(
                    out_features=out_features,
                    norm=norm,
                    norm_before_act=norm_before_act,
                    activation=activation,
                    use_bias=use_bias,
                )
        
    def call(self, x):
        """
        input.shape   ->  (Batch, n_objects, D0, ..., Dn, in_features)
        output.shape  ->  (Batch, n_objects, D0, ..., Dn, out_features)
        """

        # permute axis such that perm equiv layers can operate on...
        # (Batch, n_objects, D0, ..., Dn, in_features) -> (Batch, D0, ..., Dn, n_objects, in_features)
        x = tf.experimental.numpy.moveaxis(x, 1, -2)

        x = self._block(x)

        # permute back
        x = tf.experimental.numpy.moveaxis(x, -2, 1)

        return x

class PermBatchNorm(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PermBatchNorm, self).__init__(**kwargs)
        self.bn = keras.layers.BatchNormalization()

    def call(self, inputs):
        shape = tf.shape(inputs)
        x = tf.reshape(inputs, (-1, shape[-1]))
        x = self.bn(x)
        x = tf.reshape(x, shape)
        return x