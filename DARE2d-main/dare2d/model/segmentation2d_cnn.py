""""""
from omegaconf import DictConfig

# fmt: off
import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm
# fmt: on

# see (https://github.com/qubvel/segmentation_models)


class Segmentation2dCNN:
    def __init__(self, input_channels, target_size, backbone, optimizer, losses, metrics, evaluations=None) -> None:
        self.model = sm.Unet(
            backbone_name=backbone,
            input_shape=(target_size, target_size, len(input_channels)),
            classes=1,
            activation="sigmoid",
            encoder_weights=None
        )

        self.optimizer = optimizer

        # we need to convert dict config dict to real python dict
        self.losses = self.dictConfig2dict(losses)
        self.metrics = self.dictConfig2dict(metrics)
        self.evaluations = self.dictConfig2dict(evaluations)

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
                           loss=self.losses, metrics=self.metrics)
