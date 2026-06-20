"""Main training script with an entry point for hydra."""

from typing import Optional

import hydra
from omegaconf import DictConfig
HYDRA_FULL_ERROR=1
#import tensorflow as tf
#import keras
#config = tf.ConfigProto( device_count = {'GPU': 2 , 'CPU': 12} )
#sess = tf.Session(config=config)
# keras.backend.set_session(sess)
# from utils import extras


@hydra.main(config_path="../../config/", config_name="train.yaml", version_base="1.1")
def main(config: DictConfig) -> Optional[float]:
    """Entry point for the main training function.

    It takes a training script that is used by hydra to setup the full configuration.

    Args:
        config (DictConfig): The full configuration for the experiment

    Returns:
        float: score for the best model. Used when doing hyperparameters optimisations
    """
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from training_pipeline import train

    # extras(config)

    # Train model
    return train(config)


if __name__ == "__main__":
    main()
