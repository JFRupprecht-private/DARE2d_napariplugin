import os
import logging
import sys

import hydra
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf

from dare2d.io import Stdout2file

log = logging.getLogger(__name__)


class TrainingProcedure(object):
    def __init__(self, config) -> None:
        OmegaConf.resolve(config)
        self.config = config

    def init_datamodule(self):
        log.info(f"Instantiating datamodule <{self.config.datamodule._target_}>")
        self.datamodule = hydra.utils.instantiate(self.config.datamodule)

    def init_logging(self):
        # Init logging to file
        if self.config.get("log_file"):
            sys.stdout = Stdout2file(self.config.get("log_file"))

    def init_model(self):
        # Init model
        log.info(f"Instantiating model <{self.config.model._target_}>")

        if self.config.multi_gpu:
            self.strategy = tf.distribute.MirroredStrategy()
            print("Number of devices: {}".format(self.strategy.num_replicas_in_sync))

            # Open a strategy scope.
            with self.strategy.scope():
                self.model = hydra.utils.instantiate(self.config.model)
                self.model.compile()
        else:
            self.model = hydra.utils.instantiate(self.config.model)
            self.model.compile()
        self.model.model.summary()

    def init_callbacks(self):
        # Init callbacks
        self.callbacks = []
        if "callbacks" in self.config:
            for _, cb_conf in self.config.callbacks.items():
                if "_target_" in cb_conf:
                    log.info(f"Instantiating callback <{cb_conf._target_}>")
                    callback = hydra.utils.instantiate(cb_conf)
                    self.callbacks.append(callback)
        self.update_callbacks_parameters()

    def update_callbacks_parameters(self):
        for callback in self.callbacks:
            if hasattr(callback, "model"):
                callback.model = self.model.model
            if hasattr(callback, "model_handler"):
                callback.model_handler = self.model
            if hasattr(callback, "set_generator"):
                callback.set_generator(self.datamodule.val)
            if hasattr(callback, "mode"):
                callback.mode = "validation"

    def init_loggers(self):
        # Init loggers
        self.logger = []
        if "logger" in self.config:
            for _, lg_conf in self.config.logger.items():
                if "_target_" in lg_conf:
                    log.info(f"Instantiating logger <{lg_conf._target_}>")
                    self.logger.append(hydra.utils.instantiate(lg_conf))

    def init_trainer(self):
        # Init trainer
        log.info(f"Instantiating trainer <{self.config.trainer._target_}>")
        self.trainer = hydra.utils.instantiate(self.config.trainer)

    def train(self):
        # Train the model
        print(self.config["train"])
        if self.config.get("train"):
            log.info("Starting training!")
            self.trainer.fit(
                model_handler=self.model,
                datamodule=self.datamodule,
                callbacks=self.callbacks,
            )

    def get_metric_score(self):
        # Get metric score for hyperparameter optimization
        optimized_metric = self.config.get("optimized_metric")
        score = self.trainer.get_metric(optimized_metric)
        if optimized_metric and score is None:
            raise Exception(
                "Metric for hyperparameter optimization not found! "
                "Make sure the `optimized_metric` in `hparams_search` config is correct!"
                f"Optimized metric = {optimized_metric}"
                f"Valid metrics are = {self.trainer.get_metric_keys()}"
            )
        return score

    def test(self, checkpoint_folder="checkpoints"):
        # Test the model
        if self.config.get("test"):
            hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
            output_dir = hydra_cfg["runtime"]["output_dir"]

            ckpt_path = os.path.join(output_dir, checkpoint_folder, "best.h5")
            # If we did not train before
            # if not self.config.get("train"):
            # Set the loading path
            #    ckpt_path = self.config.trainer.get("resume_from_checkpoint")

            for callback in self.callbacks:
                if hasattr(callback, "set_generator"):
                    callback.set_generator(self.datamodule.test)
                if hasattr(callback, "mode"):
                    callback.mode = "test"

            log.info("Starting testing!")
            log.info(f"Checkpoint path: {ckpt_path}")
            self.trainer.test(
                model_handler=self.model,
                datamodule=self.datamodule,
                checkpoint=ckpt_path,
                callbacks=self.callbacks,
            )

    def init(self):
        self.init_logging()
        self.init_datamodule()
        self.init_model()
        self.init_callbacks()
        self.init_loggers()
        self.init_trainer()

    def run(self):
        self.init()
        self.train()
        self.test()

        self.score = self.get_metric_score()

        log.info(f"Final score : {self.score}")
        return self.score


def train(config: DictConfig):
    procedure = TrainingProcedure(config)
    return procedure.run()
