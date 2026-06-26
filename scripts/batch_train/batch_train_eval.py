import logging
import os
from typing import Optional

import hydra
import numpy as np
import tensorflow
import tensorflow as tf
from omegaconf import DictConfig

from scripts.tools.display_stats import display_results
from scripts.train.training_pipeline import TrainingProcedure

log = logging.getLogger(__name__)


class BatchTrainingProcedure(TrainingProcedure):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.init_generators()
        self.scores_path = self.config.batch_training.scores_path
        self.load_scores()

    def init(self):
        super().init()

    def init_generators(self):
        sets = self.config.batch_training.get("sets")
        default_generator = self.config.batch_training.generator
        self.sets = {}
        print(sets)
        print(default_generator)
        for set_name, set_path in sets.items():
            default_generator.data_folder = set_path

            log.info(f"Instantiating datamodule <{default_generator._target_}>")
            generator = hydra.utils.instantiate(default_generator)

            self.sets[set_name] = generator

    def prepare_data(self, training, val, test, samples_limit):
        self.datamodule.set_trains(training, samples_limit)
        self.datamodule.set_vals(val)
        self.datamodule.set_tests(test)

        self.update_callbacks_parameters()

    def set_subtree(self, keys, value):
        # Set metric values to regular float
        if isinstance(value, dict):
            for key in value.keys():
                value[key] = float(value[key])

        c_scores = self.scores
        for i, key in enumerate(keys):
            if key not in c_scores:
                c_scores[key] = {}

            # Last value
            if i == len(keys) - 1:
                c_scores[key] = value
            c_scores = c_scores[key]

    def in_subtree(self, keys):
        c_scores = self.scores
        for key in keys:
            if key not in c_scores:
                return False
            c_scores = c_scores[key]
        return True

    def write_scores(self):
        with open(self.scores_path, "w") as json_scores:
            import json

            json_str = json.dumps(self.scores)
            json_scores.write(json_str)
            json_scores.close()

    def load_scores(self):
        self.scores = {}
        if os.path.exists(self.scores_path):
            with open(self.scores_path, "r") as json_scores:
                import json

                self.scores = json.loads(json_scores.read())

    def add_score(self, keys):
        # score = self.get_metric_score()
        scores = self.trainer.eval_history
        if scores is None:
            scores = self.trainer.test_history
        self.set_subtree(keys, scores)
        log.info(f"Recording score {'_'.join(keys)} : {scores}")
        self.write_scores()

    def train_eval(
        self,
        keys,
        training_sets,
        validation_sets,
        test_set,
        sample_limit=np.inf,
        model_path=None,
    ):
        exp_name = "_".join(keys)
        if self.in_subtree(keys):
            log.info(f"Skipping training for {exp_name}")
            return

        self.init_model()
        if model_path:
            self.load_model(model_path)

        self.update_checkpoint_callback(exp_name)
        self.prepare_data(training_sets, validation_sets, test_set, sample_limit)
        self.train()
        self.test(checkpoint_folder=f"checkpoints_{exp_name}")
        self.add_score(keys)

    def update_checkpoint_callback(self, name):
        callbacks = list(
            filter(
                lambda a: not isinstance(a, tensorflow.keras.callbacks.ModelCheckpoint),
                self.callbacks,
            )
        )

        callbacks.append(
            tensorflow.keras.callbacks.ModelCheckpoint(
                filepath=f"checkpoints_{name}/best.h5",
                monitor="val_loss",
                save_best_only=True,
                mode="min",
                verbose=1,
            )
        )
        self.callbacks = callbacks

    def find_best_model_path(self, name):
        return f"checkpoints_{name}/best.h5"

    def load_model(self, path):
        log.info(f"Loading model weights from path : {path}")
        if self.config.multi_gpu:
            with self.strategy.scope():
                self.model.model.load_weights(path)
        else:
            self.model.model.load_weights(path)

    def run(self):
        """_summary_

        Multiple training sets with their associated validation sets
        train_val_sets = [
                        "set_1": {
                            "train": <generator>,
                            "val": <generator>
                        },
                        "set_2": {
                            "train": <generator>,
                            "val": <generator>
                        },
                        "set_3": {
                            "train": <generator>,
                            "val": <generator>
                        }]

        Returns:
            _type_: _description_
        """
        # 12 trainings * number of training sets
        # train on all combinaison of leave one out

        self.init()

        log.info(f"Starting multi training with {len(self.sets)} sets")

        for currindex, (target_set_name, target_set) in enumerate(self.sets.items()):
            # Get all training sets but the current one
            selected_sets = [
                [s, key]
                for i, (key, s) in enumerate(self.sets.items())
                if i != currindex
            ]
            training_sets, names = zip(*selected_sets)
            # Validation is the target set
            val_set = [target_set]
            keys = [target_set_name, "all_but_target"]

            # training on training_sets but target_set, eval on target_set
            log.info(f"Start training on all sets but set index {currindex}")
            log.info(f"Training sets size {[len(sset) for sset in training_sets]}")
            log.info(f"Test sets size {len(target_set)}")
            self.train_eval(keys, list(training_sets), val_set, test_set=val_set)

            # best_model_path = self.find_best_model_path('_'.join(keys))

            # step = self.config.batch_training.train_amount_step
            # max_amount = self.config.batch_training.train_amount_max
            # train_amount = list(
            #     np.arange(step, max_amount, step)) + [max_amount]

            # Start from the pretrained model and increase the number of training samples
            # for lim in train_amount:
            #     keys = [target_set_name, "limit", f"{lim}"]
            #     # Train on part of the target_train eval on target_val
            #     self.train_eval(keys, training_sets=[target_set["train"]], validation_sets=val_set,
            #                     test_set=val_set, sample_limit=lim, model_path=best_model_path)

        log.info(f"Scores : {self.scores}")
        return self.scores


@hydra.main(config_path="../../config/", config_name="train.yaml", version_base="1.1")
def main(config: DictConfig) -> Optional[float]:
    procedure = BatchTrainingProcedure(config)
    scores = procedure.run()

    # Save scores
    with open("scores.json", "w") as scores_file:
        import json

        scores_file.write(json.dumps(scores))

    # scores_str = '{"set_1": {"all": {"loss": 0.33685925602912903, "length_output_abs_mean": 8.281835556030273, "angle_output_angle_degree_abs_mean": 17.646303176879883}, "limit": {"0.25": {"loss": 0.18399514257907867, "length_output_abs_mean": 3.3567960262298584, "angle_output_angle_degree_abs_mean": 35.38542556762695}, "0.5": {"loss": 0.0946982130408287, "length_output_abs_mean": 1.9785619974136353, "angle_output_angle_degree_abs_mean": 31.222124099731445}, "0.75": {"loss": 0.4025690257549286, "length_output_abs_mean": 2.385124683380127, "angle_output_angle_degree_abs_mean": 40.09646987915039}, "1.0": {"loss": 0.13018180429935455, "length_output_abs_mean": 1.8825100660324097, "angle_output_angle_degree_abs_mean": 12.937235832214355}}, "solo": {"loss": 0.09931386262178421, "length_output_abs_mean": 1.3787223100662231, "angle_output_angle_degree_abs_mean": 7.45828914642334}}, "set_2": {"all": {"loss": 0.15341050922870636, "length_output_abs_mean": 4.904587268829346, "angle_output_angle_degree_abs_mean": 36.3009147644043}, "limit": {"0.25": {"loss": 0.11563550680875778, "length_output_abs_mean": 5.949746608734131, "angle_output_angle_degree_abs_mean": 37.262081146240234}, "0.5": {"loss": 0.06745067983865738, "length_output_abs_mean": 6.31516695022583, "angle_output_angle_degree_abs_mean": 34.92653274536133}, "0.75": {"loss": 0.08245033770799637, "length_output_abs_mean": 5.406310558319092, "angle_output_angle_degree_abs_mean": 7.514401912689209}, "1.0": {"loss": 0.13042767345905304, "length_output_abs_mean": 4.795921325683594, "angle_output_angle_degree_abs_mean": 7.66827392578125}}, "solo": {"loss": 0.09281597286462784, "length_output_abs_mean": 4.879356861114502, "angle_output_angle_degree_abs_mean": 37.217811584472656}}, "set_3": {"all": {"loss": 0.042847298085689545, "length_output_abs_mean": 4.0022077560424805, "angle_output_angle_degree_abs_mean": 3.79941987991333}, "limit": {"0.25": {"loss": 0.027062704786658287, "length_output_abs_mean": 3.4317626953125, "angle_output_angle_degree_abs_mean": 2.2994370460510254}, "0.5": {"loss": 0.023483719676733017, "length_output_abs_mean": 3.374295949935913, "angle_output_angle_degree_abs_mean": 2.6995980739593506}, "0.75": {"loss": 0.024715576320886612, "length_output_abs_mean": 3.3830320835113525, "angle_output_angle_degree_abs_mean": 3.387287139892578}, "1.0": {"loss": 0.027981434017419815, "length_output_abs_mean": 3.056091785430908, "angle_output_angle_degree_abs_mean": 2.06671404838562}}, "solo": {"loss": 0.027220310643315315, "length_output_abs_mean": 2.8992114067077637, "angle_output_angle_degree_abs_mean": 2.5194857120513916}}}'
    # import json
    # scores = json.loads(scores_str)
    # display_results("scores.json", "scores")


if __name__ == "__main__":
    main()
