import os

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

import logging

log = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        epochs=1,
        workers=1,
        use_multiprocessing=False,
        steps_per_epoch=100,
        resume_from_checkpoint=None,
        verbose=1,
        experiment_name=None,
        tracking_uri=None,
        evaluations=None,
        validation_steps=None,
        class_weight=None,
    ) -> None:
        self.epochs = epochs
        self.workers = workers
        self.use_multiprocessing = use_multiprocessing
        self.steps_per_epoch = steps_per_epoch
        self.resume_from_checkpoint = resume_from_checkpoint
        self.verbose = verbose
        self.history = None
        self.test_history = None
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.eval_history = None
        self.validation_steps = validation_steps
        self.class_weight = None
        # if self.tracking_uri:
        #     mlflow.set_tracking_uri(self.tracking_uri)
        # if self.experiment_name:
        #     mlflow.set_experiment(self.experiment_name)

    def fit(self, model_handler, datamodule, callbacks):
        if self.resume_from_checkpoint:
            model_handler.model.load_weights(self.resume_from_checkpoint)
            log.info(f"Weights loaded from : {self.resume_from_checkpoint}")

        model_handler.compile()

        # Enable mlflow
        # mlflow.keras.autolog()

        val = datamodule.val
        if self.validation_steps:
            val = val.repeat()

        self.history = model_handler.model.fit(
            datamodule.train,
            validation_data=val,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            callbacks=callbacks,
            verbose=self.verbose,
            workers=self.workers,
            validation_steps=self.validation_steps,
            use_multiprocessing=self.use_multiprocessing,
            class_weight=self.class_weight,
        )

        return self.history

    def test(self, model_handler, datamodule, checkpoint=None, callbacks=None):
        if checkpoint and os.path.exists(checkpoint):
            model_handler.model.load_weights(checkpoint)

        self.test_history = model_handler.model.evaluate(
            datamodule.test, callbacks=callbacks, return_dict=True
        )

        with open("test_scores.json", "w") as scores:
            import json

            scores.write(json.dumps(self.test_history))

        if model_handler.evaluations:
            # Predict on test
            log.info("Performing prediction on test set")
            predictions, groundtruth = self._predict(
                model_handler.model, datamodule.test
            )

            # Apply evaluations
            log.info("Started evaluations on prediction results from test set")
            self.eval_history = self._evaluate_predictions(
                predictions, groundtruth, model_handler.evaluations
            )

            with open("eval_scores.json", "w") as scores:
                import json

                scores.write(json.dumps(self.eval_history))

    def _predict(self, model, generator):
        predictions = []
        groundtruth = []
        for x, y_true in generator:
            y_pred = model.predict(x)
            predictions.append(y_pred)
            groundtruth.append(y_true)
        return np.concatenate(predictions, axis=0), np.concatenate(groundtruth, axis=0)

    def _evaluate_predictions(self, y_pred, y_true, fcnts):
        history = {}
        if isinstance(fcnts, dict):
            for fcnt_name, fcnt in fcnts.items():
                log.info(f"Evaluating with function {fcnt_name}...")
                measures = []
                for i in tqdm(range(y_pred.shape[0])):
                    measure = fcnt(y_true[i], y_pred[i])
                    if measure:
                        measures.append(measure)
                history[fcnt_name] = np.mean(measures)
        else:
            measures = []
            fcnt = fcnts
            fcnt_name = "eval_metric"
            for i in tqdm(range(y_pred.shape[0])):
                measure = fcnt(y_true[i], y_pred[i])
                if measure:
                    measures.append(measure)
            history[fcnt_name] = np.mean(measures)
        return history

    def get_test_metric(self, metric):
        return self._get_metric(metric, self.test_history)

    def _get_metric(self, metric_name, history):
        if history:
            if metric_name in history:
                return history[metric_name]
        return None

    def get_metric(self, metric_name):
        if self.history:
            print(self.history.history)
            if metric_name in self.history.history:
                return self.history.history[metric_name][-1]
        if self.test_history:
            if metric_name in self.test_history:
                return self.test_history[metric_name]
        if self.eval_history:
            if metric_name in self.eval_history:
                return self.eval_history[metric_name]
        return None

    def get_metric_keys(self):
        keys = []
        if self.history:
            keys += list(self.history.history.keys())
        if self.test_history:
            keys += list(self.test_history.keys())
        if self.eval_history:
            keys += list(self.eval_history.keys())
        return keys

    def dictConfig2dict(self, dc):
        new_dict = dc
        if isinstance(dc, DictConfig):
            new_dict = {}
            for key, v in dc.items():
                new_dict[key] = v
        return new_dict
