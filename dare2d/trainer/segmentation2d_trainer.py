import logging
import os

from dare2d.evaluation.evaluate_center2d import evaluate_center_detection
from dare2d.trainer.basic_trainer import Trainer

log = logging.getLogger(__name__)


class Segmentation2DTrainer(Trainer):
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
        max_distance_eval=10,
        validation_steps=None,
    ) -> None:
        super(Segmentation2DTrainer, self).__init__(
            epochs,
            workers,
            use_multiprocessing,
            steps_per_epoch,
            resume_from_checkpoint,
            verbose,
            experiment_name,
            tracking_uri,
            evaluations,
            validation_steps=validation_steps,
        )
        self.max_distance_eval = max_distance_eval

    def test(self, model_handler, datamodule, checkpoint=None, callbacks=None):
        if checkpoint and os.path.exists(checkpoint):
            log.info(f"Loading weights: {checkpoint}")
            model_handler.model.load_weights(checkpoint)

        generator = datamodule._test_generator
        if isinstance(generator, list):
            if len(generator) == 1:
                generator = generator[0]
            else:
                raise ValueError("Cannot evaluate with multipe generators")
        self.eval_history, _ = evaluate_center_detection(
            model_handler.model,
            generator,
            max_distance=self.max_distance_eval,
            display=False,
        )
        log.info(self.eval_history)

        with open("test_scores.json", "w") as scores:
            import json

            scores.write(json.dumps(self.eval_history))
