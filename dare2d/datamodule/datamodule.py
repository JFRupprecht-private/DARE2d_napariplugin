import tensorflow as tf
import numpy as np


class Datamodule:
    def __init__(self, train, val, test, batch_size) -> None:
        self.batch_size = batch_size
        self.set_train(train)
        self.set_val(val)
        self.set_test(test)

    def make_callable(self, generators, infinite):
        if not isinstance(generators, list):
            generators = [generators]

        def generator_fct():
            if infinite:
                while True:
                    for generator in generators:
                        X, y = next(generator)
                        yield X, y
            else:
                for generator in generators:
                    for idx in range(len(generator)):
                        X, y = generator[idx]
                        yield X, y

        return generator_fct

    def make_dataset(self, generators, batch_size=0, infinite=False):
        generator = generators
        if isinstance(generators, list):
            generator = generators[0]
        tfdataset = tf.data.Dataset.from_generator(
            self.make_callable(generators, infinite),
            output_types=generator.get_output_types(),
            output_shapes=generator.get_output_shapes(),
        )
        tfdataset = self.batch_dataset(tfdataset, batch_size)
        return tfdataset

    def batch_dataset(self, dataset, batch_size=0):
        if batch_size > 0:
            dataset = dataset.batch(
                batch_size=batch_size, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
            ).prefetch(tf.data.AUTOTUNE)
        return dataset

    def set_trains(self, train_generators, sample_limit=np.inf):
        self._train_generator, self.train = self.set_multiple(
            train_generators, is_training=True, sample_limit=sample_limit
        )

    def set_vals(self, val_generators):
        self._val_generator, self.val = self.set_multiple(val_generators)

    def set_tests(self, test_generators):
        self._test_generator, self.test = self.set_multiple(test_generators)

    def set_multiple(self, generators, is_training=False, sample_limit=np.inf):
        # Set data augmentation for training generators
        if is_training:
            for generator in generators:
                generator.set_augmentations(self.get_augmentations())
                generator.set_sample_limit(sample_limit)
        dataset = self.make_dataset(generators, batch_size=self.batch_size, infinite=is_training)
        return generators, dataset

    def set_single(self, generator, is_training=False, sample_limit=np.inf):
        if is_training:
            generator.set_augmentations(self.get_augmentations())
            generator.set_sample_limit(sample_limit)
        dataset = self.make_dataset(generator, self.batch_size, infinite=is_training)
        return generator, dataset

    def set_train(self, train_generator, sample_limit=np.inf):
        self._train_generator, self.train = self.set_single(
            train_generator, is_training=True, sample_limit=sample_limit
        )

    def set_val(self, val_generator):
        self._val_generator, self.val = self.set_single(val_generator)

    def set_test(self, test_generator):
        self._test_generator, self.test = self.set_single(test_generator)

    def get_augmentations(self):
        raise NotImplementedError()
