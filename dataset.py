"""
This module contains the data handling used in the model optimization.
"""
# === Imports

import os
import shutil
import tensorflow as tf


class Dataset:
    """
    A class that contains the data handling used in model optimization.
    """

    # === Download the IMDB dataset
    def __init__(self):
        """
        Initialize the Data class.
        """

        # url of stanford dataset
        url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

        # Download and untar the dataset
        dataset = tf.keras.utils.get_file(
            "aclImdb_v1.tar.gz", url, untar=True, cache_dir=".", cache_subdir=""
        )

        # Create the directory for the dataset, and its train and test data
        dataset_dir = os.path.join(os.path.dirname(dataset), "aclImdb")

        train_dir = os.path.join(dataset_dir, "train")
        test_dir = os.path.join(dataset_dir, "test")

        # remove unused folders to make it easier to load the data
        remove_dir = os.path.join(train_dir, "unsup")
        shutil.rmtree(remove_dir)

        # Set some parameters, AUTOTUNE is for the caching.
        buffer_size = tf.data.AUTOTUNE
        batch_size = 32
        seed = 42

        # Create the training data set.
        # All the data will be cached and prefetched. This should always be done.
        self.train = tf.keras.utils.text_dataset_from_directory(
            train_dir,
            batch_size=batch_size,
            validation_split=0.2,
            subset="training",
            seed=seed,
        )

        self.class_names = self.train.class_names
        self.train = self.train.cache().prefetch(buffer_size=buffer_size)

        # Create the validation dataset from the training data
        # Hm, it might be important to fix the seed to be the same as above.
        self.validation = tf.keras.utils.text_dataset_from_directory(
            train_dir,
            batch_size=batch_size,
            validation_split=0.2,
            subset="validation",
            seed=seed,
        )

        self.validation = self.validation.cache().prefetch(buffer_size=buffer_size)

        # Read in the test data. No splitting here.
        # Remark: Normally we would have to split train, test, and validate data
        #   from a single dataset. Here it was already split train, test...
        self.test = tf.keras.utils.text_dataset_from_directory(
            test_dir, batch_size=batch_size
        )

        self.test = self.test.cache().prefetch(buffer_size=buffer_size)

    def __str__(self):
        """
        Output string for the Data class.
        """
        return self.__class__.__name__

    # === Look at a few samples
    def look_at(self, samples=3):
        """
        Look at the first train samples.

        Keyword arguments:
        samples -- The number of samples (default 3)
        """

        for text_batch, label_batch in self.train.take(1):
            for i in range(samples):
                print(f"Review: {text_batch.numpy()[i]}")
                label = label_batch.numpy()[i]
                print(f"Label : {label} ({self.class_names[label]})")
