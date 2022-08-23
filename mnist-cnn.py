# Libraries for Machine Learning & Data Handling
import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Helper Libraries
import math
import numpy as np
import matplotlib.pyplot as plt

# Set up logging
import logging
logger = tf.getLogger()
logger.setLevel(logging.ERROR)



def main():
    # Import & Split Dataset
    dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
    train_ds, test_ds = dataset['train'], dataset['test']
    class_names = metadata.features['label'].names
    num_train_examples = metadata.splits['train'].num_examples
    num_test_examples = metadata.splits['test'].num_examples

