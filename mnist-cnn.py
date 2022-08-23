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