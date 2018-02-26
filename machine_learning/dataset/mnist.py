import os
from . import config

def load(data_path=None, one_hot=True):
    if data_path is None: data_path = os.path.join(config.DEFAULT_DATASET_PATH, 'mnist')
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets(data_path, one_hot=one_hot)
    return mnist
