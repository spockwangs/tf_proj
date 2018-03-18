from ..base.data_set import BaseDataSet
import numpy as np
import tensorflow as tf

class DataSet(BaseDataSet):
    def __init__(self, options):
        super(DataSet, self).__init__(options)
        mnist = tf.contrib.learn.datasets.DATASETS['mnist']('/tmp/mnist')
        self.train_x = mnist.train.images
        self.train_y = mnist.train.labels.astype(np.int32)
        self.test_x = mnist.test.images
        self.test_y = mnist.test.labels.astype(np.int32)

    def next_batch(self, batch_size):
        yield self.train_x, self.train_y
