from tf_models.base.data_set import BaseDataSet
import numpy as np
import tensorflow as tf

class DataSet(BaseDataSet):
    def __init__(self, options):
        super(DataSet, self).__init__(options)
        mnist = tf.contrib.learn.datasets.DATASETS['mnist']('/tmp/mnist')
        self.train_x = np.reshape(mnist.train.images, [-1, 28, 28, 1])
        self.train_y = np.eye(10)[mnist.train.labels]
        self.test_x = np.reshape(mnist.test.images, [-1, 28, 28, 1])
        self.test_y = np.eye(10)[mnist.test.labels]

    def next_batch(self, batch_size):
        idx = np.random.choice(self.train_x.shape[0], batch_size)
        yield self.train_x[idx, :, :, :], self.train_y[idx, :]
