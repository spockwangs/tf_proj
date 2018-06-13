import tensorflow as tf
import os
from tqdm import tqdm
import numpy as np
from tf_models.base.logger import Logger

class BaseTrain:
    def __init__(self, sess, model, data, options):
        self.model = model
        self.logger = Logger(sess, options)
        self.options = options
        self.sess = sess
        self.data = data
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def train(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.options.num_epochs + 1, 1):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)
