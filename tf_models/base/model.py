import tensorflow as tf
import os
from tqdm import tqdm
import numpy as np

class BaseModel:
    def __init__(self, options):
        self.options = options
        self.sess = tf.Session()
        # init the global step
        self.init_global_step()
        self.init_saver()
        self.build_model()
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)
        
    # save function thet save the checkpoint in the path defined in configfile
    def save(self):
        print("Saving model...")
        self.saver.save(self.sess, self.options.checkpoint_dir, self.global_step_tensor)
        print("Model saved")

    # load lateset checkpoint from the experiment path defined in config_file
    def load(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.options.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
            print("Model loaded")

    # just inialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.options.max_to_keep)

    def get_current_step(self):
        return self.model.global_step_tensor.eval(self.sess)
    
    def fit(self, data_set):
        for epoch in range(0, self.options.num_epochs, 1):
            self.train_epoch(data_set)
            
    def train_epoch(self, data_set):
        """
        implement the logic of epoch:
        -loop ever the number of iteration in the config and call the train step
        -add any summaries you want using the summary
        """
        loop = tqdm(range(self.options.num_iter_per_epoch))
        losses=[]
        for it in loop:
            x, y = next(data_set.next_batch(self.options.batch_size))
            cur_it, loss = self.train_step(x, y)
            losses.append(loss)
        loss = np.mean(losses)
        print("loss={loss}".format(loss=loss))
        self.save()

    def build_model(self):
        raise NotImplementedError

    def train_step(self, x, y):
        ''' Run current train step, returns current global step and computed loss.
        '''
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def score(self, x, y):
        raise NotImplementedError
    
