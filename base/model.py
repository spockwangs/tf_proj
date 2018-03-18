import tensorflow as tf
import os

class BaseModel:
    def __init__(self, options):
        self.options = options
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()
        self.build_model()
        self.init_saver()

    # save function thet save the checkpoint in the path defined in configfile
    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, self.options.checkpoint_dir, self.global_step_tensor)
        print("Model saved")

    # load lateset checkpoint from the experiment path defined in config_file
    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.options.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")

    # just inialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(1, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # just inialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.options.max_to_keep)

    def build_model(self):
        raise NotImplementedError
